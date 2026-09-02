"""
Orquestador principal del pipeline MLB Proyección v2.0.
Coordina: ingesta -> features -> modelo -> evaluación -> ejecución -> reportes.
Integración: SharpAPI (agregador de cuotas)
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from utils.config_loader import ConfigLoader
from utils.logger import StructuredLogger
from data_ingestion.mlb_statsapi import MLBStatsAPI
from data_ingestion.odds_api import OddsAPI
from features.engineering import MLBFeatureEngineer
from models.xgboost_conformal import XGBoostConformalPredictor
from evaluation.ev_calculator import EVCalculator
from execution.bankroll_manager import BankrollManager


class MLBPredictionPipeline:
    """
    Pipeline end-to-end para predicción y ejecución de apuestas MLB.
    Fuente de cuotas: SharpAPI (FanDuel, DraftKings) + fallbacks
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigLoader(config_path)
        self.logger = StructuredLogger(
            "mlb_pipeline", 
            log_dir="logs"
        )
        
        self.stats_api = MLBStatsAPI(
            cache_dir=self.config.get('data', 'paths', 'raw', default="data/raw")
        )
        self.odds_api = OddsAPI(
            cache_dir=self.config.get('data', 'paths', 'raw', default="data/raw") + "/odds"
        )
        self.feature_engineer = MLBFeatureEngineer(
            data_dir=self.config.get('data', 'paths', 'processed', default="data/processed")
        )
        self.model = XGBoostConformalPredictor(self.config.config)
        self.ev_calculator = EVCalculator(
            min_ev_threshold=self.config.min_ev_threshold,
            use_conservative=True
        )
        self.bankroll = BankrollManager(
            initial_bankroll=self.config.get('bankroll', 'initial', default=10000.0),
            kelly_fraction=self.config.kelly_fraction,
            max_stake_percent=self.config.max_stake_percent,
            max_daily_risk=self.config.max_daily_risk,
            max_picks_per_day=self.config.get('bankroll', 'daily_limits', 'max_picks', default=3),
            min_stake=self.config.get('bankroll', 'kelly', 'min_stake', default=10.0)
        )
        
        self.logger.info("Pipeline inicializado", {
            "season": self.config.season,
            "min_ev": self.config.min_ev_threshold,
            "kelly_fraction": self.config.kelly_fraction,
            "odds_source": "sharpapi"
        })
    
    def run_training(self, historical_games_path: Optional[str] = None):
        """Entrena el modelo con datos históricos."""
        self.logger.info("Iniciando entrenamiento del modelo")
        
        if historical_games_path and Path(historical_games_path).exists():
            df_games = pd.read_csv(historical_games_path, parse_dates=['date'])
        else:
            self.logger.info("Generando datos históricos desde StatsAPI...")
            start = f"{self.config.season}-04-01"
            end = self.config.val_end
            df_games = self.stats_api.get_schedule(start, end)
            df_games['target'] = 1
        
        self.logger.info("Generando features...")
        pitcher_stats = pd.DataFrame()
        bullpen_stats = pd.DataFrame()
        team_stats = pd.DataFrame()
        park_factors = pd.DataFrame()
        
        df_features = self.feature_engineer.build_features(
            df_games, pitcher_stats, bullpen_stats, team_stats, park_factors
        )
        
        self.logger.info("Entrenando XGBoost + Conformal...")
        metrics = self.model.train(
            df_features,
            train_end=self.config.train_end,
            val_end=self.config.val_end
        )
        
        self.logger.info("Entrenamiento completado", metrics)
        
        model_path = f"models/xgboost_conformal_{self.config.season}.joblib"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        
        return metrics
    
    def run_daily_prediction(self, date: Optional[str] = None):
        """Ejecuta pipeline diario con SharpAPI como fuente de cuotas."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Iniciando predicción diaria: {date}")
        self.bankroll.reset_daily()
        
        self.logger.info("Obteniendo calendario del día...")
        df_games = self.stats_api.get_schedule(date, date)
        
        if len(df_games) == 0:
            self.logger.info("No hay juegos programados para hoy")
            self._generate_no_pick_report(date, "No hay juegos programados")
            return []
        
        self.logger.info(f"Juegos encontrados: {len(df_games)}")
        
        self.logger.info("Generando features...")
        pitcher_stats = pd.DataFrame()
        bullpen_stats = pd.DataFrame()
        team_stats = pd.DataFrame()
        park_factors = pd.DataFrame()
        
        df_features = self.feature_engineer.build_features(
            df_games, pitcher_stats, bullpen_stats, team_stats, park_factors
        )
        
        self.logger.info("Generando predicciones con intervalos...")
        if not self.model.is_trained:
            model_path = f"models/xgboost_conformal_{self.config.season}.joblib"
            if Path(model_path).exists():
                self.model.load(model_path)
            else:
                raise ValueError("Modelo no entrenado. Ejecutar run_training() primero.")
        
        predictions = self.model.predict_dataframe(df_features)
        
        self.logger.info("Obteniendo cuotas desde SharpAPI...")
        try:
            odds_df = self.odds_api.get_odds(config=self.config.config)
            self.logger.info(f"Cuotas obtenidas: {len(odds_df)} registros agregados")
        except Exception as e:
            self.logger.error(f"Error obteniendo cuotas: {e}")
            self._generate_no_pick_report(date, f"Error de cuotas: {e}")
            return []
        
        self.logger.info("Evaluando EV...")
        evaluations = self.ev_calculator.evaluate_batch(predictions, odds_df)
        
        executed_bets = []
        rejected_bets = []
        
        for _, eval_row in evaluations.iterrows():
            if eval_row['decision'] == 'EXECUTE':
                if len(executed_bets) >= self.config.get('bankroll', 'daily_limits', 'max_picks', default=3):
                    self.logger.warning(f"Límite diario de picks alcanzado. Rechazando {eval_row['game_id']}")
                    continue
                
                bet = self.bankroll.place_bet(
                    game_id=eval_row['game_id'],
                    pick=eval_row['pick'],
                    odds=eval_row['odds_decimal'],
                    prob=eval_row['prob_lower'] if self.config.get('evaluation', 'use_conservative_ev') 
                        else eval_row['prob_point'],
                    ev=eval_row['ev_conservative'],
                    confidence=eval_row['confidence_score']
                )
                if bet:
                    executed_bets.append(bet)
            else:
                rejected_bets.append(eval_row)
                self.logger.bet_decision(
                    eval_row['game_id'], "REJECT", 
                    eval_row['ev_point'], eval_row['prob_point'],
                    eval_row['prob_lower'], eval_row['odds_decimal'],
                    0.0, eval_row['rejection_reason']
                )
        
        self._generate_daily_report(date, executed_bets, rejected_bets, predictions, evaluations, odds_df)
        
        self.logger.info(f"Pipeline completado. Picks ejecutados: {len(executed_bets)}")
        
        return executed_bets
    
    def _generate_daily_report(self, date: str, executed: List, rejected: List,
                               predictions: pd.DataFrame, evaluations: pd.DataFrame,
                               odds_df: pd.DataFrame):
        """Genera reporte diario estructurado."""
        report = {
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "total_games": len(predictions),
            "picks_executed": len(executed),
            "picks_rejected": len(rejected),
            "total_markets_evaluated": len(evaluations['market_type'].unique()) if len(evaluations) > 0 else 0,
            "bankroll_current": round(self.bankroll.current_bankroll, 2),
            "daily_risk": round(self.bankroll.daily_risk, 2),
            "odds_source": "sharpapi",
            "executed_bets": [
                {
                    "game_id": b.game_id,
                    "pick": b.pick,
                    "odds": b.odds_decimal,
                    "stake": b.stake,
                    "ev": b.ev,
                    "confidence": b.confidence
                } for b in executed
            ],
            "rejected_evaluations": [
                {
                    "game_id": r['game_id'],
                    "pick": r['pick'],
                    "market_type": r.get('market_type', 'unknown'),
                    "reason": r['rejection_reason'],
                    "ev_conservative": r['ev_conservative'],
                    "prob_lower": r['prob_lower']
                } for _, r in pd.DataFrame(rejected).iterrows()
            ] if rejected else [],
            "available_markets": odds_df['market_type'].unique().tolist() if len(odds_df) > 0 else [],
            "all_predictions": predictions.to_dict('records'),
            "all_evaluations": evaluations.to_dict('records')
        }
        
        report_dir = Path(self.config.get('reporting', 'daily_report', 'path', default="reports/daily"))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"report_{date}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Reporte guardado: {report_path}")
    
    def _generate_no_pick_report(self, date: str, reason: str):
        """Genera reporte para días sin picks."""
        report = {
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "total_games": 0,
            "picks_executed": 0,
            "picks_rejected": 0,
            "no_pick_reason": reason,
            "bankroll_current": round(self.bankroll.current_bankroll, 2),
            "executed_bets": [],
            "rejected_evaluations": [],
            "available_markets": [],
            "all_predictions": [],
            "all_evaluations": []
        }
        
        report_dir = Path(self.config.get('reporting', 'daily_report', 'path', default="reports/daily"))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"report_{date}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Reporte de NO PICK guardado: {report_path} | Razón: {reason}")


if __name__ == "__main__":
    pipeline = MLBPredictionPipeline()
    bets = pipeline.run_daily_prediction("2026-09-01")
    print(f"Apuestas ejecutadas hoy: {len(bets)}")
