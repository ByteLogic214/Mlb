"""
Feature Engineering para MLB.
Genera features estructuradas desde datos crudos, sin sesgos hardcodeados.
Soporta temporalidad estricta (no leakage del futuro).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MLBFeatureEngineer:
    """
    Genera features para modelado predictivo de MLB.
    CRÍTICO: Todas las features se calculan con datos disponibles HASTA la fecha del juego.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.recent_games = 10
        self.medium_games = 30
    
    def _get_prior_games(self, df_games: pd.DataFrame, team_id: int, 
                         game_date: str, n: int) -> pd.DataFrame:
        """Obtiene los N juegos previos de un equipo antes de una fecha dada."""
        prior = df_games[
            ((df_games["home_team_id"] == team_id) | 
             (df_games["away_team_id"] == team_id)) &
            (df_games["date"] < game_date) &
            (df_games["date"] >= pd.to_datetime(game_date) - timedelta(days=60))
        ].sort_values("date", ascending=False).head(n)
        return prior
    
    def _calculate_pitcher_features(self, pitcher_id: int, season: int,
                                    game_date: str, 
                                    pitcher_stats_df: pd.DataFrame) -> Dict:
        """Features de pitcher basadas en datos históricos hasta la fecha."""
        stats = pitcher_stats_df[
            (pitcher_stats_df["pitcher_id"] == pitcher_id) &
            (pitcher_stats_df["date"] < game_date)
        ].sort_values("date", ascending=False)
        
        if len(stats) == 0:
            return self._league_average_pitcher()
        
        recent = stats.head(5)
        older = stats.iloc[5:15] if len(stats) > 5 else stats
        
        def weighted_avg(col: str) -> float:
            r = recent[col].mean() if len(recent) > 0 else 0
            o = older[col].mean() if len(older) > 0 else r
            return 0.7 * r + 0.3 * o
        
        return {
            "pitcher_era": weighted_avg("era"),
            "pitcher_fip": weighted_avg("fip"),
            "pitcher_xera": weighted_avg("xera"),
            "pitcher_k_rate": weighted_avg("strikeouts") / max(weighted_avg("batters_faced"), 1),
            "pitcher_bb_rate": weighted_avg("base_on_balls") / max(weighted_avg("batters_faced"), 1),
            "pitcher_whiff_rate": weighted_avg("whiff_rate"),
            "pitcher_hard_hit": weighted_avg("hard_hit_rate"),
            "pitcher_innings_avg": weighted_avg("innings_pitched"),
            "pitcher_rest_days": self._calculate_rest_days(pitcher_id, game_date, stats),
        }
    
    def _calculate_bullpen_features(self, team_id: int, game_date: str,
                                    bullpen_stats_df: pd.DataFrame) -> Dict:
        """Features del bullpen (últimos 14 días)."""
        recent = bullpen_stats_df[
            (bullpen_stats_df["team_id"] == team_id) &
            (bullpen_stats_df["date"] < game_date) &
            (bullpen_stats_df["date"] >= pd.to_datetime(game_date) - timedelta(days=14))
        ]
        
        if len(recent) == 0:
            return self._league_average_bullpen()
        
        return {
            "bullpen_fip": recent["fip"].mean(),
            "bullpen_inherited_scored": recent["inherited_runners_scored"].mean(),
            "bullpen_k_rate": recent["strikeouts"].sum() / max(recent["batters_faced"].sum(), 1),
            "bullpen_walk_rate": recent["base_on_balls"].sum() / max(recent["batters_faced"].sum(), 1),
            "bullpen_avg_leverage": recent["avg_leverage_index"].mean(),
            "bullpen_days_rest": (pd.to_datetime(game_date) - recent["date"].max()).days,
        }
    
    def _calculate_offense_features(self, team_id: int, game_date: str,
                                    team_stats_df: pd.DataFrame) -> Dict:
        """Features ofensivas del equipo (últimos 10 juegos)."""
        recent = team_stats_df[
            (team_stats_df["team_id"] == team_id) &
            (team_stats_df["date"] < game_date) &
            (team_stats_df["date"] >= pd.to_datetime(game_date) - timedelta(days=30))
        ].sort_values("date", ascending=False).head(10)
        
        if len(recent) == 0:
            return self._league_average_offense()
        
        return {
            "off_woba": recent["woba"].mean(),
            "off_ops": recent["ops"].mean(),
            "off_iso": recent["iso"].mean(),
            "off_wrc_plus": recent["wrc_plus"].mean(),
            "off_k_rate": recent["strikeouts"].sum() / max(recent["plate_appearances"].sum(), 1),
            "off_bb_rate": recent["base_on_balls"].sum() / max(recent["plate_appearances"].sum(), 1),
            "off_babip": recent["babip"].mean(),
            "off_runs_per_game": recent["runs"].mean(),
        }
    
    def _calculate_defense_features(self, team_id: int, game_date: str,
                                    team_stats_df: pd.DataFrame) -> Dict:
        """Features defensivas (últimos 10 juegos)."""
        recent = team_stats_df[
            (team_stats_df["team_id"] == team_id) &
            (team_stats_df["date"] < game_date) &
            (team_stats_df["date"] >= pd.to_datetime(game_date) - timedelta(days=30))
        ].sort_values("date", ascending=False).head(10)
        
        if len(recent) == 0:
            return self._league_average_defense()
        
        return {
            "def_drs": recent["drs"].mean(),
            "def_uzr": recent["uzr"].mean(),
            "def_errors_per_game": recent["errors"].mean(),
            "def_double_plays": recent["double_plays"].mean(),
            "def_stolen_base_allowed": recent["stolen_bases_allowed"].mean(),
            "def_caught_stealing": recent["caught_stealing"].mean(),
        }
    
    def _get_park_factor(self, team_id: int, season: int,
                         park_factors_df: pd.DataFrame) -> float:
        """Park factor ajustado por temporada (NO boost fijo)."""
        pf = park_factors_df[
            (park_factors_df["team_id"] == team_id) &
            (park_factors_df["season"] == season)
        ]
        if len(pf) == 0:
            return 1.0
        return pf["park_factor"].values[0]
    
    def _calculate_rest_days(self, team_id: int, game_date: str,
                             df_games: pd.DataFrame) -> int:
        """Días de descanso del equipo."""
        prior = df_games[
            ((df_games["home_team_id"] == team_id) | 
             (df_games["away_team_id"] == team_id)) &
            (df_games["date"] < game_date)
        ].sort_values("date", ascending=False).head(1)
        
        if len(prior) == 0:
            return 3
        
        last_game = pd.to_datetime(prior["date"].values[0])
        current = pd.to_datetime(game_date)
        return (current - last_game).days - 1
    
    def _league_average_pitcher(self) -> Dict:
        """Promedios de liga para pitchers sin historial."""
        return {
            "pitcher_era": 4.50, "pitcher_fip": 4.50, "pitcher_xera": 4.50,
            "pitcher_k_rate": 0.22, "pitcher_bb_rate": 0.08,
            "pitcher_whiff_rate": 0.10, "pitcher_hard_hit": 0.35,
            "pitcher_innings_avg": 5.0, "pitcher_rest_days": 4,
        }
    
    def _league_average_bullpen(self) -> Dict:
        return {
            "bullpen_fip": 4.20, "bullpen_inherited_scored": 0.30,
            "bullpen_k_rate": 0.23, "bullpen_walk_rate": 0.09,
            "bullpen_avg_leverage": 1.0, "bullpen_days_rest": 1,
        }
    
    def _league_average_offense(self) -> Dict:
        return {
            "off_woba": 0.320, "off_ops": 0.750, "off_iso": 0.150,
            "off_wrc_plus": 100, "off_k_rate": 0.22, "off_bb_rate": 0.08,
            "off_babip": 0.300, "off_runs_per_game": 4.5,
        }
    
    def _league_average_defense(self) -> Dict:
        return {
            "def_drs": 0, "def_uzr": 0, "def_errors_per_game": 0.5,
            "def_double_plays": 0.8, "def_stolen_base_allowed": 0.3,
            "def_caught_stealing": 0.2,
        }
    
    def build_features(self, games_df: pd.DataFrame, 
                       pitcher_stats: pd.DataFrame,
                       bullpen_stats: pd.DataFrame,
                       team_stats: pd.DataFrame,
                       park_factors: pd.DataFrame) -> pd.DataFrame:
        """
        Construye el dataset completo de features.
        CRÍTICO: Cada fila representa un juego con features calculadas
        únicamente con información disponible antes del juego.
        """
        features_list = []
        
        for _, game in games_df.iterrows():
            game_date = game["date"]
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]
            home_pitcher = game.get("home_pitcher_id")
            away_pitcher = game.get("away_pitcher_id")
            
            home_pitcher_feat = self._calculate_pitcher_features(
                home_pitcher, 2026, game_date, pitcher_stats
            ) if pd.notna(home_pitcher) else self._league_average_pitcher()
            
            home_bullpen_feat = self._calculate_bullpen_features(
                home_id, game_date, bullpen_stats
            )
            home_offense_feat = self._calculate_offense_features(
                home_id, game_date, team_stats
            )
            home_defense_feat = self._calculate_defense_features(
                home_id, game_date, team_stats
            )
            home_park = self._get_park_factor(home_id, 2026, park_factors)
            home_rest = self._calculate_rest_days(home_id, game_date, games_df)
            
            away_pitcher_feat = self._calculate_pitcher_features(
                away_pitcher, 2026, game_date, pitcher_stats
            ) if pd.notna(away_pitcher) else self._league_average_pitcher()
            
            away_bullpen_feat = self._calculate_bullpen_features(
                away_id, game_date, bullpen_stats
            )
            away_offense_feat = self._calculate_offense_features(
                away_id, game_date, team_stats
            )
            away_defense_feat = self._calculate_defense_features(
                away_id, game_date, team_stats
            )
            away_rest = self._calculate_rest_days(away_id, game_date, games_df)
            
            feature_row = {
                "game_id": game["game_id"],
                "date": game_date,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                
                "diff_pitcher_era": home_pitcher_feat["pitcher_era"] - away_pitcher_feat["pitcher_era"],
                "diff_pitcher_fip": home_pitcher_feat["pitcher_fip"] - away_pitcher_feat["pitcher_fip"],
                "diff_pitcher_xera": home_pitcher_feat["pitcher_xera"] - away_pitcher_feat["pitcher_xera"],
                "diff_pitcher_k_rate": home_pitcher_feat["pitcher_k_rate"] - away_pitcher_feat["pitcher_k_rate"],
                "diff_pitcher_bb_rate": home_pitcher_feat["pitcher_bb_rate"] - away_pitcher_feat["pitcher_bb_rate"],
                "diff_pitcher_whiff": home_pitcher_feat["pitcher_whiff_rate"] - away_pitcher_feat["pitcher_whiff_rate"],
                "diff_pitcher_hard_hit": home_pitcher_feat["pitcher_hard_hit"] - away_pitcher_feat["pitcher_hard_hit"],
                "diff_pitcher_rest": home_pitcher_feat["pitcher_rest_days"] - away_pitcher_feat["pitcher_rest_days"],
                
                "diff_bullpen_fip": home_bullpen_feat["bullpen_fip"] - away_bullpen_feat["bullpen_fip"],
                "diff_bullpen_inherited": home_bullpen_feat["bullpen_inherited_scored"] - away_bullpen_feat["bullpen_inherited_scored"],
                "diff_bullpen_k_rate": home_bullpen_feat["bullpen_k_rate"] - away_bullpen_feat["bullpen_k_rate"],
                "diff_bullpen_walk_rate": home_bullpen_feat["bullpen_walk_rate"] - away_bullpen_feat["bullpen_walk_rate"],
                
                "diff_off_woba": home_offense_feat["off_woba"] - away_offense_feat["off_woba"],
                "diff_off_ops": home_offense_feat["off_ops"] - away_offense_feat["off_ops"],
                "diff_off_iso": home_offense_feat["off_iso"] - away_offense_feat["off_iso"],
                "diff_off_wrc": home_offense_feat["off_wrc_plus"] - away_offense_feat["off_wrc_plus"],
                "diff_off_k_rate": home_offense_feat["off_k_rate"] - away_offense_feat["off_k_rate"],
                "diff_off_bb_rate": home_offense_feat["off_bb_rate"] - away_offense_feat["off_bb_rate"],
                "diff_off_rpg": home_offense_feat["off_runs_per_game"] - away_offense_feat["off_runs_per_game"],
                
                "diff_def_drs": home_defense_feat["def_drs"] - away_defense_feat["def_drs"],
                "diff_def_uzr": home_defense_feat["def_uzr"] - away_defense_feat["def_uzr"],
                "diff_def_errors": home_defense_feat["def_errors_per_game"] - away_defense_feat["def_errors_per_game"],
                
                "park_factor": home_park,
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "rest_diff": home_rest - away_rest,
            }
            
            features_list.append(feature_row)
        
        return pd.DataFrame(features_list)
