"""
Modelo XGBoost con Conformal Prediction para intervalos calibrados.
Solo modelo predictivo autorizado en el pipeline v2.0.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score


class XGBoostConformalPredictor:
    """
    XGBoost con Split Conformal Prediction.
    Genera probabilidades puntuales + intervalos de predicción calibrados.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.calibration_scores = None
        self.calibration_quantile = None
        self.feature_names = None
        self.is_trained = False
        
        self.hyperparams = config.get('model', {}).get('hyperparameters', {})
        self.alpha = config.get('model', {}).get('conformal', {}).get('alpha', 0.05)
        self.calibration_size = config.get('model', {}).get('conformal', {}).get('calibration_size', 0.2)
    
    def _split_temporal(self, df: pd.DataFrame, train_end: str, val_end: str) -> Tuple:
        """Split estrictamente temporal. CRÍTICO: sin leakage."""
        train = df[df['date'] <= train_end].copy()
        val = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
        test = df[df['date'] > val_end].copy()
        return train, val, test
    
    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepara features y target."""
        exclude_cols = ['game_id', 'date', 'home_team_id', 'away_team_id', 
                        'home_team', 'away_team', 'home_win']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and c != 'target']
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['target'].values if 'target' in df.columns else None
        
        return X, y, feature_cols
    
    def train(self, df: pd.DataFrame, train_end: str, val_end: str) -> Dict:
        """Entrena modelo con validación temporal y calibración conformal."""
        train_df, val_df, test_df = self._split_temporal(df, train_end, val_end)
        
        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        
        n_cal = int(len(train_df) * self.calibration_size)
        cal_df = train_df.tail(n_cal).copy()
        train_df = train_df.iloc[:-n_cal].copy()
        
        X_train, y_train, _ = self._prepare_xy(train_df)
        X_cal, y_cal, _ = self._prepare_xy(cal_df)
        X_val, y_val, _ = self._prepare_xy(val_df)
        X_test, y_test, _ = self._prepare_xy(test_df)
        
        self.model = xgb.XGBClassifier(
            objective=self.hyperparams.get('objective', 'binary:logistic'),
            eval_metric=self.hyperparams.get('eval_metric', 'logloss'),
            max_depth=self.hyperparams.get('max_depth', 5),
            learning_rate=self.hyperparams.get('learning_rate', 0.05),
            n_estimators=self.hyperparams.get('n_estimators', 500),
            subsample=self.hyperparams.get('subsample', 0.8),
            colsample_bytree=self.hyperparams.get('colsample_bytree', 0.8),
            min_child_weight=self.hyperparams.get('min_child_weight', 5),
            gamma=self.hyperparams.get('gamma', 0.1),
            reg_alpha=self.hyperparams.get('reg_alpha', 0.1),
            reg_lambda=self.hyperparams.get('reg_lambda', 1.0),
            early_stopping_rounds=self.hyperparams.get('early_stopping_rounds', 50),
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calibración Conformal
        cal_probs = self.model.predict_proba(X_cal)[:, 1]
        cal_errors = np.abs(y_cal - cal_probs)
        
        n_cal = len(cal_errors)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)
        self.calibration_quantile = np.quantile(cal_errors, q_level)
        
        self.calibration_scores = cal_errors
        self.is_trained = True
        
        metrics = self._evaluate(X_test, y_test)
        metrics['calibration_size'] = n_cal
        metrics['calibration_quantile'] = float(self.calibration_quantile)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predice con intervalo de predicción conformal."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        point_probs = self.model.predict_proba(X)[:, 1]
        
        lower_bounds = np.maximum(point_probs - self.calibration_quantile, 0.0)
        upper_bounds = np.minimum(point_probs + self.calibration_quantile, 1.0)
        
        return point_probs, lower_bounds, upper_bounds
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predice y devuelve DataFrame con intervalos."""
        X, _, _ = self._prepare_xy(df)
        point, lower, upper = self.predict(X)
        
        result = df[['game_id', 'date', 'home_team', 'away_team']].copy()
        result['prob_home_win'] = point
        result['prob_home_win_lower'] = lower
        result['prob_home_win_upper'] = upper
        result['interval_width'] = upper - lower
        
        return result
    
    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evalúa métricas en test set."""
        point_probs, lower, upper = self.predict(X_test)
        
        logloss = log_loss(y_test, point_probs)
        brier = brier_score_loss(y_test, point_probs)
        auc = roc_auc_score(y_test, point_probs)
        
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        avg_width = np.mean(upper - lower)
        
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(point_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_error = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_pred = np.mean(point_probs[mask])
                avg_true = np.mean(y_test[mask])
                calibration_error += np.abs(avg_pred - avg_true) * np.sum(mask)
        
        calibration_error /= len(y_test)
        
        return {
            'log_loss': round(logloss, 4),
            'brier_score': round(brier, 4),
            'roc_auc': round(auc, 4),
            'coverage': round(coverage, 4),
            'avg_interval_width': round(avg_width, 4),
            'calibration_error': round(calibration_error, 4),
            'n_test': len(y_test)
        }
    
    def save(self, path: str):
        """Guarda modelo y calibración."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'calibration_quantile': self.calibration_quantile,
            'calibration_scores': self.calibration_scores,
            'feature_names': self.feature_names,
            'config': self.config,
            'alpha': self.alpha
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load(self, path: str):
        """Carga modelo y calibración."""
        data = joblib.load(path)
        self.model = data['model']
        self.calibration_quantile = data['calibration_quantile']
        self.calibration_scores = data['calibration_scores']
        self.feature_names = data['feature_names']
        self.config = data['config']
        self.alpha = data['alpha']
        self.is_trained = True
        print(f"Modelo cargado desde: {path}")
