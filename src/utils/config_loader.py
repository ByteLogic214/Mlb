"""
Cargador centralizado de configuración.
Soporta variables de entorno y validación estructural.
"""
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Carga y valida configuración desde YAML con interpolación de variables de entorno."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load()
        self._validate()
    
    def _load(self) -> Dict[str, Any]:
        """Carga YAML con interpolación de variables de entorno ${VAR}."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config no encontrado: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            raw = f.read()
        
        def replace_env(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                if var_name in ["PINNACLE_API_KEY", "ODDS_API_KEY"]:
                    return ""
                raise ValueError(f"Variable de entorno requerida no definida: {var_name}")
            return value
        
        raw = re.sub(r'\$\{([^}]+)\}', replace_env, raw)
        return yaml.safe_load(raw)
    
    def _validate(self):
        """Validación estructural de configuración crítica."""
        required_keys = ['data', 'model', 'evaluation', 'bankroll']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Sección requerida faltante en config: {key}")
        
        splits = self.config['data']['splits']
        assert splits['train_end'] < splits['val_start'], "Train debe terminar antes de val"
        assert splits['val_end'] < splits['test_start'], "Val debe terminar antes de test"
        
        odds_sources = self.config['data']['sources']['odds']
        active_sources = [k for k, v in odds_sources.items() if v.get('enabled', False)]
        if not active_sources:
            raise ValueError("Al menos una fuente de cuotas debe estar habilitada")
    
    def get(self, *keys, default=None):
        """Acceso seguro anidado: config.get('model', 'hyperparameters', 'max_depth')."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    @property
    def season(self) -> int:
        return self.config['data']['season']
    
    @property
    def train_end(self) -> str:
        return self.config['data']['splits']['train_end']
    
    @property
    def val_end(self) -> str:
        return self.config['data']['splits']['val_end']
    
    @property
    def test_start(self) -> str:
        return self.config['data']['splits']['test_start']
    
    @property
    def min_ev_threshold(self) -> float:
        return self.config['evaluation']['min_ev_threshold']
    
    @property
    def kelly_fraction(self) -> float:
        return self.config['bankroll']['kelly']['fraction']
    
    @property
    def max_stake_percent(self) -> float:
        return self.config['bankroll']['kelly']['max_stake_percent']
    
    @property
    def max_daily_risk(self) -> float:
        return self.config['bankroll']['daily_limits']['max_risk_percent']
