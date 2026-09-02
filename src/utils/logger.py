"""
Logging estructurado con rotación y formato JSON para trazabilidad.
"""
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class StructuredLogger:
    """Logger con salida estructurada (JSON) para análisis posterior."""
    
    def __init__(self, name: str, log_dir: str = "logs", level: str = "INFO"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_handler = logging.FileHandler(
                self.log_dir / f"{name}_{date_str}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            
            file_handler.setFormatter(file_formatter)
            console_handler.setFormatter(console_formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _log(self, level: str, message: str, extra: Optional[dict] = None):
        """Log con metadata estructurada."""
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        getattr(self.logger, level.lower())(message)
    
    def info(self, message: str, extra: Optional[dict] = None):
        self._log("INFO", message, extra)
    
    def warning(self, message: str, extra: Optional[dict] = None):
        self._log("WARNING", message, extra)
    
    def error(self, message: str, extra: Optional[dict] = None):
        self._log("ERROR", message, extra)
    
    def critical(self, message: str, extra: Optional[dict] = None):
        self._log("CRITICAL", message, extra)
    
    def bet_decision(self, game_id: str, decision: str, ev: float, 
                     prob_point: float, prob_lower: float, odds: float,
                     stake: float, reason: str = ""):
        """Log estructurado específico para decisiones de apuesta."""
        extra = {
            "event_type": "BET_DECISION",
            "game_id": game_id,
            "decision": decision,
            "ev_point": round(ev, 4),
            "prob_point": round(prob_point, 4),
            "prob_lower_95": round(prob_lower, 4),
            "odds": odds,
            "stake": round(stake, 2),
            "rejection_reason": reason
        }
        self._log("INFO" if decision == "EXECUTE" else "WARNING", 
                  f"Decisión de apuesta: {decision}", extra)
