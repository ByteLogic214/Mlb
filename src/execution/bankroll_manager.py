"""
Gestor de bankroll con Kelly fraccional y límites de riesgo.
Implementa protección contra correlación y stop-losses.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Bet:
    """Orden de apuesta ejecutable."""
    game_id: str
    pick: str
    odds_decimal: float
    prob: float
    ev: float
    stake: float
    confidence: float


class BankrollManager:
    """
    Gestiona bankroll con Kelly fraccional, límites diarios y stop-losses.
    CRÍTICO: No permite picks correlacionados del mismo partido.
    """
    
    def __init__(self, initial_bankroll: float = 10000.0,
                 kelly_fraction: float = 0.25,
                 max_stake_percent: float = 0.02,
                 max_daily_risk: float = 0.05,
                 max_picks_per_day: int = 3,
                 min_stake: float = 10.0):
        
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.max_stake_percent = max_stake_percent
        self.max_daily_risk = max_daily_risk
        self.max_picks_per_day = max_picks_per_day
        self.min_stake = min_stake
        
        self.daily_picks: List[Bet] = []
        self.daily_risk = 0.0
        self.total_picks = 0
        self.total_wagered = 0.0
        self.total_profit = 0.0
        self.peak_bankroll = initial_bankroll
        self.max_drawdown = 0.0
    
    def reset_daily(self):
        """Reinicia tracking diario."""
        self.daily_picks = []
        self.daily_risk = 0.0
    
    def calculate_kelly_stake(self, prob: float, odds: float) -> float:
        """
        Kelly Criterion: f* = (bp - q) / b
        Donde b = odds - 1, p = prob, q = 1 - p
        """
        b = odds - 1.0
        q = 1.0 - prob
        
        if b <= 0:
            return 0.0
        
        kelly_fraction = (b * prob - q) / b
        kelly_fraction = max(0.0, kelly_fraction)
        
        adjusted_fraction = kelly_fraction * self.kelly_fraction
        
        stake = self.current_bankroll * adjusted_fraction
        
        return stake
    
    def validate_bet(self, game_id: str, pick: str, odds: float, 
                     prob: float, ev: float) -> tuple:
        """
        Valida si una apuesta puede ejecutarse bajo las reglas de bankroll.
        
        Returns:
            (is_valid: bool, stake: float, reason: str)
        """
        if ev < 0:
            return False, 0.0, "EV negativo"
        
        if len(self.daily_picks) >= self.max_picks_per_day:
            return False, 0.0, f"Límite diario de {self.max_picks_per_day} picks alcanzado"
        
        for existing in self.daily_picks:
            if existing.game_id == game_id:
                return False, 0.0, "Pick correlacionado: mismo partido ya seleccionado"
        
        stake = self.calculate_kelly_stake(prob, odds)
        
        max_stake = self.current_bankroll * self.max_stake_percent
        stake = min(stake, max_stake)
        
        if stake < self.min_stake:
            return False, 0.0, f"Stake calculado (${stake:.2f}) menor al mínimo (${self.min_stake:.2f})"
        
        potential_loss = stake
        if self.daily_risk + potential_loss > self.current_bankroll * self.max_daily_risk:
            return False, 0.0, f"Excede límite de riesgo diario ({self.max_daily_risk:.1%})"
        
        daily_pnl = sum([
            (b.stake * (b.odds_decimal - 1) if b.pick == "WIN" else -b.stake)
            for b in self.daily_picks
        ])
        
        if daily_pnl < -self.current_bankroll * 0.03:
            return False, 0.0, "Stop-loss diario alcanzado (3%)"
        
        return True, stake, "OK"
    
    def place_bet(self, game_id: str, pick: str, odds: float, 
                  prob: float, ev: float, confidence: float) -> Optional[Bet]:
        """Ejecuta una apuesta si pasa validación."""
        is_valid, stake, reason = self.validate_bet(game_id, pick, odds, prob, ev)
        
        if not is_valid:
            print(f"❌ Rechazado [{game_id}]: {reason}")
            return None
        
        bet = Bet(
            game_id=game_id,
            pick=pick,
            odds_decimal=odds,
            prob=prob,
            ev=ev,
            stake=round(stake, 2),
            confidence=confidence
        )
        
        self.daily_picks.append(bet)
        self.daily_risk += stake
        self.total_picks += 1
        self.total_wagered += stake
        
        print(f"✅ Ejecutado [{game_id}]: {pick} @ {odds:.3f} | "
              f"Stake: ${stake:.2f} | EV: {ev:.2%} | Conf: {confidence:.3f}")
        
        return bet
    
    def update_result(self, bet: Bet, won: bool):
        """Actualiza bankroll tras resultado."""
        if won:
            profit = bet.stake * (bet.odds_decimal - 1)
        else:
            profit = -bet.stake
        
        self.current_bankroll += profit
        self.total_profit += profit
        
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
        
        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def get_stats(self) -> Dict:
        """Estadísticas de rendimiento."""
        roi = self.total_profit / self.total_wagered if self.total_wagered > 0 else 0
        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": round(self.current_bankroll, 2),
            "total_profit": round(self.total_profit, 2),
            "total_picks": self.total_picks,
            "total_wagered": round(self.total_wagered, 2),
            "roi": round(roi, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "peak_bankroll": round(self.peak_bankroll, 2),
        }
