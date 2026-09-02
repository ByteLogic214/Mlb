"""
Calculador de Expected Value (EV) con evaluación conservadora.
Soporta múltiples tipos de mercado: moneyline, run_line, totals, F5 variants.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MarketType(Enum):
    MONEYLINE = "moneyline"
    RUN_LINE = "run_line"
    TOTALS = "totals"
    F5_MONEYLINE = "1st_5_innings_moneyline"
    F5_RUN_LINE = "1st_5_innings_run_line"
    F5_TOTALS = "1st_5_innings_total_runs"


@dataclass
class BetEvaluation:
    """Resultado de evaluación de una apuesta potencial."""
    game_id: str
    pick: str
    market_type: str
    line: Optional[float]
    odds_decimal: float
    prob_point: float
    prob_lower: float
    prob_upper: float
    ev_point: float
    ev_conservative: float
    ev_upper: float
    decision: str
    stake: float
    rejection_reason: str = ""
    confidence_score: float = 0.0


class EVCalculator:
    """Evalúa oportunidades de apuesta con criterios estrictos."""
    
    def __init__(self, min_ev_threshold: float = 0.04, 
                 use_conservative: bool = True):
        self.min_ev_threshold = min_ev_threshold
        self.use_conservative = use_conservative
    
    def _american_to_decimal(self, american_odds: float) -> float:
        """Convierte odds americanas a decimales."""
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))
    
    def calculate_ev(self, prob: float, odds_decimal: float) -> float:
        """EV = (prob * odds) - 1"""
        return (prob * odds_decimal) - 1.0
    
    def evaluate_moneyline(self, game_id: str, home_team: str, away_team: str,
                           prob_home_point: float, prob_home_lower: float, 
                           prob_home_upper: float,
                           home_odds_american: float, away_odds_american: float,
                           vig: float = 0.0, market_type: str = "moneyline",
                           line: Optional[float] = None) -> List[BetEvaluation]:
        """Evalúa un mercado de moneyline (full game o F5)."""
        evaluations = []
        
        home_odds_dec = self._american_to_decimal(home_odds_american)
        away_odds_dec = self._american_to_decimal(away_odds_american)
        
        prob_away_point = 1.0 - prob_home_point
        prob_away_lower = 1.0 - prob_home_upper
        prob_away_upper = 1.0 - prob_home_lower
        
        # HOME
        ev_home_point = self.calculate_ev(prob_home_point, home_odds_dec)
        ev_home_lower = self.calculate_ev(prob_home_lower, home_odds_dec)
        ev_home_upper = self.calculate_ev(prob_home_upper, home_odds_dec)
        
        ev_home_eval = ev_home_lower if self.use_conservative else ev_home_point
        
        if ev_home_eval >= self.min_ev_threshold:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="HOME", market_type=market_type, line=line,
                odds_decimal=home_odds_dec, prob_point=prob_home_point,
                prob_lower=prob_home_lower, prob_upper=prob_home_upper,
                ev_point=ev_home_point, ev_conservative=ev_home_lower,
                ev_upper=ev_home_upper, decision="EXECUTE", stake=0.0,
                confidence_score=(prob_home_upper - prob_home_lower) / max(prob_home_point, 0.001)
            ))
        else:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="HOME", market_type=market_type, line=line,
                odds_decimal=home_odds_dec, prob_point=prob_home_point,
                prob_lower=prob_home_lower, prob_upper=prob_home_upper,
                ev_point=ev_home_point, ev_conservative=ev_home_lower,
                ev_upper=ev_home_upper, decision="REJECT", stake=0.0,
                rejection_reason=f"EV conservador ({ev_home_lower:.2%}) < umbral ({self.min_ev_threshold:.2%})",
                confidence_score=(prob_home_upper - prob_home_lower) / max(prob_home_point, 0.001)
            ))
        
        # AWAY
        ev_away_point = self.calculate_ev(prob_away_point, away_odds_dec)
        ev_away_lower = self.calculate_ev(prob_away_lower, away_odds_dec)
        ev_away_upper = self.calculate_ev(prob_away_upper, away_odds_dec)
        
        ev_away_eval = ev_away_lower if self.use_conservative else ev_away_point
        
        if ev_away_eval >= self.min_ev_threshold:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="AWAY", market_type=market_type, line=line,
                odds_decimal=away_odds_dec, prob_point=prob_away_point,
                prob_lower=prob_away_lower, prob_upper=prob_away_upper,
                ev_point=ev_away_point, ev_conservative=ev_away_lower,
                ev_upper=ev_away_upper, decision="EXECUTE", stake=0.0,
                confidence_score=(prob_away_upper - prob_away_lower) / max(prob_away_point, 0.001)
            ))
        else:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="AWAY", market_type=market_type, line=line,
                odds_decimal=away_odds_dec, prob_point=prob_away_point,
                prob_lower=prob_away_lower, prob_upper=prob_away_upper,
                ev_point=ev_away_point, ev_conservative=ev_away_lower,
                ev_upper=ev_away_upper, decision="REJECT", stake=0.0,
                rejection_reason=f"EV conservador ({ev_away_lower:.2%}) < umbral ({self.min_ev_threshold:.2%})",
                confidence_score=(prob_away_upper - prob_away_lower) / max(prob_away_point, 0.001)
            ))
        
        return evaluations
    
    def evaluate_totals(self, game_id: str, home_team: str, away_team: str,
                        prob_over_point: float, prob_over_lower: float,
                        prob_over_upper: float,
                        over_odds_american: float, under_odds_american: float,
                        line: float, vig: float = 0.0,
                        market_type: str = "totals") -> List[BetEvaluation]:
        """Evalúa un mercado de totals (Over/Under)."""
        evaluations = []
        
        over_odds_dec = self._american_to_decimal(over_odds_american)
        under_odds_dec = self._american_to_decimal(under_odds_american)
        
        prob_under_point = 1.0 - prob_over_point
        prob_under_lower = 1.0 - prob_over_upper
        prob_under_upper = 1.0 - prob_over_lower
        
        # OVER
        ev_over_point = self.calculate_ev(prob_over_point, over_odds_dec)
        ev_over_lower = self.calculate_ev(prob_over_lower, over_odds_dec)
        ev_over_upper = self.calculate_ev(prob_over_upper, over_odds_dec)
        
        ev_over_eval = ev_over_lower if self.use_conservative else ev_over_point
        
        if ev_over_eval >= self.min_ev_threshold:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="OVER", market_type=market_type, line=line,
                odds_decimal=over_odds_dec, prob_point=prob_over_point,
                prob_lower=prob_over_lower, prob_upper=prob_over_upper,
                ev_point=ev_over_point, ev_conservative=ev_over_lower,
                ev_upper=ev_over_upper, decision="EXECUTE", stake=0.0,
                confidence_score=(prob_over_upper - prob_over_lower) / max(prob_over_point, 0.001)
            ))
        else:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="OVER", market_type=market_type, line=line,
                odds_decimal=over_odds_dec, prob_point=prob_over_point,
                prob_lower=prob_over_lower, prob_upper=prob_over_upper,
                ev_point=ev_over_point, ev_conservative=ev_over_lower,
                ev_upper=ev_over_upper, decision="REJECT", stake=0.0,
                rejection_reason=f"EV conservador ({ev_over_lower:.2%}) < umbral ({self.min_ev_threshold:.2%})",
                confidence_score=(prob_over_upper - prob_over_lower) / max(prob_over_point, 0.001)
            ))
        
        # UNDER
        ev_under_point = self.calculate_ev(prob_under_point, under_odds_dec)
        ev_under_lower = self.calculate_ev(prob_under_lower, under_odds_dec)
        ev_under_upper = self.calculate_ev(prob_under_upper, under_odds_dec)
        
        ev_under_eval = ev_under_lower if self.use_conservative else ev_under_point
        
        if ev_under_eval >= self.min_ev_threshold:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="UNDER", market_type=market_type, line=line,
                odds_decimal=under_odds_dec, prob_point=prob_under_point,
                prob_lower=prob_under_lower, prob_upper=prob_under_upper,
                ev_point=ev_under_point, ev_conservative=ev_under_lower,
                ev_upper=ev_under_upper, decision="EXECUTE", stake=0.0,
                confidence_score=(prob_under_upper - prob_under_lower) / max(prob_under_point, 0.001)
            ))
        else:
            evaluations.append(BetEvaluation(
                game_id=game_id, pick="UNDER", market_type=market_type, line=line,
                odds_decimal=under_odds_dec, prob_point=prob_under_point,
                prob_lower=prob_under_lower, prob_upper=prob_under_upper,
                ev_point=ev_under_point, ev_conservative=ev_under_lower,
                ev_upper=ev_under_upper, decision="REJECT", stake=0.0,
                rejection_reason=f"EV conservador ({ev_under_lower:.2%}) < umbral ({self.min_ev_threshold:.2%})",
                confidence_score=(prob_under_upper - prob_under_lower) / max(prob_under_point, 0.001)
            ))
        
        return evaluations
    
    def evaluate_batch(self, predictions_df: pd.DataFrame, 
                       odds_df: pd.DataFrame) -> pd.DataFrame:
        """Evalúa batch de juegos contra cuotas del mercado."""
        all_evals = []
        
        for _, pred in predictions_df.iterrows():
            game_id = pred['game_id']
            
            odds_match = odds_df[
                (odds_df['home_team'].str.contains(pred['home_team'].split()[-1], case=False, na=False)) |
                (odds_df['away_team'].str.contains(pred['away_team'].split()[-1], case=False, na=False))
            ]
            
            if len(odds_match) == 0:
                print(f"⚠️ No se encontraron cuotas para {pred['home_team']} vs {pred['away_team']}")
                continue
            
            for _, odds_row in odds_match.iterrows():
                market_category = odds_row.get('market_category', 'moneyline')
                market_type = odds_row['market_type']
                line = odds_row.get('line')
                
                try:
                    if market_category in ['moneyline', 'run_line']:
                        evals = self.evaluate_moneyline(
                            game_id=game_id,
                            home_team=pred['home_team'],
                            away_team=pred['away_team'],
                            prob_home_point=pred['prob_home_win'],
                            prob_home_lower=pred['prob_home_win_lower'],
                            prob_home_upper=pred['prob_home_win_upper'],
                            home_odds_american=odds_row['home_odds_american'],
                            away_odds_american=odds_row['away_odds_american'],
                            vig=odds_row.get('vig', 0.0),
                            market_type=market_type,
                            line=line
                        )
                    
                    elif market_category == 'totals':
                        prob_over_point = 0.5
                        prob_over_lower = max(0.5 - 0.15, 0.0)
                        prob_over_upper = min(0.5 + 0.15, 1.0)
                        
                        evals = self.evaluate_totals(
                            game_id=game_id,
                            home_team=pred['home_team'],
                            away_team=pred['away_team'],
                            prob_over_point=prob_over_point,
                            prob_over_lower=prob_over_lower,
                            prob_over_upper=prob_over_upper,
                            over_odds_american=odds_row['over_odds_american'],
                            under_odds_american=odds_row['under_odds_american'],
                            line=line,
                            vig=odds_row.get('vig', 0.0),
                            market_type=market_type
                        )
                    
                    else:
                        continue
                    
                    for ev in evals:
                        all_evals.append({
                            'game_id': ev.game_id,
                            'pick': ev.pick,
                            'market_type': ev.market_type,
                            'line': ev.line,
                            'home_team': pred['home_team'],
                            'away_team': pred['away_team'],
                            'odds_decimal': round(ev.odds_decimal, 3),
                            'prob_point': round(ev.prob_point, 4),
                            'prob_lower': round(ev.prob_lower, 4),
                            'prob_upper': round(ev.prob_upper, 4),
                            'ev_point': round(ev.ev_point, 4),
                            'ev_conservative': round(ev.ev_conservative, 4),
                            'ev_upper': round(ev.ev_upper, 4),
                            'decision': ev.decision,
                            'stake': round(ev.stake, 2),
                            'rejection_reason': ev.rejection_reason,
                            'confidence_score': round(ev.confidence_score, 4)
                        })
                
                except KeyError as e:
                    print(f"⚠️ Error evaluando {market_type} para {game_id}: {e}")
                    continue
        
        return pd.DataFrame(all_evals)
