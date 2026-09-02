"""
Ingesta de cuotas desde SharpAPI (agregador primario) 
y Pinnacle/TheOddsAPI (fallbacks).
Maneja paginación, múltiples market_types, y filtrado estricto de mercados.
"""
import os
import re
import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd


class OddsAPI:
    """
    Cliente unificado para cuotas deportivas.
    Fuente primaria: SharpAPI (agregador de múltiples casas)
    Fallbacks: Pinnacle directo, The Odds API
    """
    
    def __init__(self, cache_dir: str = "data/raw/odds", 
                 cache_ttl_minutes: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        self.sharpapi_key = os.getenv("SHARPAPI_KEY")
        self.sharpapi_url = "https://api.sharpapi.io/api/v1"
        self.sharpapi_tier = "free"
        
        self.pinnacle_key = os.getenv("PINNACLE_API_KEY")
        self.pinnacle_url = "https://api.pinnacle.com/v2"
        
        self.fallback_key = os.getenv("ODDS_API_KEY")
        self.fallback_url = "https://api.the-odds-api.com/v4"
    
    def _get_cache(self, key: str) -> Optional[Dict]:
        """Obtiene datos de caché si no han expirado."""
        cache_path = self.cache_dir / f"{key}.json"
        if not cache_path.exists():
            return None
        
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.cache_ttl:
            return None
        
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    def _set_cache(self, key: str, data: Dict):
        """Guarda datos en caché."""
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _should_exclude_market(self, market_type: str, is_alternate: bool, 
                               is_main_line: bool, config: Dict) -> bool:
        """Determina si un mercado debe ser excluido según reglas de configuración."""
        allowed = config.get('evaluation', {}).get('allowed_markets', [])
        excluded = config.get('evaluation', {}).get('excluded_markets', [])
        
        if config.get('evaluation', {}).get('exclude_alternate_lines', True) and is_alternate:
            return True
        
        if config.get('evaluation', {}).get('exclude_if_not_main_line', True) and not is_main_line:
            return True
        
        for pattern in excluded:
            pattern_regex = pattern.replace("*", ".*")
            if re.match(pattern_regex, market_type):
                return True
        
        for pattern in allowed:
            pattern_regex = pattern.replace("*", ".*")
            if re.match(pattern_regex, market_type):
                return False
        
        return True
    
    def _devig_moneyline(self, home_odds: float, away_odds: float) -> Tuple[float, float, float]:
        """Desvigado multiplicativo para moneylines de 2 resultados."""
        def american_to_prob(american: float) -> float:
            if american > 0:
                return 100 / (american + 100)
            else:
                return abs(american) / (abs(american) + 100)
        
        home_impl = american_to_prob(home_odds)
        away_impl = american_to_prob(away_odds)
        total_impl = home_impl + away_impl
        vig = total_impl - 1.0
        
        home_fair = home_impl / total_impl
        away_fair = away_impl / total_impl
        
        return home_fair, away_fair, vig
    
    def _devig_totals(self, over_odds: float, under_odds: float) -> Tuple[float, float, float]:
        """Desvigado para mercados de totals."""
        return self._devig_moneyline(over_odds, under_odds)
    
    def _devig_runline(self, home_odds: float, away_odds: float) -> Tuple[float, float, float]:
        """Desvigado para run lines."""
        return self._devig_moneyline(home_odds, away_odds)
    
    def _parse_sharpapi_event(self, event: Dict, config: Dict) -> List[Dict]:
        """Parsea un evento de SharpAPI en múltiples registros de cuotas."""
        odds_list = []
        
        market_type = event.get("market_type", "")
        is_alternate = event.get("is_alternate_line", False)
        is_main_line = event.get("is_main_line", False)
        
        if self._should_exclude_market(market_type, is_alternate, is_main_line, config):
            return []
        
        event_id = event.get("event_id", "")
        home_team = event.get("home_team", "").strip()
        away_team = event.get("away_team", "").strip()
        selection = event.get("selection", "")
        selection_type = event.get("selection_type", "")
        team_side = event.get("team_side", "")
        market_segment = event.get("market_segment", "")
        line = event.get("line")
        
        odds_american = event.get("odds_american", 0)
        odds_decimal = event.get("odds_decimal", 0)
        odds_prob = event.get("odds_probability", 0)
        
        sportsbook = event.get("sportsbook", "unknown")
        is_live = event.get("is_live", False)
        
        home_pitcher = event.get("home_pitcher", "")
        away_pitcher = event.get("away_pitcher", "")
        
        market_category = self._categorize_market(market_type)
        
        odds_list.append({
            "event_id": event_id,
            "external_event_id": event.get("external_event_id", ""),
            "home_team": home_team,
            "away_team": away_team,
            "market_type": market_type,
            "market_category": market_category,
            "market_segment": market_segment,
            "selection": selection,
            "selection_type": selection_type,
            "team_side": team_side,
            "line": line,
            "odds_american": odds_american,
            "odds_decimal": odds_decimal,
            "odds_probability_raw": odds_prob,
            "sportsbook": sportsbook,
            "is_live": is_live,
            "is_main_line": is_main_line,
            "is_alternate_line": is_alternate,
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
            "event_start_time": event.get("event_start_time", ""),
            "timestamp": event.get("timestamp", ""),
            "source": "sharpapi"
        })
        
        return odds_list
    
    def _categorize_market(self, market_type: str) -> str:
        """Categoriza el tipo de mercado para procesamiento posterior."""
        if "moneyline" in market_type:
            return "moneyline"
        elif "run_line" in market_type:
            return "run_line"
        elif "total" in market_type:
            return "totals"
        else:
            return "other"
    
    def get_sharpapi_odds(self, sport: str = "baseball", 
                          league: str = "mlb",
                          config: Optional[Dict] = None) -> pd.DataFrame:
        """Obtiene cuotas de SharpAPI con paginación completa."""
        if not self.sharpapi_key:
            raise ValueError("SHARPAPI_KEY no configurada")
        
        if config is None:
            config = {}
        
        cache_key = f"sharpapi_{sport}_{league}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        cached = self._get_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)
        
        all_events = []
        offset = 0
        limit = 50
        
        while True:
            headers = {"Authorization": f"Bearer {self.sharpapi_key}"}
            
            params = {
                "sport": sport,
                "league": league,
                "limit": limit,
                "offset": offset,
                "is_live": "false"
            }
            
            try:
                response = requests.get(
                    f"{self.sharpapi_url}/odds",
                    headers=headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                events = data.get("data", [])
                
                if not events:
                    break
                
                for event in events:
                    parsed = self._parse_sharpapi_event(event, config)
                    all_events.extend(parsed)
                
                pagination = data.get("pagination", {})
                if not pagination.get("has_more", False):
                    break
                
                offset = pagination.get("next_offset", offset + limit)
                
                time.sleep(5.1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error en SharpAPI (offset {offset}): {e}")
                break
        
        df = pd.DataFrame(all_events)
        
        if len(df) > 0:
            self._set_cache(cache_key, all_events)
        
        return df
    
    def aggregate_event_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega cuotas por evento y market_type para obtener probabilidades desvigadas."""
        if len(df) == 0:
            return df
        
        aggregated = []
        
        grouped = df.groupby(["event_id", "market_type", "market_segment", "line"])
        
        for (event_id, market_type, segment, line), group in grouped:
            if len(group) < 2:
                continue
            
            home_team = group["home_team"].iloc[0]
            away_team = group["away_team"].iloc[0]
            market_category = group["market_category"].iloc[0]
            
            if market_category in ['moneyline', 'run_line']:
                home_row = group[group["team_side"] == "home"]
                away_row = group[group["team_side"] == "away"]
                
                if len(home_row) == 1 and len(away_row) == 1:
                    home_odds = home_row["odds_american"].values[0]
                    away_odds = away_row["odds_american"].values[0]
                    
                    home_fair, away_fair, vig = self._devig_moneyline(home_odds, away_odds)
                    
                    aggregated.append({
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_type": market_type,
                        "market_segment": segment,
                        "market_category": market_category,
                        "line": line,
                        "home_odds_american": home_odds,
                        "away_odds_american": away_odds,
                        "home_odds_decimal": home_row["odds_decimal"].values[0],
                        "away_odds_decimal": away_row["odds_decimal"].values[0],
                        "home_prob_impl": round(home_fair, 4),
                        "away_prob_impl": round(away_fair, 4),
                        "vig": round(vig, 4),
                        "sportsbook": home_row["sportsbook"].values[0],
                        "is_main_line": home_row["is_main_line"].values[0],
                        "source": "sharpapi"
                    })
            
            elif market_category == 'totals':
                over_row = group[group["selection_type"] == "over"]
                under_row = group[group["selection_type"] == "under"]
                
                if len(over_row) == 1 and len(under_row) == 1:
                    over_odds = over_row["odds_american"].values[0]
                    under_odds = under_row["odds_american"].values[0]
                    
                    over_fair, under_fair, vig = self._devig_totals(over_odds, under_odds)
                    
                    aggregated.append({
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_type": market_type,
                        "market_segment": segment,
                        "market_category": market_category,
                        "line": line,
                        "over_odds_american": over_odds,
                        "under_odds_american": under_odds,
                        "over_odds_decimal": over_row["odds_decimal"].values[0],
                        "under_odds_decimal": under_row["odds_decimal"].values[0],
                        "over_prob_impl": round(over_fair, 4),
                        "under_prob_impl": round(under_fair, 4),
                        "vig": round(vig, 4),
                        "sportsbook": over_row["sportsbook"].values[0],
                        "is_main_line": over_row["is_main_line"].values[0],
                        "source": "sharpapi"
                    })
            
            elif market_category == 'run_line':
                home_row = group[group["team_side"] == "home"]
                away_row = group[group["team_side"] == "away"]
                
                if len(home_row) == 1 and len(away_row) == 1:
                    home_odds = home_row["odds_american"].values[0]
                    away_odds = away_row["odds_american"].values[0]
                    
                    home_fair, away_fair, vig = self._devig_runline(home_odds, away_odds)
                    
                    aggregated.append({
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_type": market_type,
                        "market_segment": segment,
                        "market_category": market_category,
                        "line": line,
                        "home_odds_american": home_odds,
                        "away_odds_american": away_odds,
                        "home_odds_decimal": home_row["odds_decimal"].values[0],
                        "away_odds_decimal": away_row["odds_decimal"].values[0],
                        "home_prob_impl": round(home_fair, 4),
                        "away_prob_impl": round(away_fair, 4),
                        "vig": round(vig, 4),
                        "sportsbook": home_row["sportsbook"].values[0],
                        "is_main_line": home_row["is_main_line"].values[0],
                        "source": "sharpapi"
                    })
        
        return pd.DataFrame(aggregated)
    
    def get_odds(self, config: Optional[Dict] = None) -> pd.DataFrame:
        """Obtiene cuotas agregadas con fallback automático."""
        try:
            raw_df = self.get_sharpapi_odds(config=config)
            if len(raw_df) > 0:
                return self.aggregate_event_odds(raw_df)
        except Exception as e:
            print(f"SharpAPI falló: {e}")
        
        if self.pinnacle_key:
            try:
                print("Intentando Pinnacle...")
                return self.get_pinnacle_odds()
            except Exception as e:
                print(f"Pinnacle falló: {e}")
        
        if self.fallback_key:
            try:
                print("Intentando The Odds API...")
                return self.get_fallback_odds()
            except Exception as e:
                print(f"Fallback falló: {e}")
        
        raise ValueError("Todas las fuentes de cuotas fallaron")
    
    def get_pinnacle_odds(self, sport_id: int = 246) -> pd.DataFrame:
        """Fallback a Pinnacle directo."""
        if not self.pinnacle_key:
            raise ValueError("PINNACLE_API_KEY no configurada")
        return pd.DataFrame()
    
    def get_fallback_odds(self, sport: str = "baseball_mlb") -> pd.DataFrame:
        """Fallback a The Odds API."""
        if not self.fallback_key:
            raise ValueError("ODDS_API_KEY no configurada")
        return pd.DataFrame()
