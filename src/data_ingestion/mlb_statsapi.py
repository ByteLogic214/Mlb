"""
Ingesta de datos de MLB StatsAPI con cacheo local y rate limiting.
Solo extrae datos estructurados, sin transformación.
"""
import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd


class MLBStatsAPI:
    """Cliente para MLB StatsAPI con cacheo y rate limiting."""
    
    BASE_URL = "https://statsapi.mlb.com/api/v1"
    
    def __init__(self, cache_dir: str = "data/raw", rate_limit: int = 10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request_time = 0
    
    def _request(self, endpoint: str, params: Optional[Dict] = None, 
                 use_cache: bool = True) -> Dict:
        """Request con rate limiting y cacheo."""
        elapsed = time.time() - self.last_request_time
        if elapsed < (1.0 / self.rate_limit):
            time.sleep((1.0 / self.rate_limit) - elapsed)
        
        cache_key = f"{endpoint.replace('/', '_')}_{hash(str(params))}.json"
        cache_path = self.cache_dir / cache_key
        
        if use_cache and cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        self.last_request_time = time.time()
        return data
    
    def get_schedule(self, start_date: str, end_date: str, 
                     sport_id: int = 1) -> pd.DataFrame:
        """Obtiene calendario de juegos en rango de fechas."""
        data = self._request("schedule", {
            "startDate": start_date,
            "endDate": end_date,
            "sportId": sport_id,
            "gameType": "R",
            "fields": "dates,games,gamePk,gameDate,status,detailedState,"
                      "teams,home,away,team,id,name,probablePitcher,id"
        })
        
        games = []
        for date in data.get("dates", []):
            for game in date.get("games", []):
                if game["status"]["detailedState"] != "Final":
                    games.append({
                        "game_id": game["gamePk"],
                        "date": game["gameDate"][:10],
                        "home_team_id": game["teams"]["home"]["team"]["id"],
                        "home_team": game["teams"]["home"]["team"]["name"],
                        "away_team_id": game["teams"]["away"]["team"]["id"],
                        "away_team": game["teams"]["away"]["team"]["name"],
                        "home_pitcher_id": game["teams"]["home"]
                            .get("probablePitcher", {}).get("id"),
                        "away_pitcher_id": game["teams"]["away"]
                            .get("probablePitcher", {}).get("id"),
                    })
        
        return pd.DataFrame(games)
    
    def get_boxscore(self, game_id: int) -> Dict:
        """Obtiene boxscore completo de un juego."""
        return self._request(f"game/{game_id}/boxscore")
    
    def get_player_stats(self, player_id: int, season: int, 
                         game_type: str = "R") -> pd.DataFrame:
        """Estadísticas de un jugador en una temporada."""
        data = self._request("people", {
            "personIds": player_id,
            "season": season,
            "hydrate": f"stats(group=[pitching,hitting],type=[season],"
                        f"season={season},gameType={game_type})"
        })
        
        stats = []
        for person in data.get("people", []):
            for stat_group in person.get("stats", []):
                for split in stat_group.get("splits", []):
                    stats.append(split.get("stat", {}))
        
        return pd.DataFrame(stats)
    
    def get_team_stats(self, team_id: int, season: int) -> pd.DataFrame:
        """Estadísticas agregadas de un equipo."""
        data = self._request("teams", {
            "teamId": team_id,
            "season": season,
            "hydrate": "standings,record(recordType=away,recordType=home)"
        })
        
        teams = data.get("teams", [])
        if not teams:
            return pd.DataFrame()
        
        team = teams[0]
        records = {
            "team_id": team_id,
            "wins": team.get("record", {}).get("wins", 0),
            "losses": team.get("record", {}).get("losses", 0),
            "win_pct": team.get("record", {}).get("winningPercentage", 0),
        }
        
        for record in team.get("record", {}).get("records", []):
            rec_type = record.get("type", "")
            if rec_type in ["home", "away"]:
                records[f"{rec_type}_wins"] = record.get("wins", 0)
                records[f"{rec_type}_losses"] = record.get("losses", 0)
                records[f"{rec_type}_win_pct"] = (
                    record.get("wins", 0) / max(record.get("wins", 0) + record.get("losses", 0), 1)
                )
        
        return pd.DataFrame([records])
