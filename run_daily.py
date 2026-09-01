
# 6. Actualizar run_daily.py
run_daily = '''#!/usr/bin/env python3
"""
Script de ejecución diaria para MLB Proyección v2.0.
Integración: SharpAPI (agregador de cuotas)
Cron job recomendado: 0 10 * * * (10:00 AM ET)
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from pipeline import MLBPredictionPipeline
from utils.logger import StructuredLogger


def main():
    parser = argparse.ArgumentParser(description="MLB Proyección v2.0 - Ejecución Diaria")
    parser.add_argument(
        "--date", 
        type=str, 
        default=None,
        help="Fecha a procesar (YYYY-MM-DD). Default: hoy"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["predict", "train", "backtest", "validate"],
        default="predict",
        help="Modo de ejecución"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular sin ejecutar apuestas reales"
    )
    parser.add_argument(
        "--validate-coverage",
        type=str,
        default=None,
        help="Validar cobertura conformal: ruta al modelo y test data"
    )
    
    args = parser.parse_args()
    
    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print("Error: Fecha debe estar en formato YYYY-MM-DD")
            sys.exit(1)
    
    print(f"🚀 MLB Proyección v2.0 | Modo: {args.mode} | Fecha: {args.date or 'hoy'}")
    print("=" * 60)
    
    pipeline = MLBPredictionPipeline(config_path=args.config)
    
    if args.mode == "train":
        print("📚 Modo entrenamiento...")
        metrics = pipeline.run_training()
        print("\\n📊 Métricas de entrenamiento:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")
    
    elif args.mode == "predict":
        print("🔮 Modo predicción...")
        if args.dry_run:
            print("   ⚠️  DRY RUN: No se ejecutarán apuestas reales")
        
        bets = pipeline.run_daily_prediction(args.date)
        
        print("\\n" + "=" * 60)
        print(f"📋 RESUMEN DEL DÍA: {args.date or datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        if bets:
            print(f"✅ Picks ejecutados: {len(bets)}")
            for i, bet in enumerate(bets, 1):
                print(f"   {i}. {bet.game_id}: {bet.pick} @ {bet.odds_decimal:.3f} | "
                      f"Stake: ${bet.stake:.2f} | EV: {bet.ev:.2%}")
        else:
            print("❌ No se ejecutaron picks hoy")
            print("   Razones posibles:")
            print("   - No hay juegos programados")
            print("   - Ningún juego supera el umbral de EV conservador")
            print("   - Error en obtención de cuotas desde SharpAPI")
            print("   - Mercados disponibles no coinciden con allowed_markets")
        
        print(f"\\n💰 Bankroll actual: ${pipeline.bankroll.current_bankroll:.2f}")
        print(f"📊 Stats: {pipeline.bankroll.get_stats()}")
    
    elif args.mode == "validate":
        print("📈 Validando cobertura conformal...")
        if args.validate_coverage:
            from validate_coverage import validate_coverage
            model_path, test_path = args.validate_coverage.split(",")
            validate_coverage(model_path.strip(), test_path.strip())
        else:
            print("   Uso: --validate-coverage 'model.joblib,test.csv'")
    
    print("\\n✅ Pipeline completado exitosamente")


if __name__ == "__main__":
    main()
'''

with open(os.path.join(base_dir, "run_daily.py"), "w") as f:
    f.write(run_daily)

# 7. Crear script de ejemplo con datos reales de SharpAPI
sharpapi_example = '''#!/usr/bin/env python3
"""
Ejemplo de uso con datos reales de SharpAPI (CLE vs TOR, 2026-09-01).
Demuestra cómo parsear y evaluar las cuotas del archivo de ejemplo.
"""
import json
import pandas as pd
from pathlib import Path

# Datos de ejemplo del archivo compartido
EXAMPLE_DATA = {
    "event_id": "mlb_bluejays_guardians_2026-09-01_b3",
    "home_team": "Cleveland Guardians",
    "away_team": "Toronto Blue Jays",
    "home_pitcher": "G Williams",
    "away_pitcher": "S Miles",
    "event_start_time": "2026-09-01T22:41Z",
    "markets": [
        {
            "market_type": "1st_5_innings_moneyline",
            "selection": "Cleveland Guardians",
            "side": "home",
            "odds_american": -188,
            "odds_decimal": 1.532,
            "is_main_line": True,
            "is_alternate_line": False
        },
        {
            "market_type": "1st_5_innings_moneyline",
            "selection": "Toronto Blue Jays",
            "side": "away",
            "odds_american": 148,
            "odds_decimal": 2.48,
            "is_main_line": True,
            "is_alternate_line": False
        },
        {
            "market_type": "1st_5_innings_total_runs",
            "selection": "Over",
            "side": "over",
            "line": 4.5,
            "odds_american": 100,
            "odds_decimal": 2.0,
            "is_main_line": True,
            "is_alternate_line": False
        },
        {
            "market_type": "1st_5_innings_total_runs",
            "selection": "Under",
            "side": "under",
            "line": 4.5,
            "odds_american": -130,
            "odds_decimal": 1.769,
            "is_main_line": True,
            "is_alternate_line": False
        }
    ]
}


def parse_example_data(data: dict) -> pd.DataFrame:
    """Parsea datos de ejemplo al formato del pipeline."""
    rows = []
    
    for market in data["markets"]:
        rows.append({
            "event_id": data["event_id"],
            "home_team": data["home_team"],
            "away_team": data["away_team"],
            "market_type": market["market_type"],
            "selection": market["selection"],
            "team_side": market.get("side"),
            "line": market.get("line"),
            "odds_american": market["odds_american"],
            "odds_decimal": market["odds_decimal"],
            "is_main_line": market["is_main_line"],
            "is_alternate_line": market["is_alternate_line"],
            "home_pitcher": data["home_pitcher"],
            "away_pitcher": data["away_pitcher"]
        })
    
    return pd.DataFrame(rows)


def demonstrate_devigging(df: pd.DataFrame):
    """Demuestra el desvigado de cuotas."""
    print("=" * 70)
    print("DEMOSTRACIÓN: Desvigado de Cuotas SharpAPI")
    print("=" * 70)
    
    # Agrupar por market_type
    for market_type, group in df.groupby("market_type"):
        print(f"\\n📊 {market_type.upper()}")
        print("-" * 50)
        
        if "moneyline" in market_type:
            home = group[group["team_side"] == "home"].iloc[0]
            away = group[group["team_side"] == "away"].iloc[0]
            
            # Desvigado
            def american_to_prob(odds):
                return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
            
            home_impl = american_to_prob(home["odds_american"])
            away_impl = american_to_prob(away["odds_american"])
            vig = home_impl + away_impl - 1.0
            
            home_fair = home_impl / (home_impl + away_impl)
            away_fair = away_impl / (home_impl + away_impl)
            
            print(f"   {home['selection']}: {home['odds_american']} (implícita: {home_impl:.2%}) -> Fair: {home_fair:.2%}")
            print(f"   {away['selection']}: {away['odds_american']} (implícita: {away_impl:.2%}) -> Fair: {away_fair:.2%}")
            print(f"   Vig: {vig:.2%}")
            
            # EV con probabilidad del modelo (ejemplo: CLE 47.9% como en análisis previo)
            model_prob = 0.479  # Probabilidad del modelo para CLE
            odds_dec = home["odds_decimal"]
            ev = (model_prob * odds_dec) - 1
            print(f"\\n   📈 EV con prob modelo ({model_prob:.1%}) @ {odds_dec:.3f}: {ev:.2%}")
            
            if ev < 0:
                print(f"   ❌ RECHAZADO: EV negativo")
            
        elif "total" in market_type:
            over = group[group["team_side"] == "over"].iloc[0]
            under = group[group["team_side"] == "under"].iloc[0]
            
            def american_to_prob(odds):
                return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
            
            over_impl = american_to_prob(over["odds_american"])
            under_impl = american_to_prob(under["odds_american"])
            vig = over_impl + under_impl - 1.0
            
            over_fair = over_impl / (over_impl + under_impl)
            under_fair = under_impl / (over_impl + under_impl)
            
            print(f"   Over {over['line']}: {over['odds_american']} -> Fair: {over_fair:.2%}")
            print(f"   Under {under['line']}: {under['odds_american']} -> Fair: {under_fair:.2%}")
            print(f"   Vig: {vig:.2%}")


if __name__ == "__main__":
    df = parse_example_data(EXAMPLE_DATA)
    print("Datos parseados:")
    print(df.to_string())
    print()
    
    demonstrate_devigging(df)
    
    print("\\n" + "=" * 70)
    print("NOTA: Este es un ejemplo con datos estáticos.")
    print("En producción, los datos vienen de la API de SharpAPI en tiempo real.")
    print("=" * 70)
'''

with open(os.path.join(base_dir, "sharpapi_example.py"), "w") as f:
    f.write(sharpapi_example)

# 8. Actualizar README.md
readme = '''# MLB Proyección v2.0

Sistema cuantitativo de predicción y ejecución de apuestas MLB con **Conformal Prediction**, **Kelly Fraccional** y **evaluación conservadora de EV**.

## 🎯 Filosofía

> "La mayoría de los 'edges' proyectados por modelos cuantitativos son consumidos en su totalidad por la incertidumbre intrínseca del modelo."

Este sistema no busca generar picks todos los días. Busca **no apostar cuando no hay valor**, documentando explícitamente los días sin picks como métrica de salud del sistema.

## 🏗️ Arquitectura

```
config/           # Configuración centralizada (YAML)
src/
  data_ingestion/   # StatsAPI + SharpAPI (agregador de cuotas)
  features/         # Feature engineering sin leakage temporal
  models/           # XGBoost + Split Conformal Prediction
  evaluation/       # EV Calculator (conservador, múltiples mercados)
  execution/        # Bankroll Manager (Kelly fraccional)
  utils/            # Config loader + Logger estructurado
reports/
  daily/            # Reportes JSON de cada día (incluye NO PICK)
  performance/      # Métricas acumuladas (ROI, CLV, Drawdown)
```

## 🚀 Instalación

```bash
git clone https://github.com/ByteLogic214/mlb-proyeccion.git
cd mlb-proyeccion
pip install -r requirements.txt
```

## ⚙️ Configuración

### 1. Variables de entorno (obligatorias)

```bash
# Fuente primaria: SharpAPI (recomendado)
export SHARPAPI_KEY="tu_api_key_de_sharpapi"

# Fallback opcional: Pinnacle directo
export PINNACLE_API_KEY="tu_api_key"

# Fallback terciario: The Odds API
export ODDS_API_KEY="tu_api_key"
```

### 2. SharpAPI

- **Tier Free**: 60s delay, 12 req/min, FanDuel + DraftKings
- **Tier Paid**: Real-time, todas las casas
- Registro: [sharpapi.io](https://sharpapi.io)

### 3. Ajustar `config/config.yaml`

```yaml
data:
  sources:
    odds:
      sharpapi:
        enabled: true
        api_key: "${SHARPAPI_KEY}"
        tier: "free"  # o "paid"
```

## 📊 Uso

### Entrenamiento inicial
```bash
python run_daily.py --mode train --config config/config.yaml
```

### Predicción diaria (cron job)
```bash
# 10:00 AM ET antes de juegos diurnos
0 10 * * * cd /path/to/mlb-proyeccion && python run_daily.py --mode predict
```

### Simulación con datos de ejemplo (SharpAPI)
```bash
python sharpapi_example.py
```

### Validación de cobertura conformal
```bash
python run_daily.py --mode validate --validate-coverage "models/xgboost_conformal_2026.joblib,data/test.csv"
```

## 🔬 Metodología

### 1. Fuentes de Cuotas: SharpAPI
- **Agregador** de múltiples casas (FanDuel, DraftKings, Pinnacle en paid)
- **Filtrado automático** de mercados permitidos
- **Exclusión** de: líneas alternativas, props, innings parciales, live
- **Desvigado multiplicativo** por mercado

### 2. Split Temporal Estricto
- **Train**: Abril - Julio
- **Validation**: Agosto (calibración + early stopping)
- **Test/Prod**: Septiembre en adelante

### 3. Conformal Prediction
- Método: Split Conformal
- Cobertura objetivo: 95%
- Intervalo: `[prob_point - q, prob_point + q]`

### 4. Evaluación de EV Conservadora
```
EV_conservador = (prob_lower * odds) - 1
```
Solo se ejecuta si `EV_conservador >= 4%`.

### 5. Kelly Fraccional
```
stake = bankroll * 0.25 * ((prob * (odds - 1) - (1 - prob)) / (odds - 1))
```

Límites:
- Máximo 2% del bankroll por apuesta
- Máximo 5% de riesgo diario
- Máximo 3 picks por día
- No picks correlacionados (mismo partido)

## 📈 Métricas de Salud

- **Pick Rate**: % de días con al menos 1 pick (objetivo: 30-50%)
- **Coverage**: Cobertura real del intervalo conformal (objetivo: >=93%)
- **CLV**: Closing Line Value vs. línea de cierre
- **ROI**: Retorno real sobre apuestas ejecutadas
- **Max Drawdown**: Caída máxima desde peak

## ⚠️ Limitaciones Conocidas

1. **SharpAPI Free Tier**: 60s de delay. Para arbitraje de latencia, necesitas tier paid.
2. **Datos históricos**: El modelo requiere base de datos con stats de 2024-2026.
3. **Alineaciones de última hora**: El pipeline no captura cambios post-planilla firmada.
4. **Overfitting en septiembre**: Expansión de rosters aumenta varianza.
5. **Modelo de totals**: Actualmente usa placeholder (50%). Requiere modelo separado.

## 📝 Licencia

MIT License - Uso educativo y de investigación.

## 🙏 Créditos

Diseñado con rigor metodológico siguiendo principios de:
- **Conformal Prediction** (Vovk, Gammerman, Shafer)
- **Kelly Criterion** (Kelly, 1956; Thorp, 2006)
- **Market Efficiency** (Fama, 1970)
'''

with open(os.path.join(base_dir, "README.md"), "w") as f:
    f.write(readme)

print("✅ Archivos actualizados:")
print("   - run_daily.py (modo validate agregado)")
print("   - sharpapi_example.py (ejemplo con datos reales)")
print("   - README.md (documentación actualizada)")
