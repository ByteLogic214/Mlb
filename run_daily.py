#!/usr/bin/env python3
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
        help="Validar cobertura conformal: 'model.joblib,test.csv'"
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
        print("📚 Modo
