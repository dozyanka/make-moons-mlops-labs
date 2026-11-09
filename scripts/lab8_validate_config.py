"""Exit nonzero before starting any service when LAB8 Pydantic config invalid."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from moons_lab.lab8_config import load_lab8_config

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = load_lab8_config(args.config)
    print('VALID:', cfg.model_port)
