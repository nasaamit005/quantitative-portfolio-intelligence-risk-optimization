"""Build the enhanced reporting layer from this project's saved outputs.

Reads reports/ and data/processed (nothing is re-optimised), then writes to reports/enhanced/:
  quantitative_portfolio_analytics_enhanced.xlsx   formula-driven workbook with native charts
  enhanced_report.html                              34 interactive Plotly charts with captions
  tables/*.csv                                      every analytical table
  results.pkl                                       consumed by the Flask dashboard (/analytics/)

Usage:  python scripts/run_enhanced_reporting.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantfin.enhanced.adapter import build_results  # noqa: E402
from quantfin.enhanced.report import export_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(ROOT), help="project root (default: this repository)")
    ap.add_argument(
        "--out", default=None, help="output directory (default: <root>/reports/enhanced)"
    )
    ap.add_argument("--mc-sims", type=int, default=20000)
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    res = build_results(a.root, mc_sims=a.mc_sims)
    out = Path(a.out) if a.out else Path(a.root) / "reports" / "enhanced"
    manifest = export_all(res, out)
    print("Done.", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
