from __future__ import annotations

import argparse
import os
import pandas as pd
import time

from btc_predictor.research import run_comparison


def main():
    parser = argparse.ArgumentParser(description="Durable BTC baseline/MTF research worker")
    parser.add_argument("--data-dir", default=os.getenv("BTC_RESEARCH_DIR", "work/runtime/research"))
    parser.add_argument("--start", default=os.getenv("RESEARCH_START"))
    parser.add_argument("--end", default=os.getenv("RESEARCH_END"))
    parser.add_argument("--decision-stride", type=int, default=int(os.getenv("RESEARCH_DECISION_STRIDE", "1")))
    parser.add_argument("--watch", action="store_true", help="Remain alive after completion when used as a Render worker")
    args = parser.parse_args()
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").floor("min")
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365)
    result = run_comparison(args.data_dir, start, end, {"decision_stride":args.decision_stride})
    print(result)
    while args.watch:
        time.sleep(3600)


if __name__ == "__main__": main()
