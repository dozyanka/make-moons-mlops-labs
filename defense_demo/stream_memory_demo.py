from __future__ import annotations

import gc
import tracemalloc

import pandas as pd

from common import RESULTS, ROOT


def measure(paths):
    gc.collect()
    tracemalloc.start()

    rows = 0
    for path in paths:
        frame = pd.read_csv(path)
        rows += len(frame)
        _ = frame[["x1", "x2"]].to_numpy(float)
        del frame

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return rows, peak / (1024 * 1024)


def main():
    paths = sorted((ROOT / "data" / "chunks").glob("*.csv"))
    if len(paths) < 24:
        raise RuntimeError(
            f"Expected at least 24 chunks, found {len(paths)}"
        )

    result_rows = []
    for count in [6, 12, 24]:
        rows, peak = measure(paths[:count])
        result_rows.append(
            {
                "chunks_processed": count,
                "rows_processed": rows,
                "peak_python_memory_mb": peak,
            }
        )

    out = pd.DataFrame(result_rows)
    out.to_csv(
        RESULTS / "stream_memory_scaling.csv",
        index=False,
    )

    print("=" * 72)
    print("STREAMING MEMORY SCALING DEMO")
    print("=" * 72)
    print(out.to_string(index=False))
    print()
    print(
        "The number of processed rows grows, while only one chunk is "
        "materialized at a time."
    )


if __name__ == "__main__":
    main()
