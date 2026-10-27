"""Load the Complete Journey CSVs into typed Parquet and a DuckDB warehouse.

Reuses the typed, renamed table specs from the Nectar offer engine (pinned in pyproject.toml),
so both projects read the raw data identically and share one member ID.

Run:
    python -m smartshop.ingest.load_raw --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import duckdb
from nectar.ingest.load_raw import TABLES, find_raw_file

from smartshop.config import PROCESSED_DIR, RAW_DIR, TABLES_USED, WAREHOUSE_PATH

log = logging.getLogger(__name__)

SPECS = [spec for spec in TABLES if spec.name in TABLES_USED]


def ingest(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    warehouse_path: Path = WAREHOUSE_PATH,
) -> dict[str, int]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = warehouse_path.with_suffix(".tmp.duckdb")
    tmp_path.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp_path))
    con.execute("SET enable_progress_bar = false")
    con.execute("CREATE SCHEMA raw")
    row_counts: dict[str, int] = {}

    for spec in SPECS:
        t0 = time.time()
        csv_path = find_raw_file(raw_dir, spec.file_prefix)
        parquet_path = processed_dir / f"{spec.name}.parquet"
        src = f"read_csv('{csv_path.as_posix()}', header=true, all_varchar=true)"
        query = spec.select_sql.replace("FROM src", f"FROM {src} AS src")
        con.execute(
            f"COPY ({query}) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"CREATE TABLE raw.{spec.name} AS "
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        )
        n = con.execute(f"SELECT COUNT(*) FROM raw.{spec.name}").fetchone()[0]
        row_counts[spec.name] = n
        log.info("%-20s %12s rows  %5.1fs", spec.name, f"{n:,}", time.time() - t0)

    con.close()
    tmp_path.replace(warehouse_path)  # atomic: readers never see a half-built warehouse
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ingest(raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()
