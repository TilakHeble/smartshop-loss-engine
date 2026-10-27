"""Connect to the warehouse and run the dbt project from Python."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb

from smartshop.config import DBT_DIR, WAREHOUSE_PATH


def connect(path: Path = WAREHOUSE_PATH) -> duckdb.DuckDBPyConnection:
    """Read-only connection to the warehouse (tables live in schemas raw, staging, marts)."""
    con = duckdb.connect(str(path), read_only=True)
    con.execute("SET enable_progress_bar = false")
    return con


def run_dbt(
    command: str = "build",
    warehouse: Path = WAREHOUSE_PATH,
    vars: dict | None = None,
    target_path: Path | None = None,
    extra: list[str] | None = None,
) -> None:
    """Run `dbt <command>` against a given warehouse file. Raises if dbt fails."""
    dbt = Path(sys.executable).with_name("dbt")
    args = [str(dbt) if dbt.exists() else "dbt", command, "--profiles-dir", ".", "--quiet"]
    if vars:
        args += ["--vars", repr(vars).replace("'", '"')]
    if target_path:
        args += ["--target-path", str(target_path)]
    args += extra or []
    env = {**os.environ, "SMARTSHOP_WAREHOUSE": str(Path(warehouse).resolve())}
    result = subprocess.run(args, cwd=DBT_DIR, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dbt {command} failed:\n{result.stdout}\n{result.stderr}")
