.PHONY: install ingest eda test lint clean

RAW_DIR ?= data/raw

install:
	pip install -e ".[dev]"

ingest:
	python -m smartshop.ingest.load_raw --raw-dir $(RAW_DIR)

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	jupyter nbconvert --to html notebooks/01_eda.ipynb --output-dir reports

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

clean:
	rm -rf data/processed/*.parquet data/warehouse/*.duckdb
