.PHONY: install ingest dbt dbt-docs sample simulate basket eda test lint clean

RAW_DIR ?= data/raw

install:
	pip install -e ".[dev]"

ingest:
	python -m smartshop.ingest.load_raw --raw-dir $(RAW_DIR)

dbt:
	cd dbt && dbt build --profiles-dir .

dbt-docs:
	cd dbt && dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .

simulate:
	python -m smartshop.simulate.simulator

basket:
	python -m smartshop.basket.model

sample:  # synthetic warehouse with the real schema, for trying the pipeline without the data
	python -m smartshop.ingest.sample data/warehouse/sample.duckdb
	cd dbt && SMARTSHOP_WAREHOUSE=../data/warehouse/sample.duckdb dbt build --profiles-dir .

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_simulator.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_basket_model.ipynb
	jupyter nbconvert --to html notebooks/0*.ipynb --output-dir reports

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

clean:
	rm -rf data/processed/*.parquet data/warehouse/*.duckdb
