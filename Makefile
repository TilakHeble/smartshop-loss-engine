.PHONY: all mitigation report dashboard install ingest dbt dbt-docs sample simulate basket risk policy stress bundle weekly demo-bundle serve docker smoke eda test lint clean

RAW_DIR ?= data/raw

install:
	pip install -e ".[dev,serve,dashboard]"

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

risk:
	python -m smartshop.risk.model

policy:
	python -m smartshop.policy.search

stress:
	python -m smartshop.experiments.sensitivity

bundle:  # serving bundle from the warehouse (no release gates)
	python -m smartshop.serving.bundle

weekly:  # the weekly job: rebuild, drift and release gates, publish only if all pass
	python -m smartshop.pipeline.weekly

demo-bundle:  # synthetic bundle: try the API without the data
	python -m smartshop.serving.demo_bundle --out artifacts

serve:
	uvicorn smartshop.serving.app:app --reload --port 8000

docker:
	docker build -t smartshop-api .
	docker run --rm -d --name smartshop-api -p 8000:8000 -v "$(PWD)/artifacts:/app/artifacts:ro" smartshop-api

smoke:
	python -m smartshop.serving.smoke --url http://localhost:8000

all:  # every step from the raw CSVs (same as python -m smartshop.run_all)
	python -m smartshop.run_all

mitigation:  # v1.1: size-neutral strength, chosen on the tuning period (~10 min)
	python -m smartshop.reporting.mitigation

report:  # fairness check, trial sizing and the dashboard snapshot (~4 min)
	python -m smartshop.reporting.snapshot

dashboard:
	streamlit run dashboards/app.py

sample:  # synthetic warehouse with the real schema, for trying the pipeline without the data
	python -m smartshop.ingest.sample data/warehouse/sample.duckdb
	cd dbt && SMARTSHOP_WAREHOUSE=../data/warehouse/sample.duckdb dbt build --profiles-dir .

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_simulator.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_basket_model.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/04_risk_model.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/05_rescan_policy.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/06_trust_and_targeting.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/07_headline_and_stress_tests.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/08_fairness_and_trial.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/09_unseen_theft_and_size_neutral.ipynb
	jupyter nbconvert --to html notebooks/0*.ipynb --output-dir reports

test:
	pytest -q

lint:
	ruff check src tests dashboards
	ruff format --check src tests dashboards

clean:
	rm -rf data/processed/*.parquet data/warehouse/*.duckdb artifacts artifacts_archive runs
