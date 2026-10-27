# Deploying the dashboard and the API

## Dashboard on Streamlit Community Cloud (free, about 3 minutes)

The dashboard reads only `dashboards/snapshot/snapshot.json` (aggregates, no member data), so it
needs no data or secrets.

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns this repo.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `TilakHeble/smartshop-loss-engine`, branch `main`, main file path
   `dashboards/app.py`. Under **Advanced settings**, choose Python 3.12.
4. **Deploy.** Streamlit installs `dashboards/requirements.txt` (streamlit, pandas, altair), which
   sits next to the app, so it does not try to install the full project.
5. Copy the app URL into the README (the "Live dashboard" link at the top).

To refresh it after new results: `make report` (rewrites the snapshot), commit, push. The app
redeploys itself.

## Decision API with Docker

```bash
make weekly                 # or: make demo-bundle (synthetic bundle, no data needed)
docker build -t smartshop-api .
docker run -d -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" smartshop-api
python -m smartshop.serving.smoke --url http://localhost:8000
```

Interactive API docs at <http://localhost:8000/docs>. Set `SMARTSHOP_DECISION_LOG=/path/decisions.jsonl`
to log every decision with its probability and draw (needed for loss measurement). A new weekly
bundle written to `artifacts/` is picked up without a restart.
