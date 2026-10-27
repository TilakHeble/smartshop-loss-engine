"""Rescan decision API: called when a SmartShop shopper finishes and goes to pay.

    GET  /health                 process is up
    GET  /ready                  a bundle is loaded (Docker HEALTHCHECK)
    GET  /model                  bundle version, policy, trust rule, budget
    GET  /members/{id}/trust     the trust multiplier and clean-rescan count a member has earned
    POST /decide                 one exit decision: rescan or not, with probability and reasons

Every decision returns the probability it was made with and the random draw; both must be logged
with the rescan outcome, because loss is estimated afterwards by inverse-probability weighting
(notebook 07). Decisions are also appended to a JSONL decision log when SMARTSHOP_DECISION_LOG
is set.

The bundle hot-reloads: the weekly job publishes a new bundle atomically and the API picks it up
on the next request when the manifest's modification time changes; no restart, no downtime.

Run:
    uvicorn smartshop.serving.app:app --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from smartshop.serving.bundle import DEFAULT_DIR, Bundle

log = logging.getLogger("smartshop.api")
BUNDLE_DIR = Path(os.environ.get("SMARTSHOP_BUNDLE", DEFAULT_DIR))
DECISION_LOG = os.environ.get("SMARTSHOP_DECISION_LOG")
MAX_EVENTS = 1000


class ScanEvent(BaseModel):
    t: float = Field(ge=0, description="seconds since the session started")
    action: Literal["scan", "delete"]
    product_id: int
    unit_price: float = Field(ge=0)


class DecideRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    member_id: int
    duration_min: float = Field(gt=0, le=600)
    events: list[ScanEvent] = Field(default_factory=list, max_length=MAX_EVENTS)
    draw: float | None = Field(
        default=None, ge=0, lt=1, description="testing only: fix the random draw"
    )


class DecideResponse(BaseModel):
    session_id: str
    decision: Literal["full_rescan", "no_rescan"]
    rescan_probability: float
    draw: float
    audit_sample: bool
    risk_score: float
    reasons: list[str]
    look_here_first: list[str]
    trust_multiplier: float
    known_member: bool
    unknown_products: int
    bundle_version: str


class State:
    def __init__(self) -> None:
        self.bundle: Bundle | None = None
        self.mtime: float | None = None
        self.lock = threading.Lock()

    def maybe_reload(self) -> None:
        manifest = BUNDLE_DIR / "manifest.json"
        try:
            mtime = manifest.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self.mtime:
            return
        with self.lock:
            if mtime == self.mtime:
                return
            try:
                self.bundle = Bundle.load(BUNDLE_DIR)
                self.mtime = mtime
                log.info("loaded bundle %s", self.bundle.manifest["bundle_version"])
            except Exception:  # keep serving the old bundle if a new one is unreadable
                log.exception("bundle load failed; keeping the current one")


state = State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.maybe_reload()
    yield


app = FastAPI(title="SmartShop rescan decisions", version="0.9.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{1000 * (time.perf_counter() - start):.1f}"
    return response


def _bundle() -> Bundle:
    state.maybe_reload()
    if state.bundle is None:
        raise HTTPException(503, "no model bundle loaded")
    return state.bundle


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    b = _bundle()
    return {"status": "ready", "bundle_version": b.manifest["bundle_version"]}


@app.get("/model")
def model() -> dict:
    m = _bundle().manifest
    keys = [
        "bundle_version",
        "built_at",
        "cutoff_day",
        "features",
        "training_labels",
        "policy",
        "trust_rule",
        "budget",
        "members_profiled",
        "members_trusted",
        "library_versions",
    ]
    return {k: m.get(k) for k in keys}


@app.get("/members/{member_id}/trust")
def member_trust(member_id: int) -> dict:
    b = _bundle()
    if member_id not in b.profiles.index:
        return {
            "member_id": member_id,
            "known_member": False,
            "trust_multiplier": 1.0,
            "clean_full_rescans": 0,
        }
    p = b.profiles.loc[member_id]
    return {
        "member_id": member_id,
        "known_member": True,
        "trust_multiplier": float(p.trust_mult),
        "clean_full_rescans": int(p.prior_full_rescans - p.prior_full_found),
    }


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    b = _bundle()
    events = pd.DataFrame(
        [e.model_dump() for e in req.events], columns=["t", "action", "product_id", "unit_price"]
    ).sort_values("t", kind="stable")
    out = b.decide(req.member_id, req.duration_min, events, draw=req.draw)
    resp = DecideResponse(
        session_id=req.session_id, bundle_version=b.manifest["bundle_version"], **out
    )
    if DECISION_LOG:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "member_id": req.member_id,
            **resp.model_dump(),
        }
        with open(DECISION_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    return resp
