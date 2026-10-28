"""SmartShop Loss Engine dashboard.

Reads only the aggregate snapshot in dashboards/snapshot/ (no member, household or basket
identifiers, no ML libraries), so it runs anywhere, including Streamlit Community Cloud.

    streamlit run dashboards/app.py

Rebuild the snapshot from the data with `python -m smartshop.reporting.snapshot`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import altair as alt
import pandas as pd
import streamlit as st

SNAP = Path(__file__).parent / "snapshot" / "snapshot.json"
BLUE, ORANGE, GREY, GREEN, AMBER = "#2a78d6", "#eb6834", "#a8a7a2", "#1baf7a", "#eda100"
ARMS = alt.Scale(domain=["Today", "Recommended"], range=[GREY, BLUE])
VERDICT_COLOURS = {
    "pass": GREEN,
    "gap within chance for groups this small": GREEN,
    "gap explained by basket size": AMBER,
    "review": ORANGE,
}

st.set_page_config(page_title="SmartShop Loss Engine", page_icon="🛒", layout="wide")


@st.cache_data
def load() -> dict:
    return json.loads(SNAP.read_text())


def money(x: float, md: bool = False) -> str:
    """Dollar amount; md=True escapes $ so Streamlit markdown does not read it as maths."""
    sign, dollar = ("−" if x < 0 else ""), ("\\$" if md else "$")
    return f"{sign}{dollar}{abs(x):,.0f}"


GROUP_NAMES = {"False": "established", "True": "new", "0.0": "other hours", "1.0": "4–7pm"}


def natural(g: str):
    digits = "".join(c if c.isdigit() else " " for c in g).split()
    return ([int(x) for x in digits] or [0], g)


def smooth(score: float, t: float, floor: float, cap: float, tau: float) -> float:
    s = min(max(score, 1e-6), 1 - 1e-6)
    z = max(-50.0, min(50.0, (math.log(s / (1 - s)) - t) / tau))
    return floor + (cap - floor) / (1 + math.exp(-z))


def stores_per_arm(effect: float, cv: float, rho: float, alpha=0.05, power=0.8) -> int:
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return math.ceil(2 * z**2 * cv**2 * (1 - rho**2) / effect**2)


d = load()
h, pol, period = d["headline"], d["policy"], d["period"]
today, rec = h["today"], h["policy"]

st.title("SmartShop Loss Engine")
st.caption(
    "Risk-based SmartShop rescans with a random floor, Nectar-linked trust and unbiased loss "
    f"measurement · {d['note']} · test period {period['days']} days, "
    f"{period['sessions']:,} sessions, {period['stores']} stores, mean of {period['runs']} runs"
)

tabs = st.tabs(
    [
        "Overview",
        "Policy",
        "Store operations",
        "Fairness",
        "Measuring loss & trial",
        "Robustness",
        "Model health",
    ]
)

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.subheader("Same colleague time, less loss, far fewer honest shoppers stopped")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Loss remaining",
        money(rec["loss_remaining"]),
        f"{rec['loss_remaining'] / today['loss_remaining'] - 1:.0%} vs today",
        delta_color="inverse",
    )
    c2.metric(
        "Value recovered by rescans",
        money(rec["recovered"]),
        f"{rec['recovered'] / today['recovered']:.1f}x today",
    )
    c3.metric(
        "Recovered per colleague hour",
        money(rec["recovered"] / rec["colleague_hours"]),
        f"today {money(today['recovered'] / today['colleague_hours'])}",
    )
    c4.metric(
        "Honest shoppers stopped",
        f"{rec['honest_stops']:,.0f}",
        f"{rec['honest_stops'] / today['honest_stops'] - 1:.0%} vs today",
        delta_color="inverse",
    )
    v10 = d.get("v1_0")
    if v10:
        st.info(
            f"**Policy v{d['policy_version']}.** Risk is compared with sessions of a similar "
            "basket size before it sets the rescan chance. Against v1.0 (week 8): loss "
            f"remaining {money(v10['loss_remaining'], md=True)} → "
            f"{money(rec['loss_remaining'], md=True)}, recovered "
            f"{money(v10['recovered'], md=True)} → {money(rec['recovered'], md=True)}, "
            f"honest shoppers stopped {v10['honest_stops']:,.0f} → {rec['honest_stops']:,.0f}. "
            "It also narrows the age gap found in the week 10 fairness check (Fairness tab)."
        )
    st.caption(
        f"Today: 10% partial and 2% full rescans at random. Recommended: full rescans with "
        f"probability from {pol['floor']:.0%} (random floor) to {pol['cap']:.0%} by session "
        f"risk, lowered for trusted regulars; colleague time held at today's "
        f"({today['colleague_hours']:,.0f} hours). Shoppers respond to the policy; trust "
        f"updates over four cycles. Range over runs: loss "
        f"{money(h['policy_min']['loss_remaining'], md=True)}–"
        f"{money(h['policy_max']['loss_remaining'], md=True)}."
    )

    w = pd.DataFrame(d["weekly"])
    w = w[w.sessions >= 0.5 * w.sessions.median()]  # drop the final partial week
    long = pd.concat(
        [
            w.assign(arm="Today", recovered=w.today_recovered, stops=w.today_honest_stops),
            w.assign(arm="Recommended", recovered=w.policy_recovered, stops=w.policy_honest_stops),
        ]
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Value recovered per week**")
        st.altair_chart(
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:Q", title="Week of test period"),
                y=alt.Y("recovered:Q", title="$ recovered"),
                color=alt.Color("arm:N", scale=ARMS, title=None),
                tooltip=["week", "arm", alt.Tooltip("recovered:Q", format="$,.0f")],
            )
            .properties(height=260),
            width="stretch",
        )
    with right:
        st.markdown("**Honest shoppers stopped per week**")
        st.altair_chart(
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:Q", title="Week of test period"),
                y=alt.Y("stops:Q", title="Honest shoppers stopped"),
                color=alt.Color("arm:N", scale=ARMS, title=None),
                tooltip=["week", "arm", alt.Tooltip("stops:Q", format=",.0f")],
            )
            .properties(height=260),
            width="stretch",
        )

    st.markdown("**Where the rescans go, and what they find**")
    b = pd.DataFrame(d["bands"])
    bl = b.melt(
        id_vars="band",
        value_vars=["sessions", "rescans", "found"],
        var_name="share of",
        value_name="share",
    )
    bl["share of"] = bl["share of"].map(
        {"sessions": "sessions", "rescans": "rescans", "found": "value found"}
    )
    st.altair_chart(
        alt.Chart(bl)
        .mark_bar()
        .encode(
            y=alt.Y("band:N", title="Rescan probability", sort=list(b.band)),
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color(
                "share of:N",
                scale=alt.Scale(
                    domain=["sessions", "rescans", "value found"], range=[GREY, BLUE, GREEN]
                ),
            ),
            yOffset="share of:N",
            tooltip=["band", "share of", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=260),
        width="stretch",
    )
    top = b.iloc[-1]
    st.caption(
        f"The highest-risk {top.sessions:.1%} of sessions receive {top.rescans:.0%} of rescans "
        f"and hold {top.found:.0%} of the value found. The random floor keeps every session "
        "possible to rescan, which deters and gives an unbiased audit sample."
    )

# ------------------------------------------------------------------ Policy
with tabs[1]:
    st.subheader("How a session's risk becomes a rescan probability")
    st.write(
        "The rescan probability rises smoothly with the risk score from the random floor to "
        "the cap. The score is first compared with sessions of a similar basket size, because "
        "big baskets almost always hold an honest slip and are the slowest to rescan. "
        "Regulars whose recent full rescans were clean get a trust discount (25% per "
        "clean rescan, up to 75%), but never below the floor. Every decision is a random draw "
        "against this probability, so no shopper can learn a safe pattern."
    )
    curve = pd.DataFrame(d["curve"]).melt(id_vars="score", var_name="trust", value_name="p")
    curve["trust"] = curve.trust.str.replace("trust ", "multiplier ")
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(
            alt.Chart(curve)
            .mark_line()
            .encode(
                x=alt.X("score:Q", title="Size-adjusted risk score"),
                y=alt.Y("p:Q", axis=alt.Axis(format="%"), title="Full-rescan probability"),
                color=alt.Color(
                    "trust:N",
                    title="Trust",
                    scale=alt.Scale(
                        domain=[f"multiplier {m}" for m in ("1.0", "0.75", "0.5", "0.25")],
                        range=[BLUE, "#6250d6", "#e87ba4", GREY],
                    ),
                ),
            )
            .properties(height=320),
            width="stretch",
        )
    with right:
        st.markdown("**Try one exit decision**")
        score = st.slider("Size-adjusted risk score", 0.01, 0.99, 0.7, 0.01)
        clean = st.select_slider("Recent clean full rescans", options=[0, 1, 2, 3], value=0)
        mult = 1 - 0.25 * clean
        p = max(pol["floor"], smooth(score, pol["t"], pol["floor"], pol["cap"], pol["tau"]) * mult)
        st.metric("Full-rescan probability", f"{p:.1%}")
        st.caption(
            f"Trust multiplier {mult:.2f}. A colleague sees the reasons (for example "
            "'long pause between scans') and the shopper's usual high-risk items that are "
            "missing, to look at first."
        )
    tc = pd.DataFrame(d["trust_cycles"])
    st.markdown("**Trust over four rescan cycles**")
    c1, c2 = st.columns(2)
    c1.metric(
        "Members with a trust discount (last cycle)",
        f"{tc.members_trusted.iloc[-1]:,.0f}",
        f"from {tc.members_trusted.iloc[0]:,.0f} in cycle 1",
    )
    c2.metric(
        "Share of the discount that goes to dishonest shoppers",
        f"{tc.trust_discount_to_thieves.mean():.0%}",
    )

# ------------------------------------------------------------------ Operations
with tabs[2]:
    st.subheader("Rescans through the day, per store")
    hr = pd.DataFrame(d["hourly"])
    n = period["stores"]
    hl = pd.concat(
        [
            hr.assign(arm="Today", rescans=hr.today_rescans_per_day / n),
            hr.assign(arm="Recommended", rescans=hr.policy_rescans_per_day / n),
        ]
    )
    st.altair_chart(
        alt.Chart(hl)
        .mark_bar()
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("rescans:Q", title="Rescans per store per day"),
            color=alt.Color("arm:N", scale=ARMS, title=None),
            xOffset="arm:N",
            tooltip=["hour", "arm", alt.Tooltip("rescans:Q", format=".2f")],
        )
        .properties(height=320),
        width="stretch",
    )
    st.caption(
        "Fewer rescans than today, because the recommended policy uses only full rescans "
        "(each takes longer than a partial one) at the same colleague time. Peaks follow "
        "footfall; the policy does not use time of day, so evening shoppers are not singled "
        "out. Simulated stores are small; a real store scales with its SmartShop sessions."
    )

# ------------------------------------------------------------------ Fairness
with tabs[3]:
    st.subheader("Honest shoppers: who is asked for a rescan?")
    st.write(
        "Being asked for a full rescan when honest is the harm. Demographics are used only "
        "here, never by the model or the policy. The four-fifths ratio is the lowest group's "
        "rate divided by the highest; the permutation test shows the ratio random groups of the "
        "same sizes would give, because small groups make the ratio fall by chance alone."
    )
    fg = pd.DataFrame(d["fairness_groups"])
    attr = st.selectbox("Attribute", fg.attribute.unique())
    sub = fg[fg.attribute == attr].copy()
    sub["group"] = sub.group.map(lambda g: GROUP_NAMES.get(g, g))
    order = sorted(sub.group, key=natural)
    bars = (
        alt.Chart(sub)
        .mark_bar(color=BLUE)
        .encode(
            x=alt.X("group:N", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("policy:Q", axis=alt.Axis(format="%"), title="Honest sessions stopped"),
            tooltip=[
                "group",
                alt.Tooltip("policy:Q", format=".1%"),
                "households",
                "honest sessions",
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [d["today_rescan_rate"]]}))
        .mark_rule(color=ORANGE, strokeDash=[6, 4])
        .encode(y="y:Q")
    )
    st.altair_chart((bars + rule).properties(height=300), width="stretch")
    st.caption("Dashed line: today's rate (12% of every shopper, at random).")
    f = pd.DataFrame(d["fairness"]).set_index("attribute")
    show = f[
        [
            "four-fifths ratio",
            "ratio, same basket sizes",
            "ratio if groups were random (median)",
            "chance of a gap this large",
            "highest rate vs today",
            "fewest households in a group",
            "verdict",
        ]
    ]
    st.dataframe(
        show.style.format({c: "{:.2f}" for c in show.columns[:5]}).map(
            lambda v: f"color: {VERDICT_COLOURS.get(v, 'inherit')}", subset=["verdict"]
        ),
        width="stretch",
    )
    mit = d.get("mitigation")
    if mit:
        st.markdown("**The fix for the week 10 age gap: compare like with like**")
        st.write(
            "Raw risk scores favour big baskets, which younger family households buy. Comparing "
            "each session only with sessions of a similar size closes the gap, and also catches "
            "more: rescans move from slow big baskets full of honest slips to sessions that look "
            "unusual for their size. Strength was chosen on the tuning period (0.5); this is the "
            "test period."
        )
        mt = pd.DataFrame(mit)
        mt = mt[
            [
                "strength",
                "loss reduction vs today",
                "recovery multiple",
                "honest stops",
                "age group: four-fifths ratio",
                "age group: verdict",
                "new SmartShop user (<5 sessions): four-fifths ratio",
            ]
        ].rename(
            columns={
                "strength": "size-neutral strength",
                "age group: four-fifths ratio": "age ratio",
                "age group: verdict": "age verdict",
                "new SmartShop user (<5 sessions): four-fifths ratio": "new-user ratio",
            }
        )
        st.dataframe(
            mt.style.format(
                {
                    "loss reduction vs today": "{:.1%}",
                    "recovery multiple": "{:.1f}x",
                    "honest stops": "{:,.0f}",
                    "age ratio": "{:.2f}",
                    "new-user ratio": "{:.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Strength 1 closes the age gap fully but stops more honest shoppers and pushes new "
            "users to the edge of the four-fifths rule (0.80); 0.5 balances the three. Every "
            "group is still stopped far less than today."
        )

# ------------------------------------------------------------------ Measurement & trial
with tabs[4]:
    st.subheader("Measuring loss once rescans are targeted")
    m = pd.DataFrame(d["measurement"])
    st.dataframe(
        m.rename(
            columns={
                "method": "Estimate of total loss",
                "bias": "Bias",
                "error": "Typical error per period",
            }
        ).style.format({"Bias": "{:+.1%}", "Typical error per period": "{:.0%}"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Targeted rescans are not a random sample: averaging what they find overstates loss "
        "several times over. Weighting each rescan by one over its probability (logged with "
        "every decision) is unbiased; the 2% random floor alone is the cross-check."
    )
    st.subheader("Sizing a store-level randomised trial")
    tr = d["trial"]
    c1, c2, c3, c4 = st.columns(4)
    effect = c1.slider("Effect to detect", 0.05, 0.25, tr["effect"], 0.01, format="%.2f")
    between = c2.slider("Between-store spread (CV)", 0.05, 0.6, 0.25, 0.05)
    rho = c3.slider("Baseline-trial correlation", 0.0, 0.95, 0.7, 0.05)
    sessions = c4.number_input(
        "SmartShop sessions per store in trial", 2_000, 200_000, tr["sessions_per_store"], 1_000
    )
    meas = tr["measurement_cv"] * math.sqrt(tr["sessions_per_store"] / sessions)
    cv = math.sqrt(between**2 + meas**2)
    st.metric("Stores per arm (80% power, 5% significance)", f"{stores_per_arm(effect, cv, rho):,}")
    st.caption(
        f"Stores matched in pairs on baseline loss rate and randomised; loss measured the same "
        f"way in both arms from random full rescans (measurement error {meas:.0%} per store, "
        "scaled from the simulation); analysed with each store's baseline as a covariate."
    )

# ------------------------------------------------------------------ Robustness
with tabs[5]:
    st.subheader("Does the conclusion survive if the simulator is wrong?")
    s = pd.DataFrame(d["sensitivity"])
    has_v11 = "v1.1: loss reduction" in s
    sort_col = "v1.1: loss reduction" if has_v11 else "loss reduction"
    order = list(s.sort_values(sort_col).scenario)
    parts = [s.assign(policy="v1.0", reduction=s["loss reduction"])]
    if has_v11:
        parts.append(s.assign(policy="v1.1", reduction=s["v1.1: loss reduction"]))
    sl = pd.concat(parts)
    st.altair_chart(
        alt.Chart(sl)
        .mark_bar()
        .encode(
            y=alt.Y("scenario:N", sort=order, title=None, axis=alt.Axis(labelLimit=360)),
            x=alt.X("reduction:Q", axis=alt.Axis(format="%"), title="Loss reduction vs today"),
            color=alt.Color(
                "policy:N", scale=alt.Scale(domain=["v1.0", "v1.1"], range=[GREY, BLUE])
            ),
            yOffset="policy:N",
            tooltip=["scenario", "policy", alt.Tooltip("reduction:Q", format=".1%")],
        )
        .properties(height=520),
        width="stretch",
    )
    st.caption(
        "Each scenario changes one assumption and reruns everything: simulate, retrain the risk "
        "model, refit the policy, compare at today's colleague time (single-cycle evaluation, "
        "without trust). Barcode switching is a theft method the model was not designed "
        "around: it scans a cheap item's barcode for an expensive one, so the count, pace and "
        "deletions look normal. 'Not retrained' uses the base world's model and policy."
    )

# ------------------------------------------------------------------ Model health
with tabs[6]:
    st.subheader("Latest weekly job")
    job = d.get("weekly_job")
    if not job:
        st.info("No weekly run recorded in this snapshot.")
    else:
        st.write(
            f"Candidate `{job['candidate']}` · "
            f"{'published' if job['published'] else 'not published'} · {job['run_at'][:16]}"
        )
        checks = pd.DataFrame(
            [
                {
                    "gate": k,
                    "value": v["value"],
                    "rule": v["rule"],
                    "result": "pass" if v["pass"] else "FAIL",
                }
                for k, v in job["checks"].items()
            ]
        )
        checks["value"] = checks.value.map(
            lambda v: (
                "too few to test" if v is None else f"{v:,.3f}" if isinstance(v, float) else str(v)
            )
        )
        st.dataframe(checks, hide_index=True, width="stretch")
        drift = pd.DataFrame(job["drift"])
        if not drift.empty:
            st.markdown("**Drift (PSI) against the live bundle's last cycle**")
            st.altair_chart(
                alt.Chart(drift)
                .mark_bar(color=BLUE)
                .encode(
                    y=alt.Y("feature:N", sort="-x", title=None),
                    x=alt.X(
                        "psi:Q",
                        title="PSI (warn 0.10, alert 0.25)",
                        scale=alt.Scale(domain=[0, max(0.3, drift.psi.max())]),
                        axis=alt.Axis(tickCount=6),
                    ),
                )
                .properties(height=280)
                + alt.Chart(pd.DataFrame({"x": [0.10, 0.25]}))
                .mark_rule(color=ORANGE, strokeDash=[4, 4])
                .encode(x="x:Q"),
                width="stretch",
            )
