"""SmartShop Loss Engine dashboard.

Reads only the aggregate snapshot in dashboards/snapshot/ (no member, household or basket
identifiers, no ML libraries), so it runs anywhere, including Streamlit Community Cloud.

    streamlit run dashboards/app.py

Rebuild the snapshot from the data with `python -m smartshop.reporting.snapshot`.
"""

from __future__ import annotations

import inspect
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

# hide the arrow on a "baseline" badge where this Streamlit version supports it
NO_ARROW = (
    {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}
)

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
        "Cancellations & handsets",
        "Model health",
    ]
)

# ------------------------------------------------------------------ Overview
with tabs[0]:
    with st.expander("New here? How to read this dashboard"):
        st.markdown(
            "- **What it is:** when a SmartShop shopper scans the barcode at the till to pay, the "
            "system decides whether the till asks for their basket to be checked (a colleague "
            "then carries out the check). **Today** means the current way (about 12% of shoppers "
            "picked at random); **Recommended** means this project's policy, with the same "
            "colleague time.\n"
            "- **Where the numbers come from:** SmartShop data is not public, so shopper "
            "behaviour is simulated on real grocery baskets (dunnhumby, in US dollars) and "
            "checked against published loss research. The figures show the method works in "
            "that world; a store trial would prove it for real.\n"
            "- **The tabs:** *Policy* shows how a shop's risk becomes a rescan chance; *Store "
            "operations* when rescans happen; *Fairness* whether any group of honest shoppers "
            "is stopped more; *Measuring loss & trial* how to measure loss and test it in "
            "stores; *Robustness* whether the result survives different assumptions; "
            "*Cancellations & handsets* shops that never reach the till; *Model health* the "
            "latest weekly safety checks.\n"
            "- **Two terms:** a **full rescan** checks every item; the **random floor** gives "
            "every shopper at least a 2% chance of one, so nobody can learn they are safe."
        )
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
        "Today: 10% partial checks (about 20% of the basket; if anything is unscanned the till "
        "suggests a full check) and 2% full checks, at random. Recommended: full checks with "
        f"probability from {pol['floor']:.0%} (random floor) to {pol['cap']:.0%} by session "
        f"risk, lowered for trusted regulars; colleague time held at today's "
        f"({today['colleague_hours']:,.0f} hours). Shoppers respond to the policy; trust "
        f"updates over four cycles. Range over runs: loss "
        f"{money(h['policy_min']['loss_remaining'], md=True)}–"
        f"{money(h['policy_max']['loss_remaining'], md=True)}."
    )

    chk = d.get("checks")
    if chk:
        with st.expander("Partial or full check? (v1.4)"):
            tu = pd.DataFrame(chk["tuning"])
            tu["loss vs all-full"] = tu.groupby("colleague_time").loss_remaining.transform(
                lambda x: x / x[tu.loc[x.index, "full_share"] == 1.0].iloc[0] - 1
            )
            tu["staffing"] = tu.colleague_time.map(
                {"today": "Today's colleague time", "half": "Half of it (short-staffed)"}
            )
            tu["full checks for"] = tu.full_share.map(
                lambda v: "none" if v == 0 else "all" if v == 1 else f"top {v:.0%}"
            )
            st.markdown(
                "A **partial check** covers about a fifth of the basket; if it finds anything "
                "unscanned, the check device suggests a **full check**. When the till *targets* "
                "a shopper, "
                "should it ask for a partial check (cheaper, so more shoppers) or a full one? "
                "Below, only the highest-risk share gets a full check and the rest a partial one."
            )
            st.altair_chart(
                alt.Chart(tu)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "full checks for:N",
                        sort=list(tu["full checks for"].unique()),
                        title="Targeted checks that are full (by size-adjusted risk)",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y(
                        "loss vs all-full:Q", axis=alt.Axis(format="%"), title="Loss vs all-full"
                    ),
                    color=alt.Color(
                        "staffing:N",
                        scale=alt.Scale(
                            domain=["Today's colleague time", "Half of it (short-staffed)"],
                            range=[BLUE, ORANGE],
                        ),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=[
                        "staffing",
                        "full checks for",
                        alt.Tooltip("loss vs all-full:Q", format="+.1%"),
                    ],
                )
                .properties(height=260),
                width="stretch",
            )
            st.caption(
                "Tuning period, shoppers respond. Partial checks rarely reach items hidden on "
                "purpose, so a full check of a shopper the model picked out loses least, even "
                "short-staffed. The recommended policy keeps targeted checks full; the 2% random "
                "floor is always a full check, so training and loss measurement stay unbiased."
            )
            ss = chk.get("sample_size")
            if ss:
                ss = pd.DataFrame(ss)
                ss["share"] = ss.rule.str.extract(r"(\d+)%", expand=False).astype(int) / 100
                st.markdown(
                    "**How big should a partial check be?** A bigger sample finds more per check "
                    "but takes longer, so fewer shoppers can be checked in the same colleague "
                    "time. The best size depends on how much a quick check deters a thief, "
                    "which nobody has measured, so three assumptions are shown."
                )
                st.altair_chart(
                    alt.Chart(ss)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "share:Q",
                            axis=alt.Axis(format="%", values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75]),
                            title="Share of the basket a partial check covers",
                        ),
                        y=alt.Y(
                            "loss vs 20% rule:Q",
                            axis=alt.Axis(format=".1%"),
                            title="Loss vs a 20% sample",
                            scale=alt.Scale(zero=False),
                        ),
                        color=alt.Color(
                            "deterrence:N",
                            scale=alt.Scale(
                                domain=[
                                    "thieves fear any check alike",
                                    "a partial check deters half as much",
                                    "deterrence in proportion to items checked",
                                ],
                                range=[GREY, ORANGE, BLUE],
                            ),
                            legend=alt.Legend(orient="right", title=None, labelLimit=320),
                        ),
                        tooltip=[
                            "deterrence",
                            alt.Tooltip("share:Q", format=".0%"),
                            alt.Tooltip("loss vs 20% rule:Q", format="+.2%"),
                            alt.Tooltip("honest stops vs 20% rule:Q", format="+.1%"),
                        ],
                    )
                    .properties(height=340),
                    width="stretch",
                )
                st.caption(
                    "Same colleague time, tuning period. Loss moves by only 1-2%: small samples "
                    "win if thieves fear any check, big ones if they know a sample rarely reaches "
                    "hidden items. **25% of the basket (at least 2 items)** is within 1.4% of the "
                    "best whichever is true, and stops about 6% fewer honest shoppers than 20%. "
                    "Different shares by basket size gained nothing worth the complexity."
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
            f"Trust multiplier {mult:.2f}. If a check is asked for, the colleague who carries "
            "it out sees the reasons (for example "
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
    st.subheader("Rescans through the day")
    hr = pd.DataFrame(d["hourly"])
    hr = hr[(hr.hour >= 7) & (hr.hour <= 22)]  # trading hours; the night has too few shops
    hl = pd.concat(
        [
            hr.assign(
                arm="Today",
                per100=100 * hr.today_rescans_per_day / hr.sessions_per_day,
                share=hr.today_rescans_per_day / hr.today_rescans_per_day.sum(),
            ),
            hr.assign(
                arm="Recommended",
                per100=100 * hr.policy_rescans_per_day / hr.sessions_per_day,
                share=hr.policy_rescans_per_day / hr.policy_rescans_per_day.sum(),
            ),
        ]
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Rescans per 100 SmartShop shops, by hour**")
        st.altair_chart(
            alt.Chart(hl)
            .mark_line(point=True)
            .encode(
                x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("per100:Q", title="Rescans per 100 shops", scale=alt.Scale(zero=True)),
                color=alt.Color("arm:N", scale=ARMS, title=None),
                tooltip=["hour", "arm", alt.Tooltip("per100:Q", format=".1f")],
            )
            .properties(height=300),
            width="stretch",
        )
    with right:
        st.markdown("**Share of the day's rescans, by hour (for staffing)**")
        st.altair_chart(
            alt.Chart(hl)
            .mark_bar()
            .encode(
                x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share of rescans"),
                color=alt.Color("arm:N", scale=ARMS, title=None),
                xOffset="arm:N",
                tooltip=["hour", "arm", alt.Tooltip("share:Q", format=".1%")],
            )
            .properties(height=300),
            width="stretch",
        )
    st.caption(
        "Today 12 in every 100 shops are checked at random, mostly with partial checks of about "
        "a fifth of the basket. "
        "The recommended policy rescans about 3 to 5 in 100, all full rescans, so it uses the "
        "same colleague time on fewer, more thorough checks aimed at the riskiest shops. The rate "
        "is much the same through the busy hours: the policy does not use time of day, so evening "
        "shoppers are not singled out, and colleague demand follows footfall exactly as today "
        "(peak around 5pm)."
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

# ------------------------------------------------------------------ Cancellations & handsets
with tabs[6]:
    st.subheader("Sessions that never reach the till")
    ex = d.get("exits")
    if not ex:
        st.info("No v1.5 analysis in this snapshot.")
    else:
        st.write(
            "**Genuine cancellations:** a shopper gets an urgent call or can't find what they "
            "came for, cancels SmartShop and leaves the items. **Cancel-and-walk:** a shoplifter "
            "scans so the session looks normal, cancels, pays nothing and slips through the "
            "self-scan gates behind a customer who scanned their receipt (easier when busy). The "
            "whole basket is lost and no check ever sees it. Both can happen early or late, so "
            "timing does not tell them apart. **Dead handsets** are the store's equipment "
            "failing: today the customer has to scan everything again."
        )
        st.caption(
            "What the store has: colleagues carry no device; the SmartShop check device at the "
            "SmartShop checkout logs in and scans the barcode a shopper gets for a check; "
            "customers pick their own handset from the stand, or use their phone, and start with "
            "their Nectar card. v1.5 uses only that."
        )
        summ = ex["summary"]
        st.caption(
            f"Today cancel-and-walk is about {summ['walk_out_share_of_all_loss_today']:.0%} of "
            "all loss, all of it invisible to checks."
        )
        et = pd.DataFrame(ex["table"])
        world = st.selectbox("Assumptions", list(et.world.unique()))
        w = et[et.world == world].set_index("arm")
        today_row = w.loc["Today"]
        cols = st.columns(len(w))
        for col, arm in zip(cols, w.index, strict=True):
            r = w.loc[arm]
            col.markdown(f"**{arm}**")
            change = r["walk-out loss"] / today_row["walk-out loss"] - 1
            same = arm == "Today" or abs(change) < 0.005
            col.metric(
                "Cancel-and-walk loss",
                money(r["walk-out loss"]),
                ("baseline" if arm == "Today" else "no change")
                if same
                else f"{change:.0%} vs today",
                delta_color="off" if same else "inverse",
                **(NO_ARROW if same else {}),
            )
            col.metric(
                "Genuine cancels finished at the point",
                f"{r['genuine cancels finished at the SmartShop point']:,.0f}",
            )
            col.metric(
                "Honest members given a full check",
                f"{r['honest members checked this way']:,.0f}",
            )
            col.metric("Recovered by those checks", money(r["recovered by these checks"]))
        st.caption(
            "**Idea 1:** cancelling with items scanned asks the shopper to leave the basket at the "
            "SmartShop checkout and shows a cancel barcode, which a colleague scans on the check "
            "device. Genuine shoppers are helped and the items go back; a cancel never finished "
            "there is an *unfinished cancel*, which the store can now count. **Idea 2:** the "
            "member's next SmartShop shop that reaches the till gets a full check, with a neutral "
            "reason. After one unfinished cancel most of those checks fall on honest shoppers who "
            "forgot, so the recommended rule is **two in 90 days**. Walk-outs fall only if idea 2 "
            "deters them, which is unmeasured: pick a world above to see 25% or 50%."
        )
        st.markdown("**Dead handsets: the store's problem, not the customer's**")
        hs = pd.DataFrame(ex["handsets"])
        st.altair_chart(
            alt.Chart(hs)
            .mark_bar(color=BLUE)
            .encode(
                y=alt.Y("policy:N", sort=None, title=None, axis=alt.Axis(labelLimit=360)),
                x=alt.X("failures per 1,000 sessions:Q", title="Dead handsets per 1,000 sessions"),
                tooltip=[
                    "policy",
                    alt.Tooltip("failures per 1,000 sessions:Q", format=".1f"),
                    alt.Tooltip("min:Q", title="lowest over 5 fleets", format=".1f"),
                    alt.Tooltip("max:Q", title="highest over 5 fleets", format=".1f"),
                ],
            )
            .properties(height=180),
            width="stretch",
        )
        st.caption(
            "Customers pick any handset, and the handsets show no battery level. From the "
            "handset log alone, **A** sends handsets that die twice in 30 days for a new battery: "
            "that removes most dead handsets, because worn batteries cause most of them. **B** "
            f"has a handset back on the stand for under {summ.get('screen_minutes', 90):.0f} "
            "minutes ask the customer to take another; it helps on its own but adds little to A. "
            "When one still dies, **D** resumes the session on another handset or at a till from "
            "the Nectar card: nothing is scanned twice, and it is never a risk signal."
        )
        fr = pd.DataFrame(ex["fairness"])
        if not fr.empty:
            st.markdown("**Idea 2: honest members given a full check, by group**")
            st.dataframe(
                fr.rename(columns={"ratio": "four-fifths ratio"}).style.format(
                    {"lowest": "{:.1%}", "highest": "{:.1%}", "four-fifths ratio": "{:.2f}"}
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Under the two-in-90-days rule only 1-3% of honest members in any group are ever "
                "checked this way, so the ratios move a lot with a handful of members: monitor "
                "them in a trial rather than read them as a gap."
            )

# ------------------------------------------------------------------ Model health
with tabs[7]:
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
