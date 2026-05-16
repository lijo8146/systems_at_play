"""
app.py — Interactive Streamlit dashboard for the pandemic simulator.

Run with:
    streamlit run app.py

Features
────────
• Role picker in the sidebar (checkboxes for each player role)
• Configurable simulation parameters (expandable)
• Animated city-network graph that updates step-by-step
• Live infection timeline with per-city curves
• Real-time metrics: cure progress, total infected, outbreak count
• Monte Carlo quick-compare panel (optional, runs in sidebar)
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation import (
    build_world_network,
    InfectionSimulator,
    InfectionParams,
    Medic,
    Scientist,
    QuarantineSpecialist,
    Epidemiologist,
    Dispatcher,
    ALL_ROLES,
)
from simulation.roles import PlayerRole
from analysis.monte_carlo import run_experiment, analyse, build_scenarios

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_STEPS = 80
CITY_POSITIONS: dict[str, tuple[float, float]] = {
    "Atlanta":   (-2.0,  0.3),
    "New York":  (-1.2,  0.8),
    "São Paulo": (-1.5, -1.5),
    "London":    ( 0.0,  1.2),
    "Paris":     ( 0.4,  0.9),
    "Lagos":     ( 0.3, -0.8),
    "Cairo":     ( 1.1,  0.4),
    "Beijing":   ( 2.5,  0.9),
    "Tokyo":     ( 3.2,  0.6),
    "Sydney":    ( 3.0, -1.3),
}

ROLE_META = {
    "Scientist":             {"emoji": "🔬", "cls": Scientist,            "needs_city": False},
    "Medic":                 {"emoji": "💊", "cls": Medic,                "needs_city": True},
    "Epidemiologist":        {"emoji": "📊", "cls": Epidemiologist,       "needs_city": False},
    "Dispatcher":            {"emoji": "📡", "cls": Dispatcher,           "needs_city": False},
    "Quarantine Specialist": {"emoji": "🔒", "cls": QuarantineSpecialist, "needs_city": False},
}

CITY_NAMES = list(CITY_POSITIONS.keys())

# ─────────────────────────────────────────────────────────────────────────────
#  Simulation runner  (caches frames in session_state)
# ─────────────────────────────────────────────────────────────────────────────

def run_and_capture(
    roles: list[PlayerRole],
    seed: int,
    params: InfectionParams,
) -> list[dict]:
    """
    Run the full simulation, returning a list of per-step snapshots.

    Each snapshot is a plain dict so it can live in session_state without
    carrying any live object references.
    """
    G, cities = build_world_network()
    sim = InfectionSimulator(G, cities, params=params, seed=seed)
    sim.seed_infection("Beijing", count=5)

    def snapshot() -> dict:
        return {
            "step":           sim.step_count,
            "cure_progress":  sim.cure_progress,
            "total_infected": sim.total_currently_infected,
            "outbreaks":      len(sim.outbreaks),
            "cities": {
                name: {
                    "infected":    c.infected,
                    "recovered":   c.recovered,
                    "susceptible": c.susceptible,
                    "quarantined": c.quarantined,
                    "infection_rate": c.infection_rate,
                }
                for name, c in sim.cities.items()
            },
        }

    frames = [snapshot()]          # step-0 state (pre-spread)
    for _ in range(MAX_STEPS):
        done = sim.step(roles=roles)
        frames.append(snapshot())
        if done:
            break

    frames[-1]["outcome"] = "WIN 🏆" if sim.is_cured() else "LOSS 💀"
    frames[-1]["summary"] = sim.summary()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
#  Plotly figures
# ─────────────────────────────────────────────────────────────────────────────

def make_network_fig(frame: dict) -> go.Figure:
    """Render the city network for one simulation frame."""
    G, _ = build_world_network()          # lightweight — just for edge list
    cities_data = frame["cities"]

    # Edge traces
    traces: list[go.BaseTraceType] = []
    for u, v, data in G.edges(data=True):
        x0, y0 = CITY_POSITIONS[u]
        x1, y1 = CITY_POSITIONS[v]
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=data.get("weight", 0.5) * 3, color="rgba(120,120,120,0.25)"),
            hoverinfo="none",
            showlegend=False,
        ))

    # Node trace
    xs, ys, colors, sizes, hover_texts, labels = [], [], [], [], [], []
    for name in CITY_NAMES:
        c = cities_data[name]
        x, y = CITY_POSITIONS[name]
        xs.append(x)
        ys.append(y)
        colors.append(min(c["infection_rate"] * 4, 1.0))
        # Find original pop from susceptible + infected + recovered
        pop_proxy = c["susceptible"] + c["infected"] + c["recovered"]
        sizes.append(max(18, pop_proxy ** 0.38 * 0.55))
        q_tag = " 🔒" if c["quarantined"] else ""
        hover_texts.append(
            f"<b>{name}{q_tag}</b><br>"
            f"Infected:    {c['infected']:,}<br>"
            f"Recovered:   {c['recovered']:,}<br>"
            f"Susceptible: {c['susceptible']:,}<br>"
            f"Rate:        {c['infection_rate']*100:.2f} %"
        )
        labels.append(name)

    traces.append(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale="YlOrRd",
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title="Infection rate", thickness=14, len=0.7),
            line=dict(width=2, color="white"),
        ),
        text=labels,
        textposition="top center",
        textfont=dict(size=9, color="white"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    ))

    cure_pct = frame["cure_progress"] * 100
    step = frame["step"]
    outcome = frame.get("outcome", "")
    title_text = (
        f"Step {step} / {MAX_STEPS}  ·  "
        f"Cure: {cure_pct:.1f} %  ·  "
        f"Infected: {frame['total_infected']:,}  "
        f"{outcome}"
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=10, l=5, r=5, t=45),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.8, 4.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.1, 1.9]),
        plot_bgcolor="rgba(15,15,25,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    return fig


def make_timeline_fig(frames: list[dict], current_step: int) -> go.Figure:
    """Per-city infected-count curves up to current_step."""
    steps = [f["step"] for f in frames[: current_step + 1]]
    fig = go.Figure()

    colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    ]
    for i, city in enumerate(CITY_NAMES):
        vals = [f["cities"][city]["infected"] for f in frames[: current_step + 1]]
        fig.add_trace(go.Scatter(
            x=steps, y=vals,
            mode="lines",
            name=city,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"<b>{city}</b><br>Step %{{x}}<br>Infected: %{{y:,}}<extra></extra>",
        ))

    # Cure progress as secondary axis
    cure_vals = [f["cure_progress"] * 100 for f in frames[: current_step + 1]]
    fig.add_trace(go.Scatter(
        x=steps, y=cure_vals,
        mode="lines",
        name="Cure progress %",
        line=dict(color="#00e5a0", width=2.5, dash="dot"),
        yaxis="y2",
        hovertemplate="Cure: %{y:.1f} %<extra></extra>",
    ))

    fig.update_layout(
        title="Infection timeline by city",
        xaxis=dict(title="Week (step)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Infected citizens", gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(
            title="Cure progress (%)", overlaying="y", side="right",
            range=[0, 115], gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", y=-0.25, font=dict(size=9)),
        plot_bgcolor="rgba(15,15,25,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
        margin=dict(t=40, b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pandemic Simulator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0d0d1a; color: #e0e0e0; }
    .stMetric label { font-size: 0.75rem; color: #aaa; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    div[data-testid="column"] { padding: 0 6px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🦠 Pandemic Simulator")
    st.caption("Stochastic SIR · 10-city network")
    st.divider()

    st.subheader("👥 Player roles")
    selected_roles: list[PlayerRole] = []
    medic_city = "Beijing"

    for role_name, meta in ROLE_META.items():
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.write(meta["emoji"])
        with col2:
            active = st.checkbox(role_name, key=f"role_{role_name}")
        if active:
            if meta["needs_city"]:
                medic_city = st.selectbox(
                    "Station Medic in:", CITY_NAMES, index=CITY_NAMES.index("Beijing"),
                    key="medic_city",
                )
                selected_roles.append(meta["cls"](medic_city))
            else:
                selected_roles.append(meta["cls"]())

    st.divider()
    st.subheader("⚙️ Simulation settings")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    anim_speed = st.slider("Animation speed (s/step)", 0.05, 1.0, 0.18, step=0.05)

    with st.expander("Advanced parameters"):
        spread_rate = st.slider("Base spread rate (β)", 0.10, 0.80, 0.45, step=0.01)
        recovery_rate = st.slider("Recovery rate (γ)", 0.01, 0.15, 0.04, step=0.01)
        travel_factor = st.slider("Travel spread factor", 0.01, 0.40, 0.18, step=0.01)
        cure_gain = st.slider("Cure gain / step", 0.005, 0.05, 0.013, step=0.001,
                              format="%.3f")

    params = InfectionParams(
        base_spread_rate=spread_rate,
        recovery_rate=recovery_rate,
        travel_spread_factor=travel_factor,
        cure_gain_per_step=cure_gain,
    )

    st.divider()
    run_btn = st.button("▶ Run simulation", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📊 Monte Carlo")
    mc_n = st.slider("Simulations per scenario", 50, 500, 150, step=50)
    mc_btn = st.button("Run role comparison", use_container_width=True)

# ── Session state initialisation ──────────────────────────────────────────────

for key, default in [
    ("frames", None),
    ("frame_idx", 0),
    ("playing", False),
    ("mc_stats", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Trigger: run simulation ───────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running simulation…"):
        st.session_state.frames    = run_and_capture(selected_roles, int(seed), params)
        st.session_state.frame_idx = 0
        st.session_state.playing   = True

# ── Trigger: Monte Carlo ──────────────────────────────────────────────────────

if mc_btn:
    with st.spinner(f"Running {mc_n} simulations per scenario…"):
        df = run_experiment(n=mc_n, verbose=False)
        st.session_state.mc_stats = analyse(df)

# ── Main panel ────────────────────────────────────────────────────────────────

if st.session_state.frames is None:
    st.markdown(
        """
        ## Welcome to the Pandemic Simulator 🦠

        Pick **player roles** in the sidebar, then hit **▶ Run simulation** to watch
        the infection cascade across the 10-city network in real time.

        ---
        **How it works:**
        - Each city runs a stochastic SIR model: susceptible → infected → recovered
        - Cities spread disease to neighbours via weighted travel routes
        - Player roles modify spread rates, cure speed, or can lock down cities
        - Win by reaching 100 % cure progress before the pandemic spirals

        ---
        *Try the Scientist role first — cure speed is the most critical lever.*
        """
    )
else:
    frames = st.session_state.frames
    total_frames = len(frames)

    # Auto-advance animation
    if st.session_state.playing:
        if st.session_state.frame_idx < total_frames - 1:
            time.sleep(anim_speed)
            st.session_state.frame_idx += 1
            st.rerun()
        else:
            st.session_state.playing = False

    # ── Controls bar ──────────────────────────────────────────────────────────
    ctrl_cols = st.columns([0.12, 0.12, 0.76])
    with ctrl_cols[0]:
        if st.button("⏮ Reset"):
            st.session_state.frame_idx = 0
            st.session_state.playing = False
            st.rerun()
    with ctrl_cols[1]:
        if st.session_state.playing:
            if st.button("⏸ Pause"):
                st.session_state.playing = False
                st.rerun()
        else:
            if st.button("▶ Play"):
                if st.session_state.frame_idx >= total_frames - 1:
                    st.session_state.frame_idx = 0
                st.session_state.playing = True
                st.rerun()

    frame_idx = st.slider(
        "Step", 0, total_frames - 1, st.session_state.frame_idx,
        key="frame_slider",
    )
    if frame_idx != st.session_state.frame_idx:
        st.session_state.frame_idx = frame_idx
        st.session_state.playing = False

    frame = frames[st.session_state.frame_idx]

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    last = frames[-1]
    m1.metric("Step", f"{frame['step']} / {total_frames - 1}")
    m2.metric("Cure progress", f"{frame['cure_progress']*100:.1f} %")
    m3.metric("Currently infected", f"{frame['total_infected']:,}")
    m4.metric("Outbreaks so far", frame["outbreaks"])
    outcome_color = "🟢" if last.get("outcome", "").startswith("WIN") else "🔴"
    m5.metric("Outcome", f"{outcome_color} {last.get('outcome', '…')}")

    # ── Network + most-infected table ─────────────────────────────────────────
    net_col, table_col = st.columns([0.70, 0.30])
    with net_col:
        st.plotly_chart(make_network_fig(frame), use_container_width=True, key="network")

    with table_col:
        st.markdown("**City breakdown**")
        rows = []
        for city in CITY_NAMES:
            c = frame["cities"][city]
            rows.append({
                "City": ("🔒 " if c["quarantined"] else "") + city,
                "Infected": c["infected"],
                "Rate %": f"{c['infection_rate']*100:.1f}",
            })
        df_cities = pd.DataFrame(rows).sort_values("Infected", ascending=False)
        st.dataframe(df_cities, hide_index=True, use_container_width=True, height=380)

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.plotly_chart(
        make_timeline_fig(frames, st.session_state.frame_idx),
        use_container_width=True,
        key="timeline",
    )

# ── Monte Carlo results panel ─────────────────────────────────────────────────

if st.session_state.mc_stats is not None:
    st.divider()
    st.subheader("📊 Role comparison (Monte Carlo results)")
    stats = st.session_state.mc_stats

    # Win-rate bar chart
    fig_mc = go.Figure(go.Bar(
        x=stats["win_rate_%"],
        y=stats["scenario"],
        orientation="h",
        marker=dict(
            color=stats["win_rate_%"],
            colorscale="RdYlGn",
            cmin=0, cmax=100,
        ),
        text=stats["win_rate_%"].apply(lambda v: f"{v:.1f} %"),
        textposition="outside",
    ))
    fig_mc.update_layout(
        title="Win rate by role configuration",
        xaxis=dict(title="Win rate (%)", range=[0, 115]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(15,15,25,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=380,
        margin=dict(l=200, r=60, t=50, b=40),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("Full stats table"):
        st.dataframe(stats, hide_index=True, use_container_width=True)
