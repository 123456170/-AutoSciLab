# =====================================================================
#  AutoSciLab — Multi-Agent Autonomous Scientific Research Laboratory
# ---------------------------------------------------------------------
#  Single-file Streamlit app. No API keys. No external calls.
#  8 specialised agents debate a synthetic research problem through
#  structured messages, a shared blackboard and a full audit trail.
#  The 80-second live demo auto-plays on launch.
#
#  Run:  streamlit run app.py
# =====================================================================
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

st.set_page_config(page_title="AutoSciLab — Multi-Agent Research Lab",
                   page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] {
        background: rgba(148,163,184,.07);
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 12px; padding: 8px 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

REQ_TXT = "streamlit\nnumpy\npandas\nplotly\nscipy\n"

# =====================================================================
# Agents
# =====================================================================
AGENTS = {
    "SYS": {"name": "Lab Orchestrator", "short": "System", "emoji": "🧭",
            "color": "#64748B", "role": "Registers the problem, enforces the budget, keeps the clock."},
    "LIB": {"name": "Literature & Knowledge Agent", "short": "Literature", "emoji": "📚",
            "color": "#3B82F6", "role": "Scans the (synthetic) corpus and sets informative priors."},
    "HYP": {"name": "Hypothesis Agent", "short": "Hypothesis", "emoji": "💡",
            "color": "#EAB308", "role": "Proposes competing, falsifiable hypotheses."},
    "DES": {"name": "Experiment-Design Agent", "short": "Design", "emoji": "🧪",
            "color": "#10B981", "role": "Plans experiments that maximally discriminate hypotheses."},
    "SIM": {"name": "Simulation Agent", "short": "Simulation", "emoji": "🖥️",
            "color": "#8B5CF6", "role": "Generates noisy data from a sealed ground-truth world."},
    "STA": {"name": "Statistical-Analysis Agent", "short": "Statistics", "emoji": "📊",
            "color": "#F97316", "role": "Bayesian model comparison, goodness-of-fit, evidence ratios."},
    "CRI": {"name": "Critic Agent", "short": "Critic", "emoji": "🧐",
            "color": "#EF4444", "role": "Hunts confounders, rejects weak claims, demands orthogonal evidence."},
    "REP": {"name": "Reproducibility Agent", "short": "Repro", "emoji": "🔁",
            "color": "#14B8A6", "role": "Re-runs key experiments with fresh seeds before acceptance."},
    "RPT": {"name": "Research-Report Agent", "short": "Report", "emoji": "📝",
            "color": "#22C55E", "role": "Writes the final evidence-graded report."},
}
AGENT_ORDER = ["LIB", "HYP", "DES", "SIM", "STA", "CRI", "REP", "RPT"]

STATUS_COLORS = {
    "PROPOSED": "#94A3B8", "ACTIVE": "#EAB308", "ACCEPTED": "#16A34A",
    "REJECTED": "#DC2626", "NOT SUPPORTED": "#DC2626",
    "LEADING": "#F59E0B", "UNRESOLVED": "#F59E0B",
}


def status_color(s):
    if not s:
        return "#94A3B8"
    key = s.split()[0].upper()
    return STATUS_COLORS.get(key, "#94A3B8")


# =====================================================================
# Math helpers
# =====================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def ev_fmt(ev):
    return "∞" if math.isinf(ev) else f"{ev:.2f}"


# =====================================================================
# Synthetic research problems (sealed ground truth = H2 / H1)
# =====================================================================
def build_problems():
    P = {}

    # ---------------- Problem 1 : condensed-matter physics ------------
    def p_base(T):
        return 0.90 + 0.002 * (T - 300.0)

    def p_jump(T, tc=412.0, w=2.5):
        return 3.20 * sigmoid((T - tc) / w)

    def truth_polymer(expd, x):
        return p_base(x) + p_jump(x)

    def h1_poly(expd, x):
        a = np.exp(-850.0 / x)
        a0, a1 = np.exp(-850.0 / 300.0), np.exp(-850.0 / 450.0)
        return 0.80 + 3.60 * (a - a0) / (a1 - a0)

    def h2_poly(expd, x):
        return p_base(x) + p_jump(x)

    def h3_poly(expd, x):
        return p_base(x) + (p_jump(x) if expd.get("probe", "2probe") == "2probe" else 0.0)

    def h4_poly(expd, x):
        g = np.exp(-(2.0e5 / x) ** 0.25)
        g0 = np.exp(-(2.0e5 / 300.0) ** 0.25)
        g1 = np.exp(-(2.0e5 / 450.0) ** 0.25)
        return 0.80 + 3.60 * (g - g0) / (g1 - g0)

    P["polymer"] = {
        "short": "Polymer P-343 conductivity anomaly",
        "domain": "Condensed-matter physics (synthetic)",
        "question": "What mechanism causes the sharp conductivity rise in polymer P-343 above ~400 K?",
        "x_label": "Temperature T (K)", "y_label": "Conductivity σ (mS·cm⁻¹)",
        "x_unit": "K", "true_id": "H2",
        "truth_desc": "A genuine structural phase transition at T_c ≈ 412 K (the jump survives a 4-probe control).",
        "truth": truth_polymer, "corpus": 412,
        "priors": {"H1": 0.22, "H2": 0.30, "H3": 0.28, "H4": 0.20},
        "literature": [
            {"ref": "Vance et al., Synth. J. Mater. 2021",
             "finding": "Sharp conductivity onsets in conjugated polymers coincide with backbone-ordering transitions",
             "note": "plausibility for H2"},
            {"ref": "Okafor & Lindqvist, Comput. Mater. Sci. 2023",
             "finding": "Electrode polarization can mimic metal-like onsets in 2-probe sweeps",
             "note": "H3 must be treated as a confounder"},
            {"ref": "Deng et al., J. Polym. Simul. 2020",
             "finding": "Variable-range hopping in disordered films shows weak curvature over 300–450 K",
             "note": "H4 unlikely to mimic a sharp onset"},
        ],
        "hypotheses": [
            {"id": "H1", "name": "Impurity-band thermal activation", "color": "#F59E0B",
             "statement": "Shallow impurity states thermally activate carriers, producing a smooth Arrhenius rise.",
             "signature": "Smooth convex increase with no inflection; identical in 2- and 4-probe.",
             "predict": h1_poly},
            {"id": "H2", "name": "Structural phase transition", "color": "#22C55E",
             "statement": "A first-order backbone ordering transition at T_c ≈ 412 K opens a high-conductivity channel.",
             "signature": "Sharp sigmoid onset (width ≈ 2–3 K), independent of measurement method.",
             "predict": h2_poly},
            {"id": "H3", "name": "Electrode-polarization artifact", "color": "#EF4444",
             "statement": "The apparent rise is a contact artifact of 2-probe measurement, not bulk physics.",
             "signature": "Onset present in 2-probe but absent in 4-probe (Kelvin) configuration.",
             "predict": h3_poly},
            {"id": "H4", "name": "Variable-range hopping", "color": "#8B5CF6",
             "statement": "Conduction follows T^{-1/4} Mott hopping with a smooth crossover.",
             "signature": "Gentle T^{-1/4} curvature; no sharp onset anywhere.",
             "predict": h4_poly},
        ],
        "experiments": [
            {"id": "E1", "tag": "primary", "probe": "2probe", "lo": 300, "hi": 450, "n": 10, "sigma": 0.45,
             "name": "2-probe T-sweep 300–450 K (quick scan)",
             "purpose": "Detect and coarsely localize any anomaly",
             "tests": ["H1", "H2", "H3", "H4"]},
            {"id": "E2", "tag": "control", "probe": "4probe", "lo": 300, "hi": 450, "n": 14, "sigma": 0.34,
             "name": "4-probe Kelvin T-sweep (contact-artifact control)",
             "purpose": "Break the H2/H3 degeneracy",
             "tests": ["H2", "H3"]},
            {"id": "E3", "tag": "highres", "probe": "2probe", "lo": 395, "hi": 430, "n": 18, "sigma": 0.30,
             "name": "Fine 2-probe sweep across 395–430 K",
             "purpose": "Resolve onset sharpness and T_c",
             "tests": ["H1", "H2", "H4"]},
            {"id": "E4", "tag": "powered", "probe": "2probe", "lo": 300, "hi": 450, "n": 34, "sigma": 0.26,
             "name": "High-power 2-probe sweep (n↑, noise↓)",
             "purpose": "Maximize evidence if uncertainty persists",
             "tests": ["H1", "H2", "H3", "H4"]},
        ],
    }

    # ---------------- Problem 2 : synthetic oncology ------------------
    def emax(d):
        return 1.0 - 0.62 * d / (d + 7.0)

    def thr(d):
        return 1.0 - 0.62 * sigmoid((d - 18.0) / 6.0)

    def truth_bio(expd, x):
        assay = expd.get("assay", "imaging")
        if assay == "imaging":
            return emax(x)
        if assay == "atp":
            return emax(x)
        return np.ones_like(x)

    def h1_bio(expd, x):
        assay = expd.get("assay", "imaging")
        if assay == "vehicle":
            return np.ones_like(x)
        return emax(x)

    def h2_bio(expd, x):
        return emax(x)

    def h3_bio(expd, x):
        assay = expd.get("assay", "imaging")
        if assay == "imaging":
            return emax(x)
        return np.ones_like(x)

    def h4_bio(expd, x):
        assay = expd.get("assay", "imaging")
        if assay == "vehicle":
            return np.ones_like(x)
        return thr(x)

    P["spheroid"] = {
        "short": "Compound RZ-112 vs. glioblastoma spheroids",
        "domain": "Synthetic oncology",
        "question": "By which mechanism does compound RZ-112 reduce glioblastoma spheroid volume?",
        "x_label": "RZ-112 nominal dose (µM)", "y_label": "Spheroid volume (fold of baseline)",
        "x_unit": "µM", "true_id": "H1",
        "truth_desc": "True on-target Emax efficacy: volume and viability both fall hyperbolically; vehicle is inert.",
        "truth": truth_bio, "corpus": 388,
        "priors": {"H1": 0.34, "H2": 0.22, "H3": 0.20, "H4": 0.24},
        "literature": [
            {"ref": "Marsh et al., In Silico Oncol. 2022",
             "finding": "True kinase inhibitors show hyperbolic Emax dose-responses in spheroid panels",
             "note": "plausibility for H1"},
            {"ref": "Iqbal & Torres, Assay Interference Rev. 2021",
             "finding": "High DMSO fractions co-varying with dose can masquerade as compound efficacy",
             "note": "H2 must be ruled out with a vehicle arm"},
            {"ref": "Sun et al., Imaging Artifacts Q. 2023",
             "finding": "Compound autofluorescence can quench volumetric dyes without cell death",
             "note": "H3 testable via orthogonal viability assay"},
        ],
        "hypotheses": [
            {"id": "H1", "name": "On-target Emax efficacy", "color": "#22C55E",
             "statement": "RZ-112 engages its target; volume and viability fall with hyperbolic Emax kinetics (ED50 ≈ 7 µM).",
             "signature": "Hyperbolic dose-response in imaging AND viability; vehicle arm flat.",
             "predict": h1_bio},
            {"id": "H2", "name": "Vehicle (DMSO) toxicity", "color": "#EF4444",
             "statement": "The apparent effect is caused by DMSO carried with each dose, not the compound.",
             "signature": "Vehicle-only arm reproduces the full dose-response curve.",
             "predict": h2_bio},
            {"id": "H3", "name": "Optical measurement artifact", "color": "#F59E0B",
             "statement": "RZ-112 quenches the volumetric dye; cells are actually unharmed.",
             "signature": "Imaging shows shrinkage while ATP viability stays flat.",
             "predict": h3_bio},
            {"id": "H4", "name": "Delayed cytostatic threshold", "color": "#8B5CF6",
             "statement": "A slow cytostatic mechanism produces a thresholded, sigmoidal dose-response.",
             "signature": "Sigmoid onset near ~18 µM instead of a hyperbola.",
             "predict": h4_bio},
        ],
        "experiments": [
            {"id": "E1", "tag": "primary", "assay": "imaging", "lo": 0, "hi": 50, "n": 10, "sigma": 0.09,
             "name": "Volumetric imaging dose-response 0–50 µM",
             "purpose": "Establish the primary efficacy curve",
             "tests": ["H1", "H2", "H3", "H4"]},
            {"id": "E2", "tag": "control", "assay": "vehicle", "lo": 0, "hi": 50, "n": 14, "sigma": 0.06,
             "name": "Vehicle-only control arm (DMSO matched)",
             "purpose": "Rule out solvent toxicity (H2)",
             "tests": ["H1", "H2"]},
            {"id": "E3", "tag": "orthogonal", "assay": "atp", "lo": 0, "hi": 50, "n": 16, "sigma": 0.07,
             "name": "Orthogonal ATP-viability dose-response",
             "purpose": "Rule out optical artifact (H3)",
             "tests": ["H1", "H3"]},
            {"id": "E4", "tag": "powered", "assay": "imaging", "lo": 0, "hi": 50, "n": 34, "sigma": 0.055,
             "name": "Powered imaging dose-response (n↑, noise↓)",
             "purpose": "Maximize evidence if uncertainty persists",
             "tests": ["H1", "H2", "H3", "H4"]},
        ],
    }
    return P


PROBLEMS = build_problems()

# =====================================================================
# Pipeline state & helpers
# =====================================================================
@dataclass
class ResearchState:
    problem: dict
    seed: int
    noise_mult: float
    accept_th: float
    messages: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    datasets: list = field(default_factory=list)
    post_history: list = field(default_factory=list)
    status_timeline: list = field(default_factory=list)
    status_cycle: dict = field(default_factory=dict)
    critic_notes: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    repro: dict = None
    final: dict = None
    graph: dict = None
    report_md: str = ""
    beat: float = 0.0
    total_time: float = 0.0


def say(S, frm, to, kind, text, payload=None, beat=1.0):
    S.messages.append({"beat": S.beat, "t": 0.0, "frm": frm, "to": to,
                       "kind": kind, "text": text, "payload": payload or {}})
    S.beat += beat


def decide(S, cycle, agent, action, rationale):
    S.decisions.append({"beat": S.beat, "t": 0.0, "cycle": cycle,
                        "agent": agent, "action": action, "rationale": rationale})


def phase(S, name):
    S.phases.append({"name": name, "beat": S.beat, "t": 0.0})
    say(S, "SYS", "ALL", "PHASE", f"<b>{name}</b>", beat=1.4)


def analyze(datasets, hyps, priors):
    lls = {}
    for h in hyps:
        ll = 0.0
        for ds in datasets:
            mu = h["predict"](ds["exp"], ds["x"])
            z = (ds["y"] - mu) / ds["sigma"]
            ll += float(np.sum(-0.5 * z * z - np.log(ds["sigma"]) - 0.5 * np.log(2.0 * math.pi)))
        lls[h["id"]] = ll
    mx = max(lls.values())
    w = {k: priors[k] * math.exp(v - mx) for k, v in lls.items()}
    Z = sum(w.values())
    post = {k: v / Z for k, v in w.items()}
    ranked = sorted(post.items(), key=lambda kv: -kv[1])
    top = ranked[0][0]
    ev = ranked[0][1] / max(ranked[1][1], 1e-12) if len(ranked) > 1 else float("inf")
    htop = next(h for h in hyps if h["id"] == top)
    chi2, npts = 0.0, 0
    for ds in datasets:
        mu = htop["predict"](ds["exp"], ds["x"])
        chi2 += float(np.sum(((ds["y"] - mu) / ds["sigma"]) ** 2))
        npts += len(ds["x"])
    p_gof = float(sps.chi2.sf(chi2, max(npts - 1, 1)))
    return {"lls": lls, "post": post, "top": top, "ev": ev,
            "chi2": chi2, "p_gof": p_gof, "npts": npts}


def discrimination(expd, hyps):
    if not hyps:
        return 0.0
    xs = np.linspace(expd["lo"], expd["hi"], 140)
    P = np.vstack([h["predict"](expd, xs) for h in hyps])
    return float(np.ptp(P, axis=0).max())


def replicate(prob, expd, hyps, allow_ids, k, seed, noise_mult):
    rng = np.random.default_rng(seed)
    ok, effs = 0, []
    for _ in range(k):
        x = np.linspace(expd["lo"], expd["hi"], expd["n"])
        sg = expd["sigma"] * noise_mult
        y = prob["truth"](expd, x) + rng.normal(0.0, sg, x.size)
        lls = {}
        for h in hyps:
            z = (y - h["predict"](expd, x)) / sg
            lls[h["id"]] = float(np.sum(-0.5 * z * z))
        if max(lls, key=lls.get) in allow_ids:
            ok += 1
        effs.append(float(np.ptp(y)))
    return ok, k, effs


def powered_variant(base, idx):
    e = dict(base)
    e["id"] = f"{base['id']}P{idx}"
    e["name"] = f"Powered replication of {base['id']} · n×2.6 · noise↓"
    e["n"] = int(base["n"] * 2.6)
    e["sigma"] = round(base["sigma"] * 0.72, 3)
    e["tag"] = "powered"
    return e


def sim_note(prob, expd, x, y):
    dy = np.gradient(y, x)
    i = int(np.argmax(np.abs(dy)))
    return (f"Strongest change near x≈{x[i]:.0f} {prob['x_unit']} "
            f"(|Δy/Δx|≈{abs(dy[i]):.3f}); observed range {np.ptp(y):.2f}.")


# =====================================================================
# The full autonomous research pipeline
# =====================================================================
def run_pipeline(problem_id, seed, noise_mult=1.0, accept_th=0.90, duration=80.0):
    prob = build_problems()[problem_id]
    hyps = prob["hypotheses"]
    hids = [h["id"] for h in hyps]
    S = ResearchState(problem=prob, seed=seed, noise_mult=noise_mult, accept_th=accept_th)
    rng = np.random.default_rng(seed)

    # ---- 0 · intake --------------------------------------------------
    phase(S, "Phase 0 · Problem intake")
    say(S, "SYS", "ALL", "INTAKE",
        f"Task received: <b>{prob['question']}</b><br>Domain: {prob['domain']}. "
        f"Budget: ≤3 research cycles · ≤2 experiments per cycle. Shared blackboard initialised.",
        beat=1.6)
    decide(S, 0, "SYS", "Problem registered",
           f"Synthetic problem '{prob['short']}' loaded with sealed ground truth (hidden from all agents).")

    # ---- 1 · literature ----------------------------------------------
    phase(S, "Phase 1 · Literature & priors")
    for ref in prob["literature"]:
        say(S, "LIB", "ALL", "LITERATURE",
            f"📖 {ref['ref']}: {ref['finding']} <i>({ref['note']})</i>", beat=1.0)
    prior_txt = " · ".join(f"{h['id']} {prob['priors'][h['id']]:.0%}" for h in hyps)
    say(S, "LIB", "HYP", "PRIORS",
        f"Informative priors set from corpus of {prob['corpus']} synthetic records → {prior_txt}",
        payload=dict(prob["priors"]), beat=1.2)
    decide(S, 0, "LIB", "Priors assigned", f"Prior weights {prior_txt} derived from literature scan.")

    # ---- 2 · hypotheses ----------------------------------------------
    phase(S, "Phase 2 · Competing hypothesis generation")
    for h in hyps:
        say(S, "HYP", "DES", "HYPOTHESIS",
            f"<b>{h['id']} · {h['name']}</b> — {h['statement']}"
            f"<br><i>Falsifiable signature: {h['signature']}</i>", beat=1.15)
        decide(S, 0, "HYP", f"Proposed {h['id']}", h["statement"])
    status = {h: "PROPOSED" for h in hids}
    S.status_timeline.append({"beat": S.beat, "t": 0.0, "snapshot": dict(status)})

    # ---- 3 · iterative research cycles --------------------------------
    used, pooled = set(), []
    converged, requests, cycle = False, None, 0
    final_res, repro_summary = None, None
    primary0 = next(e for e in prob["experiments"] if e["tag"] == "primary")

    while cycle < 3 and not converged:
        cycle += 1
        phase(S, f"Phase 3 · Research cycle {cycle} — design → simulate → analyse → critique")

        # design
        if cycle == 1:
            plan = [e for e in prob["experiments"] if e["tag"] == "primary"]
            rationale = "Coarse discrimination first: maximum information per simulated data point."
        else:
            plan = requests if requests else [powered_variant(primary0, cycle)]
            rationale = "Critic-directed follow-up: orthogonal protocol to break the surviving degeneracy."
        exp_txt = " · ".join(f"{e['id']} {e['name']} (n={e['n']})" for e in plan)
        say(S, "DES", "SIM", "DESIGN",
            f"Cycle {cycle} plan → {exp_txt}<br><i>Rationale: {rationale}</i>",
            payload=[dict(e) for e in plan], beat=1.4)
        decide(S, cycle, "DES", f"Approved {len(plan)} experiment(s)", exp_txt)

        # simulate
        for e in plan:
            used.add(e["id"])
            x = np.linspace(e["lo"], e["hi"], e["n"])
            sg = e["sigma"] * noise_mult
            y = prob["truth"](e, x) + rng.normal(0.0, sg, x.size)
            ds = {"id": e["id"], "exp": e, "x": x, "y": y, "sigma": sg,
                  "cycle": cycle, "beat": S.beat, "t": 0.0}
            S.datasets.append(ds)
            pooled.append(ds)
            say(S, "SIM", "STA", "DATA",
                f"✅ <b>{e['id']}</b> complete — {e['name']} · n={e['n']} · σ={sg:.2f}. "
                f"{sim_note(prob, e, x, y)} Data written to blackboard.",
                payload={"n": int(e["n"]), "x_range": [float(x[0]), float(x[-1])],
                         "y_mean": float(np.mean(y)), "y_sd": float(np.std(y))}, beat=1.2)
            decide(S, cycle, "SIM", f"Executed {e['id']}",
                   f"Generated {e['n']} noisy observations under protocol '{e['name']}'.")

        # statistics
        res = analyze(pooled, hyps, prob["priors"])
        final_res = res
        post_txt = " · ".join(f"{h['id']} {res['post'][h['id']]:.1%}" for h in hyps)
        say(S, "STA", "CRI", "STATS",
            f"Pooled Bayesian model comparison (n={res['npts']} pts, {len(pooled)} dataset(s)):"
            f"<br><b>{post_txt}</b>"
            f"<br>Goodness-of-fit ({res['top']}): χ²={res['chi2']:.1f}, p={res['p_gof']:.2f} · "
            f"evidence ratio best/2nd = {ev_fmt(res['ev'])}",
            payload={"post": res["post"], "lls": res["lls"], "chi2": res["chi2"],
                     "p_gof": res["p_gof"], "ev": res["ev"]}, beat=1.7)
        decide(S, cycle, "STA", "Model comparison updated", f"Posteriors: {post_txt}.")
        S.post_history.append({"beat": S.beat, "t": 0.0, "cycle": cycle,
                               "post": dict(res["post"]), "top": res["top"],
                               "p_gof": res["p_gof"], "ev": res["ev"]})

        # critic
        top = res["top"]
        ranked = sorted(res["post"].items(), key=lambda kv: -kv[1])
        gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
        degenerate = gap < 0.12 and (math.isinf(res["ev"]) or res["ev"] < 3.0)
        rejects = [h for h, p in res["post"].items()
                   if p <= 0.03 and status.get(h) != "REJECTED"]
        for h in rejects:
            status[h] = "REJECTED"
            S.status_cycle[h] = cycle
            decide(S, cycle, "CRI", f"Rejected {h}",
                   f"Posterior {res['post'][h]:.2%} ≤ 3% under pooled evidence.")
        live = [h for h in hyps if status[h["id"]] != "REJECTED"]
        unused = [e for e in prob["experiments"] if e["id"] not in used]
        scored = sorted(unused, key=lambda e: discrimination(e, live), reverse=True)
        can_accept = (res["post"][top] >= accept_th) and (res["p_gof"] > 0.01) and (not degenerate)

        parts = []
        if rejects:
            parts.append("Rejecting " + ", ".join(
                f"<b>{h}</b> ({res['post'][h]:.1%})" for h in rejects) +
                " — posterior ≤ 3% under pooled evidence.")
        if degenerate and len(ranked) > 1:
            parts.append(f"⚠️ Degeneracy: <b>{ranked[0][0]}</b> vs <b>{ranked[1][0]}</b> are statistically "
                         f"indistinguishable under the current protocol (gap {gap:.1%}, EV {ev_fmt(res['ev'])}).")
        if can_accept:
            parts.append(f"<b>{top}</b> reaches the acceptance threshold ({res['post'][top]:.1%} ≥ "
                         f"{accept_th:.0%}) with adequate fit (p={res['p_gof']:.2f}). "
                         f"Recommending acceptance pending replication.")
        if (not can_accept) and scored:
            parts.append("Requesting orthogonal experiment(s): " + ", ".join(
                f"<b>{e['id']}</b> — {e['purpose']}" for e in scored[:2]) + ".")
        say(S, "CRI", "REP", "CRITIQUE",
            "<br>".join(parts) if parts else "No objections. Evidence trajectory consistent.", beat=1.8)
        decide(S, cycle, "CRI", "Verdict issued",
               ("DEGENERATE" if degenerate else ("ACCEPT-READY" if can_accept else "INSUFFICIENT-EVIDENCE"))
               + f" · top={top} @ {res['post'][top]:.1%} · rejects={rejects or '—'}")
        S.critic_notes.append(f"Cycle {cycle}: " + " ".join(
            p.replace("<b>", "").replace("</b>", "").replace("<br>", " ")
             .replace("<i>", "").replace("</i>", "") for p in parts))
        S.status_timeline.append({"beat": S.beat, "t": 0.0, "snapshot": dict(status)})

        # reproducibility
        decisive = max(plan, key=lambda e: discrimination(e, live)) if plan else primary0
        dec_ds = next((d for d in pooled if d["id"] == decisive["id"]), pooled[-1])
        allow = {ranked[0][0], ranked[1][0]} if degenerate else {top}
        ok, k, effs = replicate(prob, decisive, hyps, allow, 3,
                                seed * 1009 + cycle * 97, noise_mult)
        extra = ""
        if ok < k:
            ok, k, effs = replicate(prob, decisive, hyps, allow, k + 2,
                                    seed * 777 + cycle * 131, noise_mult)
            extra = " ⚠️ Initial pass failed → extended replication triggered."
        repro_pass = ok == k
        repro_summary = {"exp": decisive["id"], "cycle": cycle, "k": k, "ok": ok,
                         "eff_mean": float(np.mean(effs)), "eff_sd": float(np.std(effs))}
        S.repro = repro_summary
        say(S, "REP", "CRI", "REPLICATION",
            f"Replicated <b>{decisive['id']}</b> ×{k} with fresh seeds: best-model identity recovered in "
            f"<b>{ok}/{k}</b> runs · effect magnitude {np.mean(effs):.2f} ± {np.std(effs):.2f} "
            f"(original {np.ptp(dec_ds['y']):.2f}). {'✅ PASS' if repro_pass else '❌ FAIL'}.{extra}",
            beat=1.5)
        decide(S, cycle, "REP", "Replication verdict",
               f"{ok}/{k} replicates consistent for {decisive['id']} "
               f"({'PASS' if repro_pass else 'FAIL'}).")

        if can_accept and repro_pass:
            status[top] = "ACCEPTED"
            S.status_cycle[top] = cycle
            converged = True
            S.status_timeline.append({"beat": S.beat, "t": 0.0, "snapshot": dict(status)})
            decide(S, cycle, "CRI", f"Accepted {top}",
                   f"Posterior {res['post'][top]:.1%} · GoF p={res['p_gof']:.2f} · replication {ok}/{k}.")
            say(S, "SYS", "ALL", "STATUS",
                f"🎯 Convergence after cycle {cycle}: <b>{top}</b> accepted.", beat=1.2)
        else:
            requests = scored[:2] if scored else [powered_variant(primary0, cycle)]
            say(S, "SYS", "ALL", "STATUS",
                f"Cycle {cycle} closed without convergence (top posterior {res['post'][top]:.1%}). "
                f"Uncertainty high → next cycle armed.", beat=1.0)

    # ---- finalization -------------------------------------------------
    top = final_res["top"]
    if not converged:
        status[top] = "LEADING (inconclusive)"
        for h in hids:
            if (status[h] in ("PROPOSED", "ACTIVE") and final_res["post"][h] <= 0.05):
                status[h] = "NOT SUPPORTED"
        S.status_timeline.append({"beat": S.beat, "t": 0.0, "snapshot": dict(status)})
    npts = sum(len(d["x"]) for d in S.datasets)
    S.final = {"top": top, "post": final_res["post"], "p_gof": final_res["p_gof"],
               "chi2": final_res["chi2"], "ev": final_res["ev"], "cycles": cycle,
               "npts": npts, "converged": converged, "status": dict(status)}

    # ---- 4 · report -----------------------------------------------------
    phase(S, "Phase 4 · Synthesis & report")
    S.report_md = build_report(S)
    verdict_line = (f"{top} accepted ({final_res['post'][top]:.1%})" if converged
                    else f"inconclusive — leading candidate {top} ({final_res['post'][top]:.1%})")
    say(S, "RPT", "ALL", "REPORT",
        f"📄 Final report compiled — {cycle} cycle(s), {len(S.datasets)} experiments, {npts} data points. "
        f"Verdict: <b>{verdict_line}</b>. See the Report tab.", beat=2.0)
    decide(S, cycle, "RPT", "Report finalised", verdict_line)

    # ---- normalize beats → seconds -------------------------------------
    f = duration / S.beat if S.beat > 0 else 1.0
    for m in S.messages:
        m["t"] = m["beat"] * f
    for d in S.decisions:
        d["t"] = d["beat"] * f
    for p in S.phases:
        p["t"] = p["beat"] * f
    for d in S.datasets:
        d["t"] = d["beat"] * f
    for s in S.status_timeline:
        s["t"] = s["beat"] * f
    for p in S.post_history:
        p["t"] = p["beat"] * f
    S.total_time = duration
    build_graph(S)
    return S


# =====================================================================
# Report writer
# =====================================================================
def build_report(S):
    prob, fin, repro = S.problem, S.final, S.repro
    L = []
    a = L.append
    a(f"# 🔬 Autonomous Research Report — {prob['short']}")
    a("")
    a(f"*Generated by the AutoSciLab multi-agent pipeline · seed `{S.seed}` · "
      f"{time.strftime('%Y-%m-%d')} · {fin['cycles']} research cycle(s) · "
      f"{len(S.datasets)} experiments · {fin['npts']} data points.*")
    a("")
    a("## Abstract")
    if fin["converged"]:
        a(f"An eight-agent laboratory autonomously investigated: **{prob['question']}** "
          f"After {fin['cycles']} experimental cycle(s), hypothesis **{fin['top']}** is accepted with "
          f"posterior probability **{fin['post'][fin['top']]:.1%}** (goodness-of-fit p = {fin['p_gof']:.2f}, "
          f"evidence ratio vs. runner-up = {ev_fmt(fin['ev'])}), and the result was independently "
          f"replicated {repro['ok']}/{repro['k']} times with fresh random seeds.")
    else:
        a(f"An eight-agent laboratory autonomously investigated: **{prob['question']}** "
          f"The evidence budget was exhausted after {fin['cycles']} cycles without reaching the "
          f"{S.accept_th:.0%} acceptance threshold; leading candidate **{fin['top']}** stands at "
          f"{fin['post'][fin['top']]:.1%}.")
    a("")
    a("## 1 · Background & priors (Literature Agent)")
    for ref in prob["literature"]:
        a(f"- **{ref['ref']}** — {ref['finding']} _({ref['note']})_")
    a("")
    a("## 2 · Competing hypotheses")
    a("| ID | Hypothesis | Prior | Final posterior | Status |")
    a("|---|---|---|---|---|")
    for h in prob["hypotheses"]:
        a(f"| {h['id']} | {h['name']} | {prob['priors'][h['id']]:.0%} | "
          f"{fin['post'][h['id']]:.1%} | {fin['status'][h['id']]} |")
    a("")
    a("## 3 · Experimental campaign (Design + Simulation Agents)")
    a("| ID | Cycle | Protocol | n | σ | Purpose |")
    a("|---|---|---|---|---|---|")
    for ds in S.datasets:
        e = ds["exp"]
        a(f"| {e['id']} | {ds['cycle']} | {e['name']} | {e['n']} | {ds['sigma']:.2f} | {e['purpose']} |")
    a("")
    a("## 4 · Statistical evidence (Statistics Agent)")
    a("- Final posteriors: " + " · ".join(
        f"**{h['id']}** {fin['post'][h['id']]:.1%}" for h in prob["hypotheses"]))
    a(f"- Goodness-of-fit for {fin['top']}: χ² = {fin['chi2']:.1f}, p = {fin['p_gof']:.2f}")
    a(f"- Evidence ratio (best vs. second): {ev_fmt(fin['ev'])}")
    a("")
    a("## 5 · Critical review (Critic Agent)")
    for note in S.critic_notes:
        a(f"- {note}")
    a("")
    a("## 6 · Reproducibility (Reproducibility Agent)")
    if repro:
        a(f"- Protocol **{repro['exp']}** re-simulated ×{repro['k']} with fresh seeds: "
          f"**{repro['ok']}/{repro['k']}** consistent · effect magnitude "
          f"{repro['eff_mean']:.2f} ± {repro['eff_sd']:.2f}.")
    a("")
    a("## 7 · Verdict")
    acc = [h for h, s in fin["status"].items() if s == "ACCEPTED"]
    rej = [h for h, s in fin["status"].items() if s == "REJECTED"]
    if acc:
        a(f"- ✅ **Accepted:** {', '.join(acc)}")
    if rej:
        a(f"- ❌ **Rejected:** {', '.join(rej)}")
    for h, s in fin["status"].items():
        if s not in ("ACCEPTED", "REJECTED"):
            a(f"- ⚠️ **{h}:** {s}")
    a("")
    a("## 8 · Limitations")
    a("- All data are synthetic; citations are fictional; conclusions hold only inside this simulated world.")
    a("- The noise model is known to the statistician; model space is limited to the four proposed mechanisms.")
    a("- The sealed ground truth was never exposed to any agent during inference.")
    return "\n".join(L)


# =====================================================================
# Evidence graph
# =====================================================================
def build_graph(S):
    prob, fin = S.problem, S.final
    nodes, edges = [], []

    def tfirst(frm):
        for m in S.messages:
            if m["frm"] == frm:
                return m["t"]
        return S.total_time

    nodes.append({"id": "Q", "col": 0, "label": "❓ Question", "color": "#64748B",
                  "t": 0.0, "hover": prob["question"]})
    nodes.append({"id": "LIT", "col": 0, "label": "📚 Literature", "color": "#3B82F6",
                  "t": tfirst("LIB"), "hover": "Corpus scan → informative priors"})
    edges.append({"a": "Q", "b": "LIT", "t": tfirst("LIB")})
    for h in prob["hypotheses"]:
        nodes.append({"id": h["id"], "col": 1, "label": h["id"], "color": h["color"],
                      "t": tfirst("HYP"),
                      "hover": f"{h['name']} · prior {prob['priors'][h['id']]:.0%}"})
        edges.append({"a": "Q", "b": h["id"], "t": tfirst("HYP")})

    executed = {}
    for ds in S.datasets:
        e = ds["exp"]
        if e["id"] not in executed:
            executed[e["id"]] = (e, ds["t"], ds["cycle"])
    for eid, (e, t, cyc) in executed.items():
        nodes.append({"id": eid, "col": 2, "label": eid, "color": "#0EA5A4", "t": t,
                      "hover": f"{e['name']} · cycle {cyc} · n={e['n']}"})
        for h in prob["hypotheses"]:
            if h["id"] in e.get("tests", []):
                edges.append({"a": h["id"], "b": eid, "t": t})

    for ph in S.post_history:
        sid = f"S{ph['cycle']}"
        nodes.append({"id": sid, "col": 3, "label": f"📊 C{ph['cycle']}", "color": "#F97316",
                      "t": ph["t"], "hover": f"Cycle {ph['cycle']} pooled statistics · top={ph['top']}"})
        for ds in S.datasets:
            if ds["cycle"] == ph["cycle"]:
                edges.append({"a": ds["exp"]["id"], "b": sid, "t": ph["t"]})

    for h in prob["hypotheses"]:
        stt = fin["status"].get(h["id"], "—")
        cyc = S.status_cycle.get(h["id"])
        t_v = next((p["t"] for p in S.post_history if p["cycle"] == cyc),
                   S.total_time * 0.98) if cyc else S.total_time * 0.98
        nodes.append({"id": f"V{h['id']}", "col": 4,
                      "label": f"{h['id']}<br>{stt.split()[0]}",
                      "color": status_color(stt), "t": t_v,
                      "hover": f"{h['name']} → {stt} · posterior {fin['post'][h['id']]:.1%}"})
        src = f"S{cyc}" if cyc else f"S{fin['cycles']}"
        edges.append({"a": src, "b": f"V{h['id']}", "t": t_v})

    if S.repro:
        tr = max((m["t"] for m in S.messages if m["kind"] == "REPLICATION"),
                 default=S.total_time)
        nodes.append({"id": "REPRO", "col": 3, "label": "🔁 Repro", "color": "#8B5CF6",
                      "t": tr, "hover": f"{S.repro['ok']}/{S.repro['k']} replicates of {S.repro['exp']}"})
        edges.append({"a": f"S{S.repro['cycle']}", "b": "REPRO", "t": tr})
        edges.append({"a": "REPRO", "b": S.repro["exp"], "t": tr, "dashed": True})
        edges.append({"a": "REPRO", "b": f"V{fin['top']}", "t": tr})

    nodes.append({"id": "RPT", "col": 5, "label": "📄 Report", "color": "#22C55E",
                  "t": S.total_time, "hover": "Final evidence-graded report"})
    for h in prob["hypotheses"]:
        edges.append({"a": f"V{h['id']}", "b": "RPT", "t": S.total_time})
    S.graph = {"nodes": nodes, "edges": edges}


def fig_evidence(S, clock):
    g = S.graph
    cols = {}
    for n in g["nodes"]:
        cols.setdefault(n["col"], []).append(n)
    pos = {}
    for col, items in cols.items():
        nn = len(items)
        for i, n in enumerate(items):
            pos[n["id"]] = (col * 1.7, (nn - 1) / 2.0 - i)

    ns = [n for n in g["nodes"] if n["t"] <= clock]
    visible_ids = {n["id"] for n in ns}
    es = [e for e in g["edges"] if e["t"] <= clock
          and e["a"] in visible_ids and e["b"] in visible_ids]

    ymax = 1.0
    if ns:
        ymax = max(abs(pos[n["id"]][1]) for n in ns) + 1.0

    fig = go.Figure()
    solid = [e for e in es if not e.get("dashed")]
    dashed = [e for e in es if e.get("dashed")]
    for grp, sty in ((solid, {"dash": "solid", "color": "rgba(148,163,184,.55)"}),
                     (dashed, {"dash": "dash", "color": "rgba(139,92,246,.75)"})):
        xs, ys = [], []
        for e in grp:
            x0, y0 = pos[e["a"]]
            x1, y1 = pos[e["b"]]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        if xs:
            fig.add_scatter(x=xs, y=ys, mode="lines", hoverinfo="none",
                            line={"width": 1.5, **sty})
    if ns:
        fig.add_scatter(x=[pos[n["id"]][0] for n in ns],
                        y=[pos[n["id"]][1] for n in ns],
                        mode="markers", hoverinfo="text",
                        hovertext=[n["hover"] for n in ns],
                        marker={"size": 30,
                                "color": [n["color"] for n in ns],
                                "line": {"width": 1.5, "color": "rgba(255,255,255,.65)"}})
    fig.update_layout(
        height=520,
        showlegend=False,
        annotations=[{"x": pos[n["id"]][0],
                      "y": pos[n["id"]][1] - 0.36,
                      "text": n["label"],
                      "showarrow": False,
                      "font": {"size": 10.5}} for n in ns],
        xaxis={"visible": False, "range": [-0.9, 5 * 1.7 + 0.9]},
        yaxis={"visible": False, "range": [-ymax, ymax]},
        margin={"t": 20, "b": 20, "l": 10, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)")
    return fig


# =====================================================================
# Other figures
# =====================================================================
def fig_experiment(S, ds, snap):
    e, prob = ds["exp"], S.problem
    fig = go.Figure()
    fig.add_scatter(x=ds["x"], y=ds["y"], mode="markers", name="Simulated data",
                    marker={"color": "#E2E8F0", "size": 7, "opacity": .85,
                            "line": {"width": .5, "color": "#0F172A"}})
    xs = np.linspace(e["lo"], e["hi"], 240)
    for h in prob["hypotheses"]:
        rej = snap.get(h["id"]) == "REJECTED"
        fig.add_scatter(x=xs, y=h["predict"](e, xs), mode="lines", name=h["id"],
                        line={"color": h["color"], "width": 1.1 if rej else 2.4,
                              "dash": "dot" if rej else "solid"},
                        opacity=.35 if rej else .95)
    fig.update_layout(
        title={"text": f"{e['id']} · {e['name']}  —  cycle {ds['cycle']}, n={e['n']}, σ={ds['sigma']:.2f}",
               "font": {"size": 13}},
        height=330, xaxis_title=prob["x_label"], yaxis_title=prob["y_label"],
        legend={"orientation": "h", "y": -0.30},
        margin={"t": 46, "b": 8, "l": 10, "r": 10}, font={"size": 11})
    return fig


def fig_posteriors(S, clock):
    rows = []
    for ph in S.post_history:
        if ph["t"] <= clock:
            for hid, p in ph["post"].items():
                rows.append({"cycle": ph["cycle"], "Hypothesis": hid, "Posterior": p})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    cmap = {h["id"]: h["color"] for h in S.problem["hypotheses"]}
    fig = px.line(df, x="cycle", y="Posterior", color="Hypothesis", markers=True,
                  color_discrete_map=cmap)
    fig.add_hline(y=S.accept_th, line_dash="dash",
                  annotation_text=f"acceptance ≥ {S.accept_th:.0%}",
                  annotation_font_size=10)
    fig.update_layout(height=320, yaxis_tickformat=".0%",
                      xaxis={"dtick": 1, "title": "Research cycle"},
                      margin={"t": 30, "b": 8, "l": 10, "r": 10}, font={"size": 11})
    return fig


# =====================================================================
# UI helpers
# =====================================================================
def bubble_html(m):
    a = AGENTS.get(m["frm"], AGENTS["SYS"])
    txt = m["text"].replace("\n", "<br>")
    return (
        f"<div style='border:1px solid {a['color']}44;border-left:4px solid {a['color']};"
        f"border-radius:10px;padding:7px 12px;margin:6px 0;background:rgba(148,163,184,.05);'>"
        f"<div style='font-size:11.5px;opacity:.75;'>T+{m['t']:05.1f}s · {a['emoji']} "
        f"<b>{a['name']}</b> → {m['to']} · <span style='background:{a['color']}22;"
        f"padding:1px 7px;border-radius:6px;'>{m['kind']}</span></div>"
        f"<div style='margin-top:3px;font-size:13.5px;'>{txt}</div></div>")


def agent_strip(active):
    parts = []
    for aid in AGENT_ORDER:
        a = AGENTS[aid]
        act = aid == active
        bg = a["color"] if act else "rgba(148,163,184,.10)"
        fg = "#0B1220" if act else "#CBD5E1"
        nm = a["name"]
        sh = a["short"]
        parts.append(f"<span title='{nm}' style='display:inline-block;margin:2px 4px;"
                     f"padding:4px 10px;border-radius:999px;font-size:12.5px;background:{bg};"
                     f"color:{fg};border:1px solid {a['color']}66'>{a['emoji']} {sh}</span>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def status_at(S, clock):
    snap = None
    for s in S.status_timeline:
        if s["t"] <= clock:
            snap = s["snapshot"]
    return snap or {h["id"]: "PROPOSED" for h in S.problem["hypotheses"]}


def playback_tick(S):
    ss = st.session_state
    now = time.time()
    if ss.get("playing") and not ss.get("paused"):
        ss["clock"] = min(ss.get("clock", 0.0) +
                          (now - ss.get("last_tick", now)) * ss.get("speed_sel", 1),
                          S.total_time)
    ss["last_tick"] = now
    if ss.get("clock", 0.0) >= S.total_time - 1e-6:
        ss["playing"] = False
        ss["paused"] = False
    return ss.get("clock", 0.0)


# =====================================================================
# Widget callback (runs BEFORE the script re-executes, so it is the
# sanctioned way to change a widget-bound session_state key)
# =====================================================================
def randomize_seed():
    st.session_state["seed_in"] = random.randint(1, 99999)


# =====================================================================
# Sidebar
# =====================================================================
ss = st.session_state
if "seed_in" not in ss:
    ss["seed_in"] = 42

with st.sidebar:
    st.markdown("## 🔬 AutoSciLab")
    st.caption("Multi-agent autonomous research demo · no API keys · deterministic simulator · "
               "the **80-second live demo auto-plays** on load.")
    pid = st.selectbox("Synthetic research problem", list(PROBLEMS.keys()),
                       format_func=lambda k: PROBLEMS[k]["short"])
    seed_v = st.number_input("Random seed", min_value=1, max_value=99999, step=1, key="seed_in")
    noise = st.slider("Noise multiplier", 0.5, 2.0, 1.0, 0.05)
    th = st.slider("Acceptance threshold (posterior)", 0.800, 0.999, 0.900, 0.005)
    st.select_slider("Playback speed", options=[1, 2, 4, 8], value=1,
                     format_func=lambda v: f"{v}×", key="speed_sel")
    reveal = st.checkbox("🕵️ Reveal sealed ground truth (educator mode)")

    b_run = st.button("▶ Run / replay research cycle")
    cA, cB = st.columns(2)
    cA.button("🎲 New seed", on_click=randomize_seed)
    b_skip = cB.button("⏭ Skip")
    b_pause = st.button("⏸ Pause" if not ss.get("paused") else "▶ Resume")

    key = (pid, int(seed_v), round(noise, 2), round(th, 3))
    need = ("S" not in ss) or (ss.get("run_key") != key) or b_run
    if need:
        with st.spinner("Assembling agent team & sealing ground truth…"):
            S = run_pipeline(pid, int(seed_v), noise, th)
        ss["S"] = S
        ss["run_key"] = key
        ss["clock"] = 0.0
        ss["playing"] = True
        ss["paused"] = False
        ss["last_tick"] = time.time()
    S = ss["S"]
    if b_skip:
        ss["clock"] = S.total_time
        ss["playing"] = False
    if b_pause:
        ss["paused"] = not ss.get("paused")
        ss["last_tick"] = time.time()

    with st.expander("⬇️ Downloads & clipboard"):
        try:
            src = Path(__file__).resolve().read_text(encoding="utf-8")
        except Exception:
            src = "# Source unavailable in this runtime."
        st.download_button("📋 Copy/download app.py", data=src, file_name="app.py",
                           mime="text/plain")
        st.download_button("📋 Copy/download requirements.txt", data=REQ_TXT,
                           file_name="requirements.txt", mime="text/plain")
        transcript = json.dumps(
            {"problem": S.problem["short"], "seed": S.seed,
             "messages": S.messages, "decisions": S.decisions,
             "posteriors": S.post_history, "final": S.final,
             "reproducibility": S.repro},
            indent=1, default=str)
        st.download_button("📋 Download full transcript (.json)", data=transcript,
                           file_name="autosci_transcript.json", mime="application/json")
        st.code(REQ_TXT, language="text")

    with st.expander("ℹ️ Agent team & rules"):
        for aid in AGENT_ORDER:
            a = AGENTS[aid]
            st.markdown(f"{a['emoji']} **{a['name']}** — {a['role']}")
        st.caption("Rules: accept when posterior ≥ threshold + replication PASS; "
                   "reject when posterior ≤ 3%; repeat experiments while uncertain (≤3 cycles).")
    st.caption("⚠️ Synthetic science demo — data simulated, citations fictional.")

# =====================================================================
# Header
# =====================================================================
st.markdown("<h2 style='margin:0'>🔬 AutoSciLab "
            "<span style='font-size:14px;opacity:.65'>· Multi-Agent Autonomous Research Laboratory</span></h2>",
            unsafe_allow_html=True)
truth_chip = ("🔓 truth: " + S.problem["truth_desc"]) if reveal else "🔒 ground truth sealed from agents"
st.markdown(f"**{S.problem['question']}**  \n"
            f"<span style='opacity:.75'>{S.problem['domain']} · budget ≤3 cycles · {truth_chip}</span>",
            unsafe_allow_html=True)
if reveal:
    st.warning(f"Educator mode — the sealed mechanism is **{S.problem['true_id']}**: "
               f"{S.problem['truth_desc']} Agents never see this during inference.")

clock = playback_tick(S)
cur_phase = next((p["name"] for p in reversed(S.phases) if p["t"] <= clock), "Initialising…")
st.progress(min(clock / S.total_time, 1.0),
            text=f"🎬 {'LIVE' if ss.get('playing') else ('PAUSED' if ss.get('paused') else 'COMPLETE')} · "
                 f"T+{clock:05.1f}s / {S.total_time:.0f}s · {cur_phase}")

vis_msgs = [m for m in S.messages if m["t"] <= clock]
active = vis_msgs[-1]["frm"] if vis_msgs else None
agent_strip(active)

# ---- metrics ----------------------------------------------------------
vis_ds = [d for d in S.datasets if d["t"] <= clock]
vis_ph = [p for p in S.post_history if p["t"] <= clock]
snap = status_at(S, clock)
top_now = vis_ph[-1]["top"] if vis_ph else "—"
top_p = vis_ph[-1]["post"][top_now] if vis_ph else None
live_n = sum(1 for v in snap.values() if v not in ("REJECTED",))
repro_visible = any(m["kind"] == "REPLICATION" and m["t"] <= clock for m in S.messages)
mc = st.columns(6)
mc[0].metric("Research cycle", len(vis_ph))
mc[1].metric("Experiments", len(vis_ds))
mc[2].metric("Data points", sum(d["exp"]["n"] for d in vis_ds))
mc[3].metric("Hypotheses alive", f"{live_n}/{len(snap)}")
mc[4].metric("Leading hypothesis", top_now,
             delta=f"{top_p:.0%}" if top_p is not None else None)
mc[5].metric("Reproducibility",
             f"{S.repro['ok']}/{S.repro['k']}" if (repro_visible and S.repro) else "—")

# =====================================================================
# Tabs
# =====================================================================
tab_console, tab_state, tab_exp, tab_graph, tab_dec, tab_report = st.tabs(
    ["🔴 Live Console", "🧠 Research State", "📊 Experiments & Data",
     "🕸️ Evidence Graph", "🧾 Decision Log", "📄 Final Report"])

with tab_console:
    if not vis_msgs:
        st.caption("Waiting for the first agent message…")
    shown = vis_msgs[-42:]
    if len(vis_msgs) > len(shown):
        st.caption(f"… {len(vis_msgs) - len(shown)} earlier messages (export the transcript for the full log)")
    for m in shown:
        st.markdown(bubble_html(m), unsafe_allow_html=True)
        if m["kind"] == "STATS":
            st.json(m["payload"])

with tab_state:
    st.markdown("**Shared blackboard — hypothesis status & belief state** "
                "*(everything below is derived only from agent messages)*")
    cols = st.columns(len(S.problem["hypotheses"]))
    for c, h in zip(cols, S.problem["hypotheses"]):
        stt = snap.get(h["id"], "PROPOSED")
        p = vis_ph[-1]["post"][h["id"]] if vis_ph else S.problem["priors"][h["id"]]
        with c.container():
            st.markdown(
                f"<div style='border:1px solid {h['color']}55;border-left:5px solid {h['color']};"
                f"border-radius:10px;padding:10px;height:100%;'>"
                f"<b>{h['id']} · {h['name']}</b><br>"
                f"<span style='background:{status_color(stt)}22;color:{status_color(stt)};"
                f"padding:1px 8px;border-radius:999px;font-size:12px'>{stt}</span><br><br>"
                f"belief: <b>{p:.1%}</b></div>", unsafe_allow_html=True)
    f = fig_posteriors(S, clock)
    if f is not None:
        st.plotly_chart(f)
    else:
        st.caption("No statistical analysis yet.")

with tab_exp:
    if not vis_ds:
        st.caption("No experiments completed yet — the Simulation Agent is standing by.")
    for ds in vis_ds:
        st.plotly_chart(fig_experiment(S, ds, snap))

with tab_graph:
    st.caption("Question → hypotheses → experiments → statistics/replication → verdicts → report. "
               "Dashed purple edge = replication re-run.")
    st.plotly_chart(fig_evidence(S, clock))

with tab_dec:
    vis_dec = [d for d in S.decisions if d["t"] <= clock]
    if vis_dec:
        df = pd.DataFrame(vis_dec)[["t", "cycle", "agent", "action", "rationale"]]
        df["t"] = df["t"].map(lambda v: f"{v:05.1f}s")
        df.columns = ["time", "cycle", "agent", "decision", "rationale"]
        st.dataframe(df, height=440, hide_index=True)
    else:
        st.caption("No decisions recorded yet.")

with tab_report:
    report_done = any(m["kind"] == "REPORT" and m["t"] <= clock for m in S.messages)
    if report_done:
        st.download_button("⬇️ Download report (.md)", data=S.report_md,
                           file_name="autosci_report.md", mime="text/markdown")
        st.markdown(S.report_md)
    else:
        st.caption("The Research-Report Agent writes the final report after the last cycle completes. "
                   "Use ⏭ Skip to jump to the end.")

# =====================================================================
# Playback engine (auto-playing live demo)
# =====================================================================
if ss.get("playing") and clock < S.total_time:
    time.sleep(0.28)
    st.rerun()