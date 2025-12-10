import os
import sys
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------- Formatação PT/BR ----------
def fmt_pt(x, decimals=0):
    if x is None:
        return "-"
    try:
        s = f"{float(x):,.{decimals}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def euro(x, decimals=0):
    return f"€ {fmt_pt(x, decimals)}"

# ---------- Resample ponderado por energia ----------
def resample_weighted_hourly(price_df, gen_df):
    """
    - Preço horário ponderado por energia: sum(price*gen) / sum(gen). Se sum(gen)==0, usa média simples de preço.
    - Geração: soma por hora (MWh/hora).
    """
    p = price_df.copy(); g = gen_df.copy()
    p["datetime"] = pd.to_datetime(p["datetime"]).dt.floor("T")
    g["datetime"] = pd.to_datetime(g["datetime"]).dt.floor("T")
    p = p.sort_values("datetime"); g = g.sort_values("datetime")

    # alinhar a grade temporal antes do resample (evita buracos)
    # (opcional: aqui assumimos que os CSVs já estão bem comportados)
    g_h = g.set_index("datetime").resample("1H").sum(numeric_only=True)
    # para ponderar, precisamos da geração por passo fino; então fazemos join com preços por minuto
    # e depois agregamos hora a hora
    pg = p.merge(g, on="datetime", how="left")
    pg["gen_MWh"] = pg["gen_MWh"].fillna(0.0)
    # agrega por hora ponderando
    w = pg.set_index("datetime").resample("1H").apply(
        lambda x: np.nan if x["gen_MWh"].sum() == 0 else (x["price_EUR_per_MWh"] @ x["gen_MWh"]) / x["gen_MWh"].sum()
    )
    # quando não há geração, cai na média simples da hora
    p_h_simple = p.set_index("datetime").resample("1H").mean(numeric_only=True)
    p_h = w.rename(columns={"price_EUR_per_MWh": "price_EUR_per_MWh"})
    p_h["price_EUR_per_MWh"].fillna(p_h_simple["price_EUR_per_MWh"], inplace=True)
    p_h = p_h[["price_EUR_per_MWh"]].reset_index()

    g_h = g_h.reset_index()
    g_h["gen_MWh"] = g_h["gen_MWh"].astype(float)
    return p_h, g_h

# ---------- Thin plot ----------
def _thin_df(df, max_points=2000):
    if df is None or len(df) == 0:
        return df
    if len(df) <= max_points:
        return df
    step = max(1, len(df)//max_points)
    return df.iloc[::step].copy()

# ---------- Projeto paths ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.optimization import (
    run_baseline,
    run_price_threshold_dispatch,
    run_with_bess,
    run_sensitivities,
    run_batch_scenarios,
    run_milp_rolling_year,
)

st.set_page_config(page_title="Energy + BESS Optimizer", layout="wide")

with st.expander("ℹ️ Ajuda & Legenda"):
    st.markdown("""
**Como usar**
1) Envie os CSVs:
   - **Preços**: `datetime,price_EUR_per_MWh`
   - **Geração**: `datetime,gen_MWh`
2) Escolha o **modo de execução**:
   - **Agregado 1h (ponderado por energia)** → rápido (±1–2% de erro nos KPIs).
   - **Rolling 15-min (semanas)** → alta precisão (±1–3% KPIs; E/P ±5–10%).
3) Ajuste parâmetros econômicos e limites (export/import) e **limites de E/P** do BESS.
4) Veja Baseline, Heurística (Preço>LCOE) e MILP; exporte PDF.

**Abreviações** – LCOE, EBITDA, BESS, E_cap, P_cap, Throughput (descarga), CRF.
""")

mode = st.radio("Modo", ["Agregado 1h (ponderado)", "Rolling 15-min (semanas)"])

# Uploads
st.header("Dados de entrada")
c1, c2 = st.columns(2)
with c1:
    price_file = st.file_uploader("Preços (CSV)", type=["csv"], key="price")
    price_df_raw = pd.read_csv(price_file) if price_file else None
with c2:
    gen_file = st.file_uploader("Geração (CSV)", type=["csv"], key="gen")
    gen_df_raw = pd.read_csv(gen_file) if gen_file else None

if price_df_raw is None or gen_df_raw is None:
    st.info("Envie **ambos** os CSVs para continuar.")
    st.stop()

# Parâmetros comuns
st.header("Parâmetros econômicos e operacionais")
colA, colB, colC = st.columns(3)
with colA:
    capex_gen = st.number_input("CAPEX usina (EUR)", min_value=0.0, value=10_000_000.0, step=100_000.0)
    lifetime_years = st.number_input("Vida útil (anos)", min_value=1, value=15, step=1)
    discount_rate = st.number_input("Taxa de desconto (%)", min_value=0.0, value=8.0, step=0.5)
    availability_pct = st.number_input("Disponibilidade (%)", min_value=0.0, max_value=100.0, value=97.0, step=0.5)
with colB:
    opex_fix_gen = st.number_input("OPEX fixo usina (EUR/ano)", min_value=0.0, value=200_000.0, step=10_000.0)
    opex_fix_bess = st.number_input("OPEX fixo BESS (EUR/ano)", min_value=0.0, value=60_000.0, step=5_000.0)
    opex_var_gen = st.number_input("OPEX var. geração (EUR/MWh)", min_value=0.0, value=1.0, step=0.1)
with colC:
    opex_var_trade = st.number_input("OPEX var. mercado (EUR/MWh)", min_value=0.0, value=0.5, step=0.1)
    P_grid_export_max = st.number_input("Export máx (MW)", min_value=0.0, value=100.0, step=1.0)
    P_grid_import_max = st.number_input("Import máx (MW)", min_value=0.0, value=100.0, step=1.0)

st.subheader("BESS – limites, eficiência e degradação")
colE, colF, colG = st.columns(3)
with colE:
    E_cap_max = st.number_input("E_cap máx (MWh)", min_value=0.0, value=200.0, step=10.0)
    P_cap_max = st.number_input("P_cap máx (MW)", min_value=0.0, value=100.0, step=5.0)
with colF:
    c_E_capex = st.number_input("CAPEX BESS (EUR/kWh)", min_value=0.0, value=250.0, step=10.0)
    c_P_capex = st.number_input("CAPEX BESS (EUR/kW)", min_value=0.0, value=150.0, step=10.0)
with colG:
    eta_charge = st.number_input("Eficiência carga (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
    eta_discharge = st.number_input("Eficiência descarga (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
    deg_cost = st.number_input("Degradação (€/MWh descarregado)", min_value=0.0, value=2.0, step=0.5)

allow_grid_charging = st.checkbox("Permitir carga pela rede (MILP)?", value=True)
solver_time_limit_s = st.number_input("Limite de tempo solver (s)", min_value=10, value=180, step=10)
cycles_max = st.number_input("Limite de ciclos/ano (0 = sem limite)", min_value=0, value=300, step=10)

params_common = {
    "capex_gen": capex_gen,
    "lifetime_years": lifetime_years,
    "discount_rate": discount_rate,
    "availability_pct": availability_pct,
    "opex_fix_gen": opex_fix_gen,
    "opex_fix_bess": opex_fix_bess,
    "opex_var_gen_eur_per_mwh": opex_var_gen,
    "opex_var_trade_eur_per_mwh": opex_var_trade,
    "P_grid_export_max": P_grid_export_max,
    "P_grid_import_max": P_grid_import_max,
    "allow_grid_charging": allow_grid_charging,
    "c_E_capex": c_E_capex,
    "c_P_capex": c_P_capex,
    "eta_charge": eta_charge / 100.0,
    "eta_discharge": eta_discharge / 100.0,
    "deg_cost_eur_per_MWh_throughput": deg_cost,
    "cycles_per_year_max": cycles_max,
    "solver_time_limit_s": solver_time_limit_s,
    "E_cap_max": E_cap_max,
    "P_cap_max": P_cap_max,
}

# ---------- Gráficos helpers ----------
def _make_compare_figure(ebitda_base, ebitda_bess):
    fig, ax = plt.subplots()
    ax.bar([0,1], [ebitda_base/1e6, ebitda_bess/1e6], tick_label=["Sem BESS", "Com BESS"], width=0.6)
    ax.set_title("EBITDA (M€/ano) – Base vs BESS")
    return fig

def _make_schedule_figure(df, title):
    dfp = _thin_df(df, 2000); fig, ax = plt.subplots()
    for col in ["gen_MWh","sold_from_gen_MWh","discharge_MWh","charge_from_ren_MWh","charge_from_grid_MWh","spill_MWh"]:
        if col in dfp.columns: ax.plot(dfp["datetime"], dfp[col], label=col)
    ax.set_title(title); ax.legend()
    return fig

# ---------- PDF ----------
def _build_pdf_single(base, bess, fig_compare, fig_sched, project="Energy+BESS", scenario="Cenário"):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); W,H = A4
    c.setFont("Helvetica-Bold", 16); c.drawString(2*cm, H-2.5*cm, f"Relatório – {project}")
    c.setFont("Helvetica", 12); c.drawString(2*cm, H-3.5*cm, f"{scenario} | {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
    c.line(2*cm, H-3.8*cm, W-2*cm, H-3.8*cm)

    y = H-5*cm; c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "KPIs principais"); y -= 0.6*cm
    c.setFont("Helvetica", 11)
    lines = [
        f"LCOE base: {euro(base['LCOE_base_EUR_per_MWh'],2)} /MWh",
        f"EBITDA base: {euro(base['EBITDA_annual_EUR'],0)}/ano",
        f"Receita base: {euro(base['Revenue_annual_EUR'],0)}/ano",
    ]
    if bess and bess.get("status_text","").startswith(("Solução","Rolling")):
        lines += [
            f"LCOE total entregue: {euro(bess['LCOE_total_entregue_EUR_per_MWh'],2)} /MWh",
            f"EBITDA projeto (com BESS): {euro(bess['EBITDA_project_annual_EUR'],0)}/ano",
            f"BESS: {fmt_pt(bess['E_cap_opt_MWh'],2)} MWh / {fmt_pt(bess['P_cap_opt_MW'],2)} MW",
        ]
    for L in lines: c.drawString(2.3*cm, y, L); y -= 0.5*cm

    if fig_compare:
        p = io.BytesIO(); fig_compare.savefig(p, format="png", bbox_inches="tight"); p.seek(0)
        c.drawImage(ImageReader(p), 2*cm, y-7*cm, width=W-4*cm, height=7*cm); c.showPage()

    if fig_sched:
        p2 = io.BytesIO(); fig_sched.savefig(p2, format="png", bbox_inches="tight"); p2.seek(0)
        c.drawImage(ImageReader(p2), 2*cm, H-2.5*cm-12*cm, width=W-4*cm, height=10*cm); c.showPage()

    c.save(); buf.seek(0); return buf

# =========================================================
# Execução por modo
# =========================================================
if mode == "Agregado 1h (ponderado)":
    # 1) agregação ponderada por energia
    price_df, gen_df = resample_weighted_hourly(price_df_raw, gen_df_raw)

    # 2) Baseline
    st.header("Baseline (sem BESS)")
    baseline = run_baseline(price_df, gen_df, params_common)
    c1,c2,c3 = st.columns(3)
    c1.metric("Energia anual (MWh/ano)", fmt_pt(baseline["E_annual_MWh"],0))
    c2.metric("EBITDA base (EUR/ano)", euro(baseline["EBITDA_annual_EUR"],0))
    c3.metric("LCOE base (EUR/MWh)", euro(baseline["LCOE_base_EUR_per_MWh"],2))
    st.dataframe(baseline["schedule_baseline"].head(200))

    # 3) Heurística preço>LCOE
    st.header("Heurística – Preço > LCOE (usa BESS, sem rede)")
    heur = run_price_threshold_dispatch(price_df, gen_df, baseline, params_common)
    c1,c2,c3 = st.columns(3)
    c1.metric("Energia (MWh/ano)", fmt_pt(heur["Energy_annual_MWh"],0))
    c2.metric("Receita (EUR/ano)", euro(heur["Revenue_annual_EUR"],0))
    c3.metric("Throughput (MWh/ano)", fmt_pt(heur["Throughput_annual_MWh"],0))
    st.dataframe(heur["schedule"].head(200))

    # 4) MILP ótimo
    st.header("MILP – Ótimo com BESS (agregado 1h)")
    p2 = dict(params_common); p2["baseline_E_annual_MWh"] = baseline["E_annual_MWh"]
    res = run_with_bess(price_df, gen_df, p2)

    if res.get("status_text","").startswith("Solução"):
        c1,c2,c3 = st.columns(3)
        c1.metric("E_cap ótima (MWh)", fmt_pt(res["E_cap_opt_MWh"],2))
        c2.metric("P_cap ótima (MW)", fmt_pt(res["P_cap_opt_MW"],2))
        c3.metric("Energia entregue (MWh/ano)", fmt_pt(res["Energy_annual_MWh"],0))

        c4,c5,c6 = st.columns(3)
        c4.metric("EBITDA projeto (EUR/ano)", euro(res["EBITDA_project_annual_EUR"],0))
        c5.metric("LCOE total entregue", euro(res["LCOE_total_entregue_EUR_per_MWh"],2))
        c6.metric("Throughput (MWh/ano)", fmt_pt(res["Throughput_annual_MWh"],0))

        fig_cmp = _make_compare_figure(baseline["EBITDA_annual_EUR"], res["EBITDA_project_annual_EUR"]); st.pyplot(fig_cmp); plt.close(fig_cmp)
        fig_sch = _make_schedule_figure(res["schedule"], "Schedule – Com BESS (MILP)"); st.pyplot(fig_sch); plt.close(fig_sch)
    else:
        st.warning(res.get("status_text","Falha no solver."))
        fig_cmp = fig_sch = None

    st.header("Exportar PDF")
    pdf = _build_pdf_single(baseline, res if res.get("status_text","").startswith("Solução") else None, fig_cmp, fig_sch,
                            project="Energy+BESS", scenario="Agregado 1h")
    st.download_button("⬇️ Baixar PDF", data=pdf, file_name="energy_bess_report.pdf", mime="application/pdf")

else:
    st.header("Rolling 15-min (semanas representativas + ano inteiro)")
    sizing_weeks = st.number_input("Semanas representativas para dimensionamento (k)", min_value=1, value=4, step=1)
    window_days = st.number_input("Tamanho da janela (dias)", min_value=3, value=7, step=1)
    step_days = st.number_input("Passo entre janelas (dias)", min_value=1, value=7, step=1)
    time_limit_s_sizing = st.number_input("Tempo solver (s) – sizing", min_value=30, value=180, step=10)
    time_limit_s_dispatch = st.number_input("Tempo solver (s) – despacho por janela", min_value=5, value=15, step=5)

    # Rolling usa os CSVs originais (15 min)
    res_roll = run_milp_rolling_year(
        price_df_raw, gen_df_raw, params_common,
        sizing_weeks=int(sizing_weeks),
        window_days=int(window_days),
        step_days=int(step_days),
        time_limit_s_sizing=int(time_limit_s_sizing),
        time_limit_s_dispatch=int(time_limit_s_dispatch),
    )

    if res_roll.get("status_text","").startswith("Rolling"):
        c1,c2,c3 = st.columns(3)
        c1.metric("E_cap ótima (MWh)", fmt_pt(res_roll["E_cap_opt_MWh"],2))
        c2.metric("P_cap ótima (MW)", fmt_pt(res_roll["P_cap_opt_MW"],2))
        c3.metric("Energia entregue (MWh/ano)", fmt_pt(res_roll["Energy_annual_MWh"],0))
        c4,c5,c6 = st.columns(3)
        c4.metric("EBITDA projeto (EUR/ano)", euro(res_roll["EBITDA_project_annual_EUR"],0))
        c5.metric("LCOE total entregue", euro(res_roll["LCOE_total_entregue_EUR_per_MWh"],2))
        c6.metric("Throughput (MWh/ano)", fmt_pt(res_roll["Throughput_annual_MWh"],0))

        fig_cmp = _make_compare_figure(0.0, res_roll["EBITDA_project_annual_EUR"])  # sem base aqui
        st.pyplot(fig_cmp); plt.close(fig_cmp)

        # Para não estourar memória, não plotamos toda a série; mostramos amostra:
        st.subheader("Schedule (amostra)")
        st.dataframe(_thin_df(res_roll["schedule"], 400).head(400))
        fig_sch = _make_schedule_figure(_thin_df(res_roll["schedule"], 2000), "Schedule – Rolling 15-min"); st.pyplot(fig_sch); plt.close(fig_sch)

        st.header("Exportar PDF (resumo)")
        pdf = _build_pdf_single(
            {"LCOE_base_EUR_per_MWh": np.nan, "EBITDA_annual_EUR": 0.0, "Revenue_annual_EUR": 0.0},
            res_roll, fig_cmp, fig_sch, project="Energy+BESS", scenario="Rolling 15-min"
        )
        st.download_button("⬇️ Baixar PDF", data=pdf, file_name="energy_bess_report_rolling.pdf", mime="application/pdf")
    else:
        st.warning(res_roll.get("status_text","Falha no rolling-horizon."))
