# app/ui_app.py
import os, sys, io
from datetime import datetime

# --- robustez de import do pacote core ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core.optimization import (
    baselines_mw,
    run_site_bess_mw,
    optimize_site_bess_mw,
)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- formatação PT/BR ----------
def fmt_pt(x, d=0):
    try:
        s = f"{float(x):,.{d}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"

def euro(x, d=0): return f"€ {fmt_pt(x, d)}"

st.set_page_config(page_title="Calculadora BESS (MW, 15 s)", layout="wide")
st.title("Calculadora BESS — em MW (dados 15 s)")

with st.expander("ℹ️ Instruções rápidas"):
    st.markdown("""
**Arquivos CSV (mesmo período, amostragem 15 s):**  
- Preços: `datetime,price_EUR_per_MWh`  
- PV (opcional): `datetime,pv_MW`  
- Carga (opcional): `datetime,load_MW`

**Cenários suportados:** arbitragem pura; consumo+arbitragem; PV+arbitragem; PV+consumo+arbitragem.  
**BESS em MW:** define **P (MW)** e **C-rate (1/h)** → energia **E (MWh) = P/C**.
""")

# ---------------- uploads ----------------
c1, c2, c3 = st.columns(3)
with c1:
    f_price = st.file_uploader("Preços (CSV, 15 s)", type=["csv"])
    price_df = pd.read_csv(f_price) if f_price else None
with c2:
    f_pv = st.file_uploader("PV (CSV, 15 s) — opcional", type=["csv"])
    pv_df = pd.read_csv(f_pv) if f_pv else None
with c3:
    f_load = st.file_uploader("Carga (CSV, 15 s) — opcional", type=["csv"])
    load_df = pd.read_csv(f_load) if f_load else None

if price_df is None:
    st.info("Envie ao menos o CSV de **preços**.")
    st.stop()

# ------------- parâmetros -------------
st.header("Preços / Rede / Custos")
colA, colB, colC = st.columns(3)
with colA:
    import_fee = st.number_input("Tarifa de importação (€/MWh)", 0.0, value=0.0, step=1.0)
    export_fee = st.number_input("Tarifa de exportação (€/MWh)", 0.0, value=0.0, step=1.0)
    P_imp = st.number_input("Limite de importação (MW)", 0.0, value=200.0, step=5.0)
with colB:
    P_exp = st.number_input("Limite de exportação (MW)", 0.0, value=200.0, step=5.0)
    allow_grid = st.checkbox("Permitir carga pela rede", value=True)
    solver_time = st.number_input("Tempo máx. solver (s)", 10, value=120, step=10)
with colC:
    deg_cost = st.number_input("Degradação (€/MWh descarregado)", 0.0, value=2.0, step=0.5)
    opex_fix_bess = st.number_input("OPEX fixo BESS (€/ano)", 0.0, value=60_000.0, step=5_000.0)
    opex_fix_gen  = st.number_input("OPEX fixo usina (€/ano) — se houver", 0.0, value=0.0, step=10_000.0)

colD, colE, colF = st.columns(3)
with colD:
    opex_var_trade = st.number_input("OPEX var. mercado (€/MWh)", 0.0, value=0.5, step=0.1)
    opex_var_gen   = st.number_input("OPEX var. geração (€/MWh)", 0.0, value=0.0, step=0.1)
    capex_gen = st.number_input("CAPEX usina (EUR) — se houver", 0.0, value=0.0, step=100_000.0)
with colE:
    discount = st.number_input("Taxa de desconto (%)", 0.0, value=8.0, step=0.5)
    lifetime = st.number_input("Vida útil (anos)", 1, value=15, step=1)
    eta_c = st.number_input("Eficiência de carga (%)", 0.0, 100.0, 95.0, step=1.0)
with colF:
    eta_d = st.number_input("Eficiência de descarga (%)", 0.0, 100.0, 95.0, step=1.0)
    soc_min = st.number_input("SoC mínimo (%)", 0.0, 100.0, 0.0, step=5.0)
    soc_max = st.number_input("SoC máximo (%)", 0.0, 100.0, 100.0, step=5.0)

st.subheader("CAPEX BESS")
cE, cP = st.columns(2)
with cE:
    capex_E = st.number_input("CAPEX (€/kWh)", 0.0, value=250.0, step=10.0)
with cP:
    capex_P = st.number_input("CAPEX (€/kW)", 0.0, value=150.0, step=10.0)

params = {
    "import_fee_eur_per_MWh": import_fee,
    "export_fee_eur_per_MWh": export_fee,
    "P_grid_import_max": P_imp,
    "P_grid_export_max": P_exp,
    "allow_grid_charging": allow_grid,
    "deg_cost_eur_per_MWh_throughput": deg_cost,
    "opex_fix_bess": opex_fix_bess,
    "opex_fix_gen": opex_fix_gen,
    "opex_var_trade_eur_per_MWh": opex_var_trade,
    "opex_var_gen_eur_per_mwh": opex_var_gen,
    "discount_rate": discount,
    "lifetime_years": lifetime,
    "c_E_capex": capex_E,
    "c_P_capex": capex_P,
    "capex_gen": capex_gen,
    "eta_charge": eta_c / 100.0,
    "eta_discharge": eta_d / 100.0,
    "soc_min": soc_min / 100.0,
    "soc_max": soc_max / 100.0,
    "solver_time_limit_s": solver_time,
}

# ------------- baselines -------------
st.header("Baselines (referência)")
base = baselines_mw(price_df, pv_df, load_df, import_fee, export_fee)
b1, b2 = st.columns(2)
b1.metric("Custo consumo (sem PV/BESS)", euro(base["Cost_consumption_annual_EUR"], 0))
b2.metric("Receita só solar (exportando tudo)", euro(base["Revenue_solar_only_annual_EUR"], 0))

# ------------- BESS -------------
st.header("BESS — definir P (MW) e C-rate (1/h)")
mode = st.radio("Modo", ["Rodar tamanho fixo", "Otimizar (P e C-rate)"], horizontal=True)

if mode == "Rodar tamanho fixo":
    cc1, cc2 = st.columns(2)
    with cc1:
        P_bess = st.number_input("P_bess (MW)", 0.0, value=50.0, step=5.0)
    with cc2:
        C_rate = st.number_input("C-rate (1/h)", 0.05, value=0.5, step=0.05)
    res = run_site_bess_mw(price_df, pv_df, load_df, params, P_bess, C_rate, return_schedule=True)
else:
    st.markdown("**Varredura (grade pequena)**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        P_min = st.number_input("P_min (MW)", 0.0, value=20.0, step=5.0)
        P_max = st.number_input("P_max (MW)", 0.0, value=100.0, step=5.0)
    with cc2:
        C_min = st.number_input("C_min (1/h)", 0.05, value=0.33, step=0.01)
        C_max = st.number_input("C_max (1/h)", 0.05, value=1.0, step=0.01)
    with cc3:
        N = st.slider("Pontos por eixo", 2, 6, 3)
        objective = st.selectbox("Objetivo", ["ROI", "EBITDA"], index=0)
    P_vals = np.linspace(P_min, P_max, N)
    C_vals = np.linspace(C_min, C_max, N)
    res = optimize_site_bess_mw(price_df, pv_df, load_df, params, P_vals, C_vals, objective)

# ------------- resultados -------------
st.subheader("Resultados")
r1, r2, r3 = st.columns(3)
r1.metric("P_bess (MW)", fmt_pt(res["P_cap_MW"], 2))
r2.metric("E_bess (MWh) = P/C", fmt_pt(res["E_cap_MWh"], 2))
r3.metric("Throughput (MWh/ano)", fmt_pt(res["Throughput_annual_MWh"], 0))

r4, r5, r6 = st.columns(3)
r4.metric("Margem bruta (€/ano)", euro(res["Gross_margin_annual_EUR"], 0))
r5.metric("EBITDA (€/ano)", euro(res["EBITDA_project_annual_EUR"], 0))
r6.metric("ROI anual (%)", fmt_pt(res["ROI_annual_%"], 2))

r7, r8 = st.columns(2)
r7.metric("Payback (anos)", fmt_pt(res["Payback_years"], 2))
r8.metric("LCOE total (€/MWh)", fmt_pt(res["LCOE_total_EUR_per_MWh"], 2))

st.markdown("**Componentes**")
k1, k2, k3 = st.columns(3)
k1.metric("Receita export (€/ano)", euro(res["Revenue_export_annual_EUR"], 0))
k2.metric("Economia autoconsumo (€/ano)", euro(res["Savings_selfcons_annual_EUR"], 0))
k3.metric("Custo carga da rede (€/ano)", euro(res["Cost_charge_grid_annual_EUR"], 0))

if "schedule" in res:
    st.subheader("Schedule (amostra)")
    dfp = res["schedule"].iloc[::max(1, len(res["schedule"]) // 2500)]
    st.dataframe(dfp.head(400))
    fig, ax = plt.subplots()
    for col in ["pv_MWh","load_MWh","c_grid_MWh","c_pv_MWh","d_grid_MWh","d_load_MWh","pv_load_MWh","pv_exp_MWh","g_load_MWh"]:
        if col in dfp.columns:
            ax.plot(dfp["datetime"], dfp[col], label=col, linewidth=0.9)
    ax.legend(ncol=3); ax.set_title("Fluxos (MWh por passo)"); ax.grid(True, alpha=0.2)
    st.pyplot(fig)

# ------------- PDF -------------
st.header("Exportar PDF (resumo)")
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

def build_pdf(base, res):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); W,H=A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H-2.4*cm, "Relatório — Calculadora BESS (MW, 15 s)")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H-3.1*cm, f"Gerado em {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
    c.line(2*cm, H-3.3*cm, W-2*cm, H-3.3*cm)

    y = H-4.3*cm; c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Baselines"); y -= 0.55*cm
    c.setFont("Helvetica", 11)
    c.drawString(2.2*cm, y, f"• Custo consumo: {euro(base['Cost_consumption_annual_EUR'],0)}"); y -= 0.45*cm
    c.drawString(2.2*cm, y, f"• Receita só solar: {euro(base['Revenue_solar_only_annual_EUR'],0)}"); y -= 0.8*cm

    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "BESS (resultado)"); y -= 0.55*cm
    c.setFont("Helvetica", 11)
    c.drawString(2.2*cm, y, f"P_bess: {fmt_pt(res['P_cap_MW'],2)} MW  |  E_bess: {fmt_pt(res['E_cap_MWh'],2)} MWh"); y -= 0.45*cm
    c.drawString(2.2*cm, y, f"Margem bruta: {euro(res['Gross_margin_annual_EUR'],0)}  |  EBITDA: {euro(res['EBITDA_project_annual_EUR'],0)}"); y -= 0.45*cm
    c.drawString(2.2*cm, y, f"ROI anual: {fmt_pt(res['ROI_annual_%'],2)} %  |  Payback: {fmt_pt(res['Payback_years'],2)} anos"); y -= 0.45*cm
    c.drawString(2.2*cm, y, f"Receita export: {euro(res['Revenue_export_annual_EUR'],0)}  |  Economia autoconsumo: {euro(res['Savings_selfcons_annual_EUR'],0)}"); y -= 0.45*cm
    c.drawString(2.2*cm, y, f"Custo carga rede: {euro(res['Cost_charge_grid_annual_EUR'],0)}"); y -= 0.8*cm
    c.drawString(2*cm, y, f"LCOE total: {fmt_pt(res['LCOE_total_EUR_per_MWh'],2)} €/MWh  |  Throughput: {fmt_pt(res['Throughput_annual_MWh'],0)} MWh/ano")

    c.showPage(); c.save(); buf.seek(0); return buf

pdf = build_pdf(base, res)
st.download_button("⬇️ Baixar PDF", data=pdf, file_name="bess_mw_report.pdf", mime="application/pdf")
