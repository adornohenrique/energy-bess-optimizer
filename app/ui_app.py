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

from core.markets import (
    COUNTRY_CHOICES,
    fetch_entsoe_day_ahead_prices,
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
st.title("Calculadora BESS — em MW (day-ahead + dados 15 s)")

with st.expander("Como usar a calculadora"):
    st.markdown("""
1) Na seção **Fonte de preços**, escolha **API ENTSO-E (day-ahead)** ou **Arquivo CSV**.  
   - API: selecione o **país**, período, cole o **token** ENTSO-E e clique **Buscar preços**.  
   - CSV: envie um arquivo `datetime,price_EUR_per_MWh`.  
2) Envie (opcional) **PV** `datetime,pv_MW` e **Carga** `datetime,load_MW` — ambos em MW e 15 s.  
3) Ajuste **taxas (€/MWh)** e **limites de rede (MW)**.  
4) Defina **P_bess (MW)** e **C-rate (1/h)** **ou** use **Otimizar (P e C-rate)**.  
5) Veja **KPIs** (EBITDA, ROI, Payback, LCOE) e baixe o PDF.
""")

# ===========================
#  FONTE DE PREÇOS
# ===========================
st.header("Fonte de preços (spot day-ahead)")

price_source = st.radio("Escolha a fonte",
                        ["API ENTSO-E (day-ahead)", "Arquivo CSV"],
                        horizontal=True)

price_df = None

if price_source == "API ENTSO-E (day-ahead)":
    c1, c2, c3, c4 = st.columns([1.1,1,1,1])
    with c1:
        display = [c[0] for c in COUNTRY_CHOICES]
        code_by_name = {c[0]: c[1] for c in COUNTRY_CHOICES}
        country_name = st.selectbox("País / Zona", display, index=display.index("Portugal") if "Portugal" in display else 0)
        country_code = code_by_name[country_name]
    with c2:
        start_date = st.date_input("Início", value=pd.Timestamp.utcnow().date().replace(month=1, day=1))
    with c3:
        end_date   = st.date_input("Fim", value=pd.Timestamp.utcnow().date())
    with c4:
        entsoe_token = st.text_input("Token ENTSO-E", type="password", help="Crie em transparency.entsoe.eu → Settings → API Token")

    btn = st.button("🔎 Buscar preços")
    if btn:
        with st.spinner("Baixando preços day-ahead da ENTSO-E..."):
            try:
                price_df = fetch_entsoe_day_ahead_prices(
                    country_alpha2=country_code,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    token=entsoe_token,
                )
                st.success(f"Preços carregados: {len(price_df):,} amostras (15 s).")
            except Exception as e:
                st.error(f"Falha ao obter preços da ENTSO-E: {e}")

else:
    f_price = st.file_uploader("Preços (CSV, 15 s): `datetime,price_EUR_per_MWh`", type=["csv"])
    if f_price:
        tmp = pd.read_csv(f_price)
        # normaliza para 15 s e UTC
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
        tmp = tmp.sort_values("datetime")
        tmp = tmp.set_index("datetime").resample("15S").ffill().reset_index()
        price_df = tmp[["datetime","price_EUR_per_MWh"]]

if price_df is None:
    st.info("Carregue os **preços** pela API ou CSV para continuar.")
    st.stop()

# ===========================
#  PV / CARGA (opcionais)
# ===========================
st.header("Séries opcionais (MW, 15 s)")

c1, c2 = st.columns(2)
with c1:
    f_pv = st.file_uploader("PV (CSV, 15 s) — `datetime,pv_MW`", type=["csv"])
    pv_df = None
    if f_pv:
        pv_df = pd.read_csv(f_pv)
        pv_df["datetime"] = pd.to_datetime(pv_df["datetime"], utc=True)
        pv_df = pv_df.sort_values("datetime").set_index("datetime").resample("15S").ffill().reset_index()[["datetime","pv_MW"]]
with c2:
    f_load = st.file_uploader("Carga (CSV, 15 s) — `datetime,load_MW`", type=["csv"])
    load_df = None
    if f_load:
        load_df = pd.read_csv(f_load)
        load_df["datetime"] = pd.to_datetime(load_df["datetime"], utc=True)
        load_df = load_df.sort_values("datetime").set_index("datetime").resample("15S").ffill().reset_index()[["datetime","load_MW"]]

# ===========================
#  PREÇOS/REDE/CUSTOS
# ===========================
st.header("Parâmetros de preço, rede e custos")

colA, colB, colC = st.columns(3)
with colA:
    import_fee = st.number_input("Tarifa de importação (€/MWh)", 0.0, value=0.0, step=1.0,
                                 help="Taxa do fornecedor + rede ao importar")
    export_fee = st.number_input("Tarifa de exportação (€/MWh)", 0.0, value=0.0, step=1.0,
                                 help="Taxa de rede/market maker ao exportar")
    P_imp = st.number_input("Limite de importação (MW)", 0.0, value=200.0, step=5.0)
with colB:
    P_exp = st.number_input("Limite de exportação (MW)", 0.0, value=200.0, step=5.0)
    allow_grid = st.checkbox("Permitir carga pela rede", value=True)
    solver_time = st.number_input("Tempo máx. solver (s)", 10, value=120, step=10)
with colC:
    deg_cost = st.number_input("Degradação (€/MWh descarregado)", 0.0, value=2.0, step=0.5)
    opex_fix_bess = st.number_input("OPEX fixo BESS (€/ano)", 0.0, value=60_000.0, step=5_000.0)
    opex_fix_gen  = st.number_input("OPEX fixo usina (€/ano)", 0.0, value=0.0, step=10_000.0)

colD, colE, colF = st.columns(3)
with colD:
    opex_var_trade = st.number_input("OPEX var. mercado (€/MWh)", 0.0, value=0.5, step=0.1)
    opex_var_gen   = st.number_input("OPEX var. geração (€/MWh)", 0.0, value=0.0, step=0.1)
    capex_gen = st.number_input("CAPEX usina (EUR)", 0.0, value=0.0, step=100_000.0)
with colE:
    discount = st.number_input("Taxa de desconto (%)", 0.0, value=8.0, step=0.5)
    lifetime = st.number_input("Vida útil (anos)", 1, value=15, step=1)
    eta_c = st.number_input("Eficiência de carga (%)", 0.0, 100.0, 95.0, step=1.0)
with colF:
    eta_d = st.number_input("Eficiência de descarga (%)", 0.0, 100.0, 95.0, step=1.0)
    soc_min = st.number_input("SoC mínimo (%)", 0.0, 100.0, 0.0, step=5.0)
    soc_max = st.number_input("SoC máximo (%)", 0.0, 100.0, 100.0, step=5.0)

st.subheader("CAPEX BESS (€/kWh e €/kW)")
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

# ===========================
#  BASELINES
# ===========================
st.header("Baselines (referência)")
base = baselines_mw(price_df, pv_df, load_df, import_fee, export_fee)
b1, b2 = st.columns(2)
b1.metric("Custo consumo (sem PV/BESS)", euro(base["Cost_consumption_annual_EUR"], 0))
b2.metric("Receita só solar (exportando tudo)", euro(base["Revenue_solar_only_annual_EUR"], 0))

# ===========================
#  BESS — FIXO ou OTIMIZAR
# ===========================
st.header("BESS — definir P (MW) e C-rate (1/h)")
mode = st.radio("Modo", ["Rodar tamanho fixo", "Otimizar (P e C-rate)"], horizontal=True)

from core.optimization import run_site_bess_mw, optimize_site_bess_mw

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

# ===========================
#  RESULTADOS
# ===========================
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

# ===========================
#  PDF
# ===========================
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
