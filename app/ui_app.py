# app/ui_app.py
import os, sys, io
from datetime import datetime

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.optimization import run_site_bess_mw, baselines_mw
from core.markets import (
    COUNTRY_CHOICES, get_country_defaults,
    fetch_entsoe_day_ahead_prices, entsoe_available,
)

# -------------------- formatação --------------------
def fmt_pt(x, d=0):
    try:
        s = f"{float(x):,.{d}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"

def euro(x, d=0): return f"€ {fmt_pt(x, d)}"


st.set_page_config(page_title="BESS – modo simples (MW)", layout="wide")
st.title("BESS – modo simples (MW) • Sem upload de arquivos")

with st.expander("Como funciona (30s)"):
    st.markdown("""
- **Preços**: buscados na API **ENTSO-E** (precisa de token).  
- **PV**: gerado automaticamente a partir de **MWp** e **yield anual** (kWh/kWp·ano).  
- **Nada de uploads** — só parâmetros.  
- **Saídas**: Receita anual, EBITDA, ROI, payback, throughput, LCOE etc.
""")

# ===========================
#  PREÇOS – ENTSO-E OU SINTÉTICO (SEM UPLOAD)
# ===========================
st.header("Preços spot (ENTSO-E day-ahead)")

from core.markets import entsoe_available

price_df = None
country_name = None
country_code = None

def make_utc_15s_range(d0, d1):
    idx = pd.date_range(
        start=pd.Timestamp(d0).tz_localize("UTC"),
        end=pd.Timestamp(d1).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=15),
        freq="15S",
        inclusive="both",
    )
    return idx

def synth_prices(index_utc: pd.DatetimeIndex,
                 base_eur_mwh: float = 65.0,
                 diurnal_amplitude: float = 25.0,
                 volatility: float = 0.30,
                 neg_share: float = 0.25,
                 lower: float = -150.0,
                 upper: float = 350.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(index_utc, utc=True))
    h = idx.hour.values + idx.minute.values/60 + idx.second.values/3600
    phi = np.pi * (h - 19.0) / 6.0  # pico de preço ~19h UTC
    shape = np.cos(phi)
    shape = (shape + 1.0) / 2.0    # 0..1
    # sazonal simplificada por mês
    month_vec = np.array([0.9,0.95,1.0,0.95,0.9,0.85,0.85,0.9,0.95,1.0,1.05,1.0])
    m = idx.month.values
    season = month_vec[m-1]
    core = base_eur_mwh*season + diurnal_amplitude*shape
    noise = np.random.normal(0.0, volatility*base_eur_mwh, size=len(idx))
    p = core + noise
    # força fração de negativos (ordena e empurra cauda para baixo)
    k = int(len(p)*neg_share)
    if k > 0:
        order = np.argsort(p)
        tail = order[:k]
        p[tail] = np.linspace(lower, np.percentile(p, 5), k)
    # clip geral
    p = np.clip(p, lower, upper)
    return pd.DataFrame({"datetime": idx, "price_EUR_per_MWh": p})

# UI
names = [c[0] for c in COUNTRY_CHOICES]
code_by_name = {c[0]: c[1] for c in COUNTRY_CHOICES}
c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    country_name = st.selectbox("País/Zona", names, index=names.index("Portugal") if "Portugal" in names else 0)
    country_code = code_by_name[country_name]
with c2:
    start_date = st.date_input("Início", value=pd.Timestamp.utcnow().date().replace(month=1, day=1))
with c3:
    end_date   = st.date_input("Fim", value=pd.Timestamp.utcnow().date())

if entsoe_available():
    t = st.text_input("Token ENTSO-E", type="password")
    if st.button("🔎 Buscar preços no ENTSO-E"):
        with st.spinner("Baixando preços day-ahead..."):
            try:
                price_df = fetch_entsoe_day_ahead_prices(country_code, str(start_date), str(end_date), t)
                st.success(f"OK! {len(price_df):,} amostras (15 s).")
            except Exception as e:
                st.error(f"Falha ENTSO-E: {e}")
else:
    st.warning("ENTSO-E indisponível neste ambiente. Use **Preços sintéticos** abaixo.")
    with st.expander("Preços sintéticos (sem upload) – configurar"):
        b1, b2, b3 = st.columns(3)
        with b1:
            base = st.number_input("Preço base (€/MWh)", 0.0, value=65.0, step=1.0)
            amp  = st.number_input("Amplitude diurna (€/MWh)", 0.0, value=25.0, step=1.0)
        with b2:
            vol  = st.number_input("Volatilidade (fração do base)", 0.0, 2.0, 0.30, 0.05)
            neg  = st.number_input("Share de preços negativos (%)", 0.0, 100.0, 25.0, 1.0)
        with b3:
            low  = st.number_input("Mínimo (€/MWh)", -2000.0, value=-150.0, step=25.0)
            high = st.number_input("Máximo (€/MWh)", 0.0, value=350.0, step=25.0)

    if st.button("⚡ Gerar preços sintéticos"):
        idx = make_utc_15s_range(start_date, end_date)
        np.random.seed(42)  # reprodutível
        price_df = synth_prices(idx, base_eur_mwh=base, diurnal_amplitude=amp,
                                volatility=vol, neg_share=neg/100.0,
                                lower=low, upper=high)
        st.success(f"Gerados {len(price_df):,} pontos de preço (15 s).")

if price_df is None:
    st.info("Carregue ou gere os preços para continuar.")
    st.stop()

# =====================================================
# 2) PV SINTÉTICO (sem upload)
# =====================================================
st.header("Usina fotovoltaica (sem upload)")

# Sugestões rápidas de yield (kWh/kWp·ano)
YIELD_HINTS = {
    "Portugal": 1750, "Spain": 1800, "France": 1350, "Germany (DE-LU)": 1150,
    "Italy": 1500, "Netherlands": 1100, "Belgium": 1100, "Poland": 1050,
    "Greece": 1700, "Finland": 950, "Sweden": 1000, "Norway": 900,
}

hint = YIELD_HINTS.get(country_name, 1300)

c1, c2 = st.columns(2)
with c1:
    mwp = st.number_input("Capacidade instalada (MWp)", 0.0, value=100.0, step=5.0)
with c2:
    yield_kwh_per_kwp = st.number_input("Yield anual (kWh/kWp·ano)", 200.0, value=float(hint), step=50.0)

st.caption("Dica: PT ≈ 1 650–1 800 • ES ≈ 1 700–1 900 • DE ≈ 1 050–1 200 • IT ≈ 1 400–1 600")

def pv_synth(index_utc: pd.DatetimeIndex, mwp: float, yield_kwh_per_kwp: float) -> pd.DataFrame:
    """
    Perfil sintético em 15 s:
    - forma diária suave (pico ao meio-dia UTC),
    - fator sazonal por mês,
    - escalado para bater energia anual alvo: MWp * yield.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index_utc, utc=True))
    dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0
    T = len(idx)

    # Fatores mensais (genéricos Europa), normalizados depois
    month_vec = np.array([0.35,0.50,0.70,0.90,1.00,1.05,1.05,1.00,0.80,0.60,0.45,0.35])
    m = idx.month.values
    f_month = month_vec[m-1]

    # Forma diária: 0 fora ~6–18 UTC, sino no meio-dia
    h = idx.hour.values + idx.minute.values/60 + idx.second.values/3600
    phi = np.pi * (h - 12.0) / 6.0  # 12h=0 ; ±6h = ±π
    f_day = np.cos(phi)
    f_day = np.clip(f_day, 0.0, None) ** 1.5  # recorta e suaviza

    base = f_month * f_day
    if base.max() <= 0:
        pv_mw = np.zeros(T)
    else:
        base = base / base.max()
        pv_mw = mwp * base

    # Ajuste de energia anual
    energy_goal_MWh = mwp * yield_kwh_per_kwp  # 1 MWp ~ 1000 kWp → MWh = MWp * kWh/kWp
    energy_est_MWh = (pv_mw * dt_h).sum()
    if energy_est_MWh > 1e-6:
        scale = energy_goal_MWh / energy_est_MWh
        pv_mw = np.minimum(mwp, pv_mw * scale)  # limita por MWp

    return pd.DataFrame({"datetime": idx, "pv_MW": pv_mw})

pv_df = pv_synth(price_df["datetime"], mwp, yield_kwh_per_kwp)

# Plot rápido
with st.expander("Visualizar PV sintético (amostra)"):
    show = pv_df.iloc[::max(1, len(pv_df)//2000)]
    fig, ax = plt.subplots()
    ax.plot(show["datetime"], show["pv_MW"], lw=0.8)
    ax.set_title("PV (MW) – amostra")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

# =====================================================
# 3) BESS (só MWh + C-rate)
# =====================================================
st.header("BESS (parâmetros essenciais)")

cB1, cB2, cB3 = st.columns(3)
with cB1:
    e_bess_mwh = st.number_input("Capacidade do BESS (MWh)", 0.0, value=200.0, step=20.0)
with cB2:
    c_rate = st.number_input("C-rate (1/h)", 0.05, value=0.5, step=0.05)
with cB3:
    round_trip_eff = st.number_input("Eficiência round-trip (%)", 1.0, 100.0, 90.0, 1.0)

# dividir round-trip em carga/descarga simétricas
eta_charge = (round_trip_eff/100.0) ** 0.5
eta_discharge = eta_charge

st.subheader("SoC")
s1, s2, s3 = st.columns(3)
with s1:
    soc_min = st.number_input("SoC mínimo (%)", 0.0, 100.0, 5.0, 1.0)
with s2:
    soc_init = st.number_input("SoC inicial (%)", 0.0, 100.0, 50.0, 1.0)
with s3:
    enforce_equal_terminal = st.checkbox("Forçar SoC final = inicial", value=True)

allow_grid = st.checkbox("Permitir carga pela rede (grid charging)", value=True)

# Potência nominal pela combinação E + C
p_bess_mw = e_bess_mwh * c_rate

# =====================================================
# 4) REDE – tarifas e limites (constantes)
# =====================================================
st.header("Rede (tarifas simples e limites)")

preset = get_country_defaults(code_by_name[country_name])

g1, g2, g3, g4 = st.columns(4)
with g1:
    imp_fee = st.number_input("Tarifa de importação (€/MWh)", 0.0, value=float(preset["import_fee"]), step=0.5)
with g2:
    exp_fee = st.number_input("Tarifa de exportação (€/MWh)", 0.0, value=float(preset["export_fee"]), step=0.5)
with g3:
    p_imp_max = st.number_input("Limite de importação (MW)", 0.0, value=float(preset["P_imp"]), step=10.0)
with g4:
    p_exp_max = st.number_input("Limite de exportação (MW)", 0.0, value=float(preset["P_exp"]), step=10.0)

# =====================================================
# 5) CAPEX & OPEX (defaults bons; editar se quiser)
# =====================================================
with st.expander("CAPEX/OPEX (opcional – defaults realistas)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        capex_gen = st.number_input("CAPEX da usina (EUR)", 0.0, value=0.0, step=100_000.0)
        capex_E = st.number_input("CAPEX BESS (€/kWh)", 0.0, value=250.0, step=10.0)
    with c2:
        capex_P = st.number_input("CAPEX BESS (€/kW)", 0.0, value=150.0, step=10.0)
        opex_fix_bess = st.number_input("OPEX fixo BESS (€/ano)", 0.0, value=60_000.0, step=5_000.0)
    with c3:
        opex_fix_gen = st.number_input("OPEX fixo usina (€/ano)", 0.0, value=0.0, step=10_000.0)
        deg_cost = st.number_input("Degradação (€/MWh throughput)", 0.0, value=2.0, step=0.5)

    d1, d2, d3 = st.columns(3)
    with d1:
        discount = st.number_input("Taxa de desconto (%)", 0.0, value=8.0, step=0.5)
    with d2:
        lifetime = st.number_input("Vida útil (anos)", 1, value=15, step=1)
    with d3:
        solver_time = st.number_input("Tempo máx. solver (s)", 10, value=120, step=10)

# =====================================================
# 6) Monta parâmetros e roda
# =====================================================
params = {
    "P_grid_import_max": p_imp_max,
    "P_grid_export_max": p_exp_max,
    "allow_grid_charging": allow_grid,
    "deg_cost_eur_per_MWh_throughput": deg_cost,
    "opex_fix_bess": opex_fix_bess,
    "opex_fix_gen": opex_fix_gen,
    "opex_var_trade_eur_per_MWh": 0.5,   # leve custo variável de trading
    "opex_var_gen_eur_per_mwh": 0.0,
    "discount_rate": discount,
    "lifetime_years": lifetime,
    "c_E_capex": capex_E,
    "c_P_capex": capex_P,
    "capex_gen": capex_gen,
    "eta_charge": eta_charge,
    "eta_discharge": eta_discharge,
    "eta_ac2dc": 0.985,
    "eta_dc2ac": 0.985,
    "soc_min": soc_min/100.0,
    "soc_max": 1.0,
    "soc_init": soc_init/100.0,
    "soc_final_min": soc_init/100.0 if enforce_equal_terminal else 0.0,
    "enforce_terminal_equals_init": enforce_equal_terminal,
    "solver_time_limit_s": solver_time,
    # Tarifas simples (série constante)
    "import_fee_series": np.full(len(price_df), float(imp_fee)),
    "export_fee_series": np.full(len(price_df), float(exp_fee)),
    "import_fee_eur_per_MWh": float(imp_fee),
    "export_fee_eur_per_MWh": float(exp_fee),
}

# Rodar
res = run_site_bess_mw(
    price_df=price_df,
    pv_df=pv_df,
    load_df=None,             # sem carga local neste modo simples
    params=params,
    P_bess_MW=float(p_bess_mw),
    c_rate_per_hour=float(c_rate),
    return_schedule=True
)

# =====================================================
# 7) Saídas – faturamento anual e métricas
# =====================================================
st.header("Resultados")

# “Faturamento anual”: receita de exportação + economia por autoconsumo
faturamento = float(res["Revenue_export_annual_EUR"] + res["Savings_selfcons_annual_EUR"])

m1, m2, m3 = st.columns(3)
m1.metric("Faturamento anual (€/ano)", euro(faturamento, 0))
m2.metric("EBITDA (€/ano)", euro(res["EBITDA_project_annual_EUR"], 0))
m3.metric("ROI anual (%)", fmt_pt(res["ROI_annual_%"], 2))

m4, m5, m6 = st.columns(3)
m4.metric("Payback (anos)", fmt_pt(res["Payback_years"], 2))
m5.metric("Throughput (MWh/ano)", fmt_pt(res["Throughput_annual_MWh"], 0))
m6.metric("LCOE total (€/MWh)", fmt_pt(res["LCOE_total_EUR_per_MWh"], 2))

st.markdown("**Componentes**")
k1, k2, k3 = st.columns(3)
k1.metric("Receita de exportação (€/ano)", euro(res["Revenue_export_annual_EUR"], 0))
k2.metric("Economia por autoconsumo (€/ano)", euro(res["Savings_selfcons_annual_EUR"], 0))
k3.metric("Custo de carga pela rede (€/ano)", euro(res["Cost_charge_grid_annual_EUR"], 0))

# tabela/schedule (amostra)
with st.expander("Ver schedule (amostra)"):
    show = res["schedule"].iloc[::max(1, len(res["schedule"])//2000)]
    st.dataframe(show.head(400))

    fig, ax = plt.subplots()
    for col in ["pv_MWh","c_grid_MWh","c_pv_MWh","d_grid_MWh","d_load_MWh","pv_exp_MWh","pv_load_MWh"]:
        if col in show.columns:
            ax.plot(show["datetime"], show[col], label=col, lw=0.8)
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.2); ax.set_title("Fluxos (MWh por passo)")
    st.pyplot(fig)

# PDF simples
st.subheader("Exportar PDF")
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

def build_pdf(res, faturamento):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); W,H=A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H-2.4*cm, "Relatório — BESS (modo simples)")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H-3.1*cm, f"Gerado em {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
    c.line(2*cm, H-3.3*cm, W-2*cm, H-3.3*cm)

    y=H-4.2*cm; c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Faturamento anual: {euro(faturamento,0)}"); y-=0.45*cm
    c.drawString(2*cm, y, f"EBITDA: {euro(res['EBITDA_project_annual_EUR'],0)}   ROI: {fmt_pt(res['ROI_annual_%'],2)} %"); y-=0.45*cm
    c.drawString(2*cm, y, f"Payback: {fmt_pt(res['Payback_years'],2)} anos   LCOE: {fmt_pt(res['LCOE_total_EUR_per_MWh'],2)} €/MWh"); y-=0.45*cm
    c.drawString(2*cm, y, f"Throughput: {fmt_pt(res['Throughput_annual_MWh'],0)} MWh/ano"); y-=0.8*cm
    c.drawString(2*cm, y, f"Receita exportação: {euro(res['Revenue_export_annual_EUR'],0)}"); y-=0.45*cm
    c.drawString(2*cm, y, f"Economia autoconsumo: {euro(res['Savings_selfcons_annual_EUR'],0)}"); y-=0.45*cm
    c.drawString(2*cm, y, f"Custo carga rede: {euro(res['Cost_charge_grid_annual_EUR'],0)}")

    c.showPage(); c.save(); buf.seek(0); return buf

pdf = build_pdf(res, faturamento)
st.download_button("⬇️ Baixar PDF", data=pdf, file_name="bess_modo_simples.pdf", mime="application/pdf")
