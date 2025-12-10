import os
import sys
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------- Formatação PT/BR ----------
def fmt_pt(x, decimals=0):
    """Formata número com . para milhar e , para decimais."""
    if x is None:
        return "-"
    try:
        s = f"{float(x):,.{decimals}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def euro(x, decimals=0):
    return f"€ {fmt_pt(x, decimals)}"

# Tornar o pacote raiz visível (para Streamlit Cloud)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.optimization import (
    run_baseline,
    run_with_bess,
    run_sensitivities,
    run_batch_scenarios,
)

st.set_page_config(page_title="Energy + BESS Optimizer", layout="wide")

# ---------------------------------------------------------
# Ajuda & Legenda
# ---------------------------------------------------------
with st.expander("ℹ️ **Ajuda & Legenda** (clique para expandir)"):
    st.markdown("""
**Como rodar:**
1. Envie dois CSVs:  
   • **Preços**: `datetime,price_EUR_per_MWh`  
   • **Geração**: `datetime,gen_MWh`  
   (mesmo passo temporal – ex.: 15 min)
2. Preencha **parâmetros econômicos** (CAPEX, vida útil, taxa, OPEX, disponibilidade).
3. Defina **limites de conexão** (export/import) e **eficiências do BESS**.
4. Informe **CAPEX BESS** (EUR/kWh, EUR/kW) e parâmetros de **degradação** (€/MWh_throughput, ciclos/ano).
5. Veja **Baseline** e **Com BESS (MILP)**. O MILP alterna carga/descarga com **Big-M**.
6. Opcional: **Sensibilidades** (±10/±20%) e **múltiplos cenários** (P50/P90...).
7. Exporte **PDF** com um ou com **todos** os cenários.

**Abreviações:**  
- **LCOE**: Levelized Cost of Energy (EUR/MWh)  
- **EBITDA**: Earnings Before Interest, Taxes, Depreciation and Amortization  
- **BESS**: Battery Energy Storage System  
- **E_cap (MWh)**: capacidade de energia da bateria  
- **P_cap (MW)**: potência da bateria  
- **Throughput (MWh/ano)**: carga + descarga no ano  
- **CRF**: Capital Recovery Factor  
- **P50/P90**: cenários de probabilidade
""")

# ---------------------------------------------------------
# Modo de execução
# ---------------------------------------------------------
mode = st.radio("Modo de execução", ["Cenário único", "Múltiplos cenários (P50/P90, etc.)"])

# ---------------------------------------------------------
# Uploads
# ---------------------------------------------------------
st.header("Dados de entrada")

if mode == "Cenário único":
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Preços de energia")
        price_file = st.file_uploader("CSV (colunas: datetime, price_EUR_per_MWh)", type=["csv"], key="price")
        price_df = pd.read_csv(price_file) if price_file is not None else None

    with c2:
        st.subheader("Geração da planta")
        gen_file = st.file_uploader("CSV (colunas: datetime, gen_MWh)", type=["csv"], key="gen")
        gen_df = pd.read_csv(gen_file) if gen_file is not None else None

    if price_df is None or gen_df is None:
        st.info("Envie **preços** e **geração** para continuar.")
        st.stop()

else:
    st.subheader("Upload de **múltiplos** CSVs")
    c1, c2 = st.columns(2)
    with c1:
        price_files = st.file_uploader("CSV de preços (vários) – `datetime,price_EUR_per_MWh`",
                                       type=["csv"], key="price_multi", accept_multiple_files=True)
    with c2:
        gen_files = st.file_uploader("CSV de geração (vários) – `datetime,gen_MWh`",
                                     type=["csv"], key="gen_multi", accept_multiple_files=True)

    labels_raw = st.text_input("Rótulos (na ordem dos arquivos) – ex.: P50,P90", value="P50,P90")
    labels = [x.strip() for x in labels_raw.split(",") if x.strip()]

    if not price_files or not gen_files or len(price_files) != len(gen_files) or len(labels) != len(price_files):
        st.info("Envie **mesmo número** de arquivos de preço e geração e informe **rótulos** na mesma quantidade.")
        st.stop()

    price_df_list = [pd.read_csv(f) for f in price_files]
    gen_df_list = [pd.read_csv(f) for f in gen_files]

# ---------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------
st.header("Parâmetros econômicos e operacionais")
colA, colB, colC = st.columns(3)

with colA:
    capex_gen = st.number_input("CAPEX da usina (EUR)", min_value=0.0, value=10_000_000.0, step=100_000.0)
    lifetime_years = st.number_input("Vida útil (anos)", min_value=1, value=15, step=1)
    discount_rate = st.number_input("Taxa de desconto (%)", min_value=0.0, value=8.0, step=0.5)
    availability_pct = st.number_input("Disponibilidade da planta (%)", min_value=0.0, max_value=100.0, value=97.0, step=0.5)

with colB:
    opex_fix_gen = st.number_input("OPEX fixo usina (EUR/ano)", min_value=0.0, value=200_000.0, step=10_000.0)
    opex_fix_bess = st.number_input("OPEX fixo BESS (EUR/ano)", min_value=0.0, value=60_000.0, step=5_000.0)
    opex_var = st.number_input("OPEX variável (EUR/MWh vendido)", min_value=0.0, value=1.5, step=0.1)

with colC:
    P_grid_export_max = st.number_input("Export máx (MW)", min_value=0.0, value=100.0, step=1.0)
    P_grid_import_max = st.number_input("Import máx (MW)", min_value=0.0, value=100.0, step=1.0)
    allow_grid_charging = st.checkbox("Permitir carga pela rede?", value=True)

# --------- BESS (ÚNICA SEÇÃO) ---------
st.subheader("BESS – custos, eficiência e degradação")
colE, colF, colG = st.columns(3)
with colE:
    c_E_capex = st.number_input("CAPEX BESS (EUR/kWh)", min_value=0.0, value=250.0, step=10.0, key="cE")
    c_P_capex = st.number_input("CAPEX BESS (EUR/kW)", min_value=0.0, value=150.0, step=10.0, key="cP")
with colF:
    eta_charge = st.number_input("Eficiência de carga (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0, key="etac")
    eta_discharge = st.number_input("Eficiência de descarga (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0, key="etad")
with colG:
    deg_cost = st.number_input("Custo de degradação (EUR/MWh throughput)", min_value=0.0, value=2.0, step=0.5, key="deg")
    cycles_max = st.number_input("Limite de ciclos/ano (0 = sem limite)", min_value=0, value=300, step=10, key="cycles")

run_sens = st.checkbox("Rodar sensibilidades (±10/±20% em c_E e c_P)?", value=True)

params_common = {
    "capex_gen": capex_gen,
    "lifetime_years": lifetime_years,
    "discount_rate": discount_rate,
    "availability_pct": availability_pct,
    "opex_fix_gen": opex_fix_gen,
    "opex_fix_bess": opex_fix_bess,
    "opex_var_eur_per_mwh": opex_var,
    "P_grid_export_max": P_grid_export_max,
    "P_grid_import_max": P_grid_import_max,
    "allow_grid_charging": allow_grid_charging,
    "c_E_capex": c_E_capex,
    "c_P_capex": c_P_capex,
    "eta_charge": eta_charge / 100.0,
    "eta_discharge": eta_discharge / 100.0,
    "deg_cost_eur_per_MWh_throughput": deg_cost,
    "cycles_per_year_max": cycles_max,
}

# ---------------------------------------------------------
# Helpers de gráfico (também usados no PDF)
# ---------------------------------------------------------
def _make_compare_figure(baseline, with_bess):
    fig, ax = plt.subplots()
    items = ["LCOE (€/MWh)", "EBITDA (M€/ano)"]
    base_vals = [baseline["LCOE_base_EUR_per_MWh"], baseline["Revenue_annual_EUR"] / 1e6]
    bess_vals = [with_bess["LCOE_with_BESS_EUR_per_MWh"], with_bess["EBITDA_annual_EUR"] / 1e6]
    x = range(len(items))
    ax.bar([i - 0.15 for i in x], base_vals, width=0.3, label="Sem BESS")
    ax.bar([i + 0.15 for i in x], bess_vals, width=0.3, label="Com BESS")
    ax.set_xticks(list(x))
    ax.set_xticklabels(items)
    ax.set_title("Comparação Sem BESS vs Com BESS")
    ax.legend()
    return fig

def _make_schedule_figure(df, title="Schedule – energia (MWh por passo)"):
    fig, ax = plt.subplots()
    ax.plot(df["datetime"], df.get("gen_MWh", 0), label="Geração")
    if "sold_from_gen_MWh" in df.columns:
        ax.plot(df["datetime"], df["sold_from_gen_MWh"], label="Venda direta")
    if "discharge_MWh" in df.columns:
        ax.plot(df["datetime"], df["discharge_MWh"], label="Descarga BESS")
    if "charge_from_grid_MWh" in df.columns:
        ax.plot(df["datetime"], df["charge_from_grid_MWh"], label="Carga da rede")
    ax.set_title(title)
    ax.legend()
    return fig

def _plot_compare_baseline_bess(baseline, with_bess):
    fig = _make_compare_figure(baseline, with_bess)
    st.pyplot(fig)
    return fig

def _plot_schedule(df, title="Schedule – energia (MWh por passo)"):
    fig = _make_schedule_figure(df, title)
    st.pyplot(fig)
    return fig

# ---------------------------------------------------------
# PDFs (com números formatados PT/BR)
# ---------------------------------------------------------
def _build_pdf_single(baseline, with_bess, fig_compare, fig_sched, project_name="Projeto", scenario_name="Cenário"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, height-2.5*cm, f"Relatório – {project_name}")
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, height-3.5*cm, f"{scenario_name} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    c.line(2*cm, height-3.8*cm, width-2*cm, height-3.8*cm)

    y = height-5.0*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "KPIs principais")
    y -= 0.6*cm
    c.setFont("Helvetica", 11)
    lines = [
        f"LCOE base: {euro(baseline['LCOE_base_EUR_per_MWh'], 2)} /MWh",
        f"Receita base: {euro(baseline['Revenue_annual_EUR'], 0)}/ano",
    ]
    if with_bess and with_bess.get("status_text", "").startswith("Solução"):
        lines += [
            f"LCOE com BESS: {euro(with_bess['LCOE_with_BESS_EUR_per_MWh'], 2)} /MWh",
            f"EBITDA com BESS: {euro(with_bess['EBITDA_annual_EUR'], 0)}/ano",
            f"BESS ótimo: {fmt_pt(with_bess['E_cap_opt_MWh'], 2)} MWh / {fmt_pt(with_bess['P_cap_opt_MW'], 2)} MW",
            f"Throughput anual: {fmt_pt(with_bess['Throughput_annual_MWh'], 0)} MWh/ano",
        ]
    for line in lines:
        c.drawString(2.3*cm, y, line)
        y -= 0.5*cm

    y_img = y - 0.5*cm
    img_w = width - 4*cm
    img_h = 7*cm
    if fig_compare:
        png1 = io.BytesIO()
        fig_compare.savefig(png1, format="png", bbox_inches="tight")
        png1.seek(0)
        c.drawImage(ImageReader(png1), 2*cm, y_img-img_h, width=img_w, height=img_h)

    c.showPage()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, height-2.5*cm, "Schedule – energia")
    if fig_sched:
        png2 = io.BytesIO()
        fig_sched.savefig(png2, format="png", bbox_inches="tight")
        png2.seek(0)
        c.drawImage(ImageReader(png2), 2*cm, height-2.5*cm-12*cm, width=width-4*cm, height=10*cm)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def _build_pdf_multi(results, compare_table, project_name="Energy+BESS – Batch"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Capa
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, height-2.5*cm, f"Relatório – {project_name}")
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, height-3.5*cm, f"{len(results)} cenários | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    c.line(2*cm, height-3.8*cm, width-2*cm, height-3.8*cm)
    c.showPage()

    # Resumo comparativo
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2*cm, height-2.5*cm, "Resumo comparativo (principais KPIs)")
    y = height - 3.3*cm
    c.setFont("Helvetica-Bold", 10)
    headers = ["Cenário", "E_cap (MWh)", "P_cap (MW)", "EBITDA (EUR/ano)",
               "LCOE base", "LCOE BESS", "Receita base", "Receita BESS", "Throughput (MWh/ano)"]
    col_x = [2*cm, 6.2*cm, 8.2*cm, 10.2*cm, 13.2*cm, 15.2*cm, 2*cm, 6.2*cm, 10.2*cm]
    c.drawString(col_x[0], y, headers[0])
    c.drawString(col_x[1], y, headers[1])
    c.drawString(col_x[2], y, headers[2])
    c.drawString(col_x[3], y, headers[3]); y -= 0.5*cm
    c.drawString(col_x[4], y, headers[4])
    c.drawString(col_x[5], y, headers[5])
    c.drawString(col_x[6], y, headers[6])
    c.drawString(col_x[7], y, headers[7])
    c.drawString(col_x[8], y, headers[8]); y -= 0.4*cm
    c.line(2*cm, y, width-2*cm, y); y -= 0.35*cm
    c.setFont("Helvetica", 9)

    for _, row in compare_table.iterrows():
        if y < 2.5*cm:
            c.showPage()
            c.setFont("Helvetica-Bold", 13)
            c.drawString(2*cm, height-2.5*cm, "Resumo comparativo (continuação)")
            y = height - 3.3*cm
            c.setFont("Helvetica", 9)
        # linha 1
        c.drawString(col_x[0], y, str(row["Cenário"]))
        c.drawRightString(col_x[1]+1.6*cm, y, fmt_pt(row["E_cap_MWh"], 2) if pd.notna(row["E_cap_MWh"]) else "-")
        c.drawRightString(col_x[2]+1.6*cm, y, fmt_pt(row["P_cap_MW"], 2) if pd.notna(row["P_cap_MW"]) else "-")
        c.drawRightString(col_x[3]+3.2*cm, y, euro(row["EBITDA_EUR_ano"], 0) if pd.notna(row["EBITDA_EUR_ano"]) else "-")
        y -= 0.45*cm
        # linha 2
        c.drawRightString(col_x[4]+1.6*cm, y, fmt_pt(row["LCOE_base"], 2) if pd.notna(row["LCOE_base"]) else "-")
        c.drawRightString(col_x[5]+1.6*cm, y, fmt_pt(row["LCOE_com_BESS"], 2) if pd.notna(row["LCOE_com_BESS"]) else "-")
        c.drawRightString(col_x[6]+3.0*cm, y, euro(row["Receita_base_EUR_ano"], 0) if pd.notna(row["Receita_base_EUR_ano"]) else "-")
        c.drawRightString(col_x[7]+3.0*cm, y, euro(row["Receita_BESS_EUR_ano"], 0) if pd.notna(row["Receita_BESS_EUR_ano"]) else "-")
        c.drawRightString(col_x[8]+3.2*cm, y, fmt_pt(row["Throughput_MWh_ano"], 0) if pd.notna(row["Throughput_MWh_ano"]) else "-")
        y -= 0.55*cm

    c.showPage()

    # Páginas por cenário
    for item in results:
        label = item["label"]
        base = item["baseline"]
        w = item["with_bess"]

        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, height-2.5*cm, f"Cenário: {label}")
        c.setFont("Helvetica", 11)

        y = height - 3.5*cm
        lines = [
            f"LCOE base: {euro(base['LCOE_base_EUR_per_MWh'], 2)} /MWh",
            f"Receita base: {euro(base['Revenue_annual_EUR'], 0)}/ano",
        ]
        if w and w.get("status_text", "").startswith("Solução"):
            lines += [
                f"LCOE com BESS: {euro(w['LCOE_with_BESS_EUR_per_MWh'], 2)} /MWh",
                f"EBITDA com BESS: {euro(w['EBITDA_annual_EUR'], 0)}/ano",
                f"BESS ótimo: {fmt_pt(w['E_cap_opt_MWh'], 2)} MWh / {fmt_pt(w['P_cap_opt_MW'], 2)} MW",
                f"Throughput anual: {fmt_pt(w['Throughput_annual_MWh'], 0)} MWh/ano",
            ]
        for line in lines:
            c.drawString(2*cm, y, line)
            y -= 0.55*cm

        if w and w.get("status_text", "").startswith("Solução"):
            fig_compare = _make_compare_figure(base, w)
            png1 = io.BytesIO()
            fig_compare.savefig(png1, format="png", bbox_inches="tight")
            png1.seek(0)
            img_w = width - 4*cm
            img_h = 7*cm
            y_img = y - 0.5*cm
            if y_img - img_h < 2*cm:
                c.showPage()
                y_img = height - 2.5*cm
            c.drawImage(ImageReader(png1), 2*cm, y_img - img_h, width=img_w, height=img_h)
            c.showPage()

            fig_sched = _make_schedule_figure(w["schedule"], f"Schedule – {label}")
            png2 = io.BytesIO()
            fig_sched.savefig(png2, format="png", bbox_inches="tight")
            png2.seek(0)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2*cm, height-2.5*cm, f"Schedule – {label}")
            c.drawImage(ImageReader(png2), 2*cm, height-2.5*cm-12*cm, width=width-4*cm, height=10*cm)
            c.showPage()
        else:
            c.showPage()

    c.save()
    buf.seek(0)
    return buf

# ---------------------------------------------------------
# Execução – Cenário único
# ---------------------------------------------------------
if mode == "Cenário único":
    st.header("Resultados – Baseline (sem BESS)")
    baseline = run_baseline(price_df, gen_df, params_common)
    c1, c2, c3 = st.columns(3)
    c1.metric("Energia anual (MWh/ano)", fmt_pt(baseline['E_annual_MWh'], 0))
    c2.metric("Receita anual (EUR/ano)", euro(baseline['Revenue_annual_EUR'], 0))
    c3.metric("LCOE base (EUR/MWh)", euro(baseline['LCOE_base_EUR_per_MWh'], 2))

    st.dataframe(
        baseline["schedule_baseline"].head(200).style.format({
            "gen_MWh":           lambda v: fmt_pt(v, 3),
            "sold_direct_MWh":   lambda v: fmt_pt(v, 3),
            "spill_MWh":         lambda v: fmt_pt(v, 3),
            "price_EUR_per_MWh": lambda v: fmt_pt(v, 2),
        })
    )

    st.header("Resultados – Com BESS (MILP, arbitragem ótima)")
    res_bess = run_with_bess(price_df, gen_df, params_common)
    if res_bess.get("status_text", "").startswith("Solução"):
        c1, c2, c3 = st.columns(3)
        c1.metric("E_cap ótima (MWh)", fmt_pt(res_bess['E_cap_opt_MWh'], 2))
        c2.metric("P_cap ótima (MW)", fmt_pt(res_bess['P_cap_opt_MW'], 2))
        c3.metric("Energia vendida (MWh/ano)", fmt_pt(res_bess['Energy_annual_MWh'], 0))

        c4, c5, c6 = st.columns(3)
        c4.metric("Receita (EUR/ano)", euro(res_bess['Revenue_annual_EUR'], 0))
        c5.metric("Custo energia rede (EUR/ano)", euro(res_bess['Grid_energy_cost_annual_EUR'], 0))
        c6.metric("EBITDA (EUR/ano)", euro(res_bess['EBITDA_annual_EUR'], 0))

        c7, c8, c9 = st.columns(3)
        c7.metric("CAPEX BESS (EUR)", euro(res_bess['BESS_CAPEX_EUR'], 0))
        c8.metric("LCOE com BESS (EUR/MWh)", euro(res_bess['LCOE_with_BESS_EUR_per_MWh'], 2))
        c9.metric("Throughput (MWh/ano)", fmt_pt(res_bess['Throughput_annual_MWh'], 0))

        st.subheader("Gráficos")
        fig_compare = _plot_compare_baseline_bess(baseline, res_bess)
        fig_sched = _plot_schedule(res_bess["schedule"], "Schedule – Com BESS")

        st.dataframe(
            res_bess["schedule"].head(200).style.format({
                "gen_MWh":                 lambda v: fmt_pt(v, 3),
                "sold_from_gen_MWh":       lambda v: fmt_pt(v, 3),
                "spill_MWh":               lambda v: fmt_pt(v, 3),
                "charge_from_ren_MWh":     lambda v: fmt_pt(v, 3),
                "charge_from_grid_MWh":    lambda v: fmt_pt(v, 3),
                "discharge_MWh":           lambda v: fmt_pt(v, 3),
                "soc_MWh":                 lambda v: fmt_pt(v, 3),
                "price_EUR_per_MWh":       lambda v: fmt_pt(v, 2),
            })
        )
    else:
        st.warning(res_bess.get("status_text", "Falha ao otimizar com BESS."))
        fig_compare, fig_sched = None, None

    # Sensibilidades (opcional)
    if res_bess.get("status_text", "").startswith("Solução") and run_sens and st.checkbox("Mostrar sensibilidades (tabela)", value=True):
        st.header("Sensibilidades – c_E e c_P (±10/±20%)")
        sens_df = run_sensitivities(price_df, gen_df, params_common)
        st.dataframe(
            sens_df.style.format({
                "delta_cE_%":           lambda v: fmt_pt(v, 0),
                "delta_cP_%":           lambda v: fmt_pt(v, 0),
                "E_cap_MWh":            lambda v: fmt_pt(v, 2),
                "P_cap_MW":             lambda v: fmt_pt(v, 2),
                "EBITDA_annual_EUR":    lambda v: euro(v, 0),
                "LCOE_with_BESS":       lambda v: fmt_pt(v, 2),
            })
        )

    # Exportar PDF do cenário único
    st.header("Exportar PDF")
    pdf_bytes = _build_pdf_single(
        baseline=baseline,
        with_bess=res_bess if res_bess.get("status_text", "").startswith("Solução") else None,
        fig_compare=fig_compare,
        fig_sched=fig_sched,
        project_name="Energy+BESS",
        scenario_name="Cenário único",
    )
    st.download_button(
        "⬇️ Baixar PDF (narrativa + gráficos)",
        data=pdf_bytes,
        file_name="energy_bess_report.pdf",
        mime="application/pdf",
    )

# ---------------------------------------------------------
# Execução – Múltiplos cenários
# ---------------------------------------------------------
else:
    st.header("Resultados – Vários cenários (P50/P90, etc.)")
    results, compare_table = run_batch_scenarios(price_df_list, gen_df_list, labels, params_common)

    st.dataframe(
        compare_table.style.format({
            "E_cap_MWh":             lambda v: "-" if pd.isna(v) else fmt_pt(v, 2),
            "P_cap_MW":              lambda v: "-" if pd.isna(v) else fmt_pt(v, 2),
            "EBITDA_EUR_ano":        lambda v: "-" if pd.isna(v) else euro(v, 0),
            "LCOE_base":             lambda v: "-" if pd.isna(v) else fmt_pt(v, 2),
            "LCOE_com_BESS":         lambda v: "-" if pd.isna(v) else fmt_pt(v, 2),
            "Receita_base_EUR_ano":  lambda v: "-" if pd.isna(v) else euro(v, 0),
            "Receita_BESS_EUR_ano":  lambda v: "-" if pd.isna(v) else euro(v, 0),
            "Throughput_MWh_ano":    lambda v: "-" if pd.isna(v) else fmt_pt(v, 0),
        })
    )

    if len(compare_table) > 0:
        fig, ax = plt.subplots()
        ax.bar(compare_table["Cenário"], compare_table["LCOE_base"], label="LCOE base")
        ax.plot(compare_table["Cenário"], compare_table["LCOE_com_BESS"], marker="o", label="LCOE com BESS")
        ax.set_title("LCOE – Base vs Com BESS (por cenário)")
        ax.legend()
        st.pyplot(fig)

    # Exporta um único PDF com TODOS os cenários
    st.header("Exportar PDF – TODOS os cenários")
    pdf_all = _build_pdf_multi(results, compare_table, project_name="Energy+BESS – Todos os cenários")
    st.download_button(
        "⬇️ Baixar PDF (todos os cenários)",
        data=pdf_all,
        file_name="energy_bess_all_scenarios.pdf",
        mime="application/pdf",
    )
