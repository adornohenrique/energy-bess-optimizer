import streamlit as st
import pandas as pd

from core.optimization import run_optimization  # você vai implementar


st.set_page_config(page_title="Energy + BESS Optimizer", layout="wide")


def main():
    st.title("Energy Plant + BESS – Otimizador de Projeto")

    # --- Estado de cenários em memória (para MVP) ---
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = {}
    if "current_scenario_name" not in st.session_state:
        st.session_state.current_scenario_name = None

    # Sidebar: seleção e gestão de cenários
    st.sidebar.header("Cenários")

    scenario_names = list(st.session_state.scenarios.keys())
    selected = st.sidebar.selectbox(
        "Selecionar cenário",
        options=["(novo cenário)"] + scenario_names
    )

    if selected != "(novo cenário)":
        st.session_state.current_scenario_name = selected
        scenario = st.session_state.scenarios[selected]
    else:
        scenario = None

    new_name = st.sidebar.text_input("Nome do cenário", value=selected if selected != "(novo cenário)" else "Cenário 1")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("Salvar/Atualizar"):
        st.session_state.scenarios[new_name] = scenario or {}
        st.session_state.current_scenario_name = new_name
        st.sidebar.success(f"Cenário '{new_name}' salvo.")

    if col_btn2.button("Duplicar cenário atual", disabled=scenario is None):
        if scenario is not None:
            dup_name = new_name + " (cópia)"
            st.session_state.scenarios[dup_name] = scenario.copy()
            st.session_state.current_scenario_name = dup_name
            st.sidebar.success(f"Cenário duplicado como '{dup_name}'.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Excluir cenário atual", disabled=scenario is None):
        if scenario is not None:
            del st.session_state.scenarios[st.session_state.current_scenario_name]
            st.session_state.current_scenario_name = None
            st.sidebar.success("Cenário excluído.")

    # --- Abas: Inputs / Resultados / Comparar ---
    tab_inputs, tab_results, tab_compare = st.tabs(["Inputs do cenário", "Resultados", "Comparar cenários"])

    with tab_inputs:
        st.subheader("Parâmetros do cenário")

        col1, col2 = st.columns(2)

        with col1:
    st.markdown("### Usina de geração")
    capex_gen = st.number_input(
        "CAPEX da planta de geração (EUR)",
        min_value=0.0,
        value=10_000_000.0,
        step=100_000.0,
    )
    plant_mwp = st.number_input(
        "Capacidade instalada da planta (MWp)",
        min_value=0.0,
        value=100.0,
        step=1.0,
    )

        with col2:
            st.markdown("### Preços de energia")
            price_source = st.radio("Fonte de preços", ["Arquivo CSV", "API (a implementar)"])
            price_file = None
            if price_source == "Arquivo CSV":
                price_file = st.file_uploader("Preços spot (CSV)", type=["csv"], key="price_file")

        st.markdown("### Parâmetros do BESS")
        colB1, colB2, colB3 = st.columns(3)
        with colB1:
            c_E_capex = st.number_input("CAPEX BESS (EUR/kWh)", min_value=0.0, value=250.0, step=10.0)
            c_P_capex = st.number_input("CAPEX BESS (EUR/kW)", min_value=0.0, value=150.0, step=10.0)
        with colB2:
            lifetime_years = st.number_input("Vida útil (anos)", min_value=1, value=15, step=1)
            discount_rate = st.number_input("Taxa de desconto (%)", min_value=0.0, value=8.0, step=0.5)
        with colB3:
            eta_charge = st.number_input("Eficiência de carga (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
            eta_discharge = st.number_input("Eficiência de descarga (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
            allow_grid_charging = st.checkbox("Permitir carga pela rede?", value=True)

        st.markdown("### Targets financeiros")
        colT1, colT2 = st.columns(2)
        with colT1:
            roi_target = st.number_input("ROI alvo (%/ano) (opcional)", value=15.0, step=1.0)
        with colT2:
            ebitda_target = st.number_input("EBITDA alvo (EUR/ano) (opcional)", value=0.0, step=100_000.0)

        opt_mode = st.radio(
            "Modo de otimização",
            ["Encontrar BESS mínimo que cumpre ROI alvo", "Maximizar ROI/EBITDA (sem target obrigatório)"]
        )

        run_button = st.button("Rodar otimização", type="primary")

       if run_button:
    if price_source == "Arquivo CSV" and price_file is None:
        st.error("Por favor, envie o arquivo de preços.")
    else:
        with st.spinner("Rodando modelo de otimização..."):
            # Lê arquivo de preços
            price_df = pd.read_csv(price_file) if price_file is not None else None

            # Monta config do cenário (simplificado)
            scenario_cfg = {
                "capex_gen": capex_gen,
                "plant_mwp": plant_mwp,
                "c_E_capex": c_E_capex,
                "c_P_capex": c_P_capex,
                "lifetime_years": lifetime_years,
                "discount_rate": discount_rate,
                "eta_charge": eta_charge / 100.0,
                "eta_discharge": eta_discharge / 100.0,
                "allow_grid_charging": allow_grid_charging,
                "roi_target": roi_target / 100.0,
                "ebitda_target": ebitda_target if ebitda_target > 0 else None,
                "opt_mode": opt_mode,
            }

            # Agora o run_optimization gera a curva de geração
            opt_result = run_optimization(
                gen_df=None,
                price_df=price_df,
                scenario=scenario_cfg,
            )

            if st.session_state.current_scenario_name is None:
                st.session_state.current_scenario_name = new_name
            st.session_state.scenarios[st.session_state.current_scenario_name] = {
                "config": scenario_cfg,
                "result": opt_result,
            }

            st.success("Otimização concluída! Veja a aba 'Resultados'.")

    with tab_results:
        st.subheader("Resultados do cenário atual")
        current_name = st.session_state.current_scenario_name
        if not current_name or current_name not in st.session_state.scenarios:
            st.info("Nenhum cenário selecionado ou rodado ainda.")
        else:
            data = st.session_state.scenarios[current_name]
            result = data.get("result")
            if not result:
                st.info("Este cenário ainda não foi otimizado.")
            else:
                st.markdown(f"### Cenário: **{current_name}**")
                colR1, colR2, colR3 = st.columns(3)
                colR1.metric("Capacidade ótima (MWh)", f"{result['E_cap_opt_MWh']:.2f}")
                colR2.metric("Potência ótima (MW)", f"{result['P_cap_opt_MW']:.2f}")
                colR3.metric("ROI (%)", f"{result['ROI_percent']:.2f}")

                st.metric("EBITDA (EUR/ano)", f"{result['EBITDA_annual_EUR']:.0f}")
                st.metric("Status", result["status_text"])

                if "schedule" in result:
                    schedule_df = result["schedule"]
                    st.markdown("#### Preview do schedule")
                    st.dataframe(schedule_df.head(200))

    with tab_compare:
        st.subheader("Comparar cenários")
        if not st.session_state.scenarios:
            st.info("Nenhum cenário salvo ainda.")
        else:
            rows = []
            for name, data in st.session_state.scenarios.items():
                res = data.get("result")
                if not res:
                    continue
                rows.append({
                    "Cenário": name,
                    "E_cap_MWh": res["E_cap_opt_MWh"],
                    "P_cap_MW": res["P_cap_opt_MW"],
                    "ROI_%": res["ROI_percent"],
                    "EBITDA_EUR_ano": res["EBITDA_annual_EUR"],
                    "Status": res["status_text"],
                })
            if rows:
                table = pd.DataFrame(rows)
                st.dataframe(table)
            else:
                st.info("Nenhum cenário com resultados ainda.")


if __name__ == "__main__":
    main()
