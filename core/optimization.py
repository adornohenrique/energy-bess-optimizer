import pandas as pd
import pulp

from core.finance import (
    crf,
    bess_capex_eur,
    annualized_bess_cost_eur,
    ebitda_annual_eur,
    roi_from_ebitda,
)


def _build_df_with_generation(price_df, scenario, gen_df=None):
    """
    Prepara o DataFrame com:
      datetime, price_EUR_per_MWh, gen_MWh

    Se gen_df for None, gera gen_MWh a partir de:
      - plant_mwp (MWp)
      - perfil solar sintético (6h–18h em formato "sino").
    """
    if "datetime" not in price_df.columns:
        raise ValueError("prices.csv precisa ter coluna 'datetime'.")
    if "price_EUR_per_MWh" not in price_df.columns:
        raise ValueError("prices.csv precisa ter coluna 'price_EUR_per_MWh'.")

    df_p = price_df.copy()
    df_p["datetime"] = pd.to_datetime(df_p["datetime"])
    df_p = df_p.sort_values("datetime").reset_index(drop=True)

    if len(df_p) < 2:
        raise ValueError("Precisam haver pelo menos 2 linhas de preços.")

    dt_seconds = (df_p.loc[1, "datetime"] - df_p.loc[0, "datetime"]).total_seconds()
    dt_hours = dt_seconds / 3600.0
    if dt_hours <= 0:
        raise ValueError("Passo de tempo inválido (datetime).")

    # Caso 1: o usuário passou um gen_df com gen_MWh -> usa direto
    if gen_df is not None and "gen_MWh" in gen_df.columns:
        g = gen_df.copy()
        g["datetime"] = pd.to_datetime(g["datetime"])
        g = g.sort_values("datetime").reset_index(drop=True)
        df = g.merge(df_p, on="datetime", how="inner")
        return df, dt_hours

    # Caso 2: gerar curva sintética a partir de MWp
    plant_mwp = float(scenario.get("plant_mwp", 0.0))
    if plant_mwp <= 0:
        raise ValueError(
            "É necessário informar 'plant_mwp' > 0 no cenário para gerar a curva de geração."
        )

    df = df_p.copy()

    # Perfil solar genérico: 0 fora de 6–18h; dentro, cf = sin(pi * x)
    hours_float = (
        df["datetime"].dt.hour
        + df["datetime"].dt.minute / 60.0
        + df["datetime"].dt.second / 3600.0
    )
    sunrise = 6.0
    sunset = 18.0
    span = sunset - sunrise

    cap_factors = []
    from math import sin, pi

    for h in hours_float:
        if h <= sunrise or h >= sunset:
            cap_factors.append(0.0)
        else:
            x = (h - sunrise) / span  # 0–1
            cap_factors.append(max(0.0, sin(pi * x)))

    df["cap_factor"] = cap_factors
    df["gen_MWh"] = plant_mwp * df["cap_factor"] * dt_hours

    return df, dt_hours


def run_optimization(gen_df, price_df, scenario):
    """
    Otimiza tamanho da bateria (E_cap, P_cap) e operação (charge/discharge),
    com objetivo de MAXIMIZAR EBITDA ANUAL, sujeito à restrição de LCOE:

        LCOE_total <= LCOE_base * (1 + margem%)

    Onde:
      LCOE_total = Cost_annual / E_annual
      Cost_annual = CRF * (CAPEX_gen + CAPEX_bess)
      E_annual = energia entregue no ano (MWh)

    Parâmetros em 'scenario':
      capex_gen           : CAPEX da usina de geração (EUR)
      plant_mwp           : capacidade instalada (MWp)
      P_grid_max          : potência máxima de exportação (MW)
      c_E_capex           : CAPEX BESS (EUR/kWh)
      c_P_capex           : CAPEX BESS (EUR/kW)
      lifetime_years      : vida útil da bateria (anos)
      discount_rate       : taxa de desconto (% a.a.)
      eta_charge          : eficiência de carga (0–1)
      eta_discharge       : eficiência de descarga (0–1)
      allow_grid_charging : bool
      lcoe_base           : LCOE de referência (EUR/MWh)
      lcoe_margin_pct     : margem (%) sobre o LCOE base
    """
    if price_df is None:
        raise ValueError("price_df não pode ser None.")

    df, dt_hours = _build_df_with_generation(price_df, scenario, gen_df)
    T = len(df)

    # Fator para anualizar (se a série não for 1 ano)
    hours_in_series = T * dt_hours
    if hours_in_series <= 0:
        raise ValueError("Série de preços com duração inválida.")
    annual_factor = 8760.0 / hours_in_series

    # ----- Parâmetros financeiros e operacionais -----
    capex_gen = float(scenario["capex_gen"])
    c_E_capex = float(scenario["c_E_capex"])  # EUR/kWh
    c_P_capex = float(scenario["c_P_capex"])  # EUR/kW
    lifetime_years = int(scenario["lifetime_years"])
    discount_rate = float(scenario["discount_rate"])
    eta_c = float(scenario["eta_charge"])
    eta_d = float(scenario["eta_discharge"])
    allow_grid_charging = bool(scenario["allow_grid_charging"])
    lcoe_base = float(scenario["lcoe_base"])
    lcoe_margin_pct = float(scenario["lcoe_margin_pct"])

    # Target de LCOE
    lcoe_target = lcoe_base * (1.0 + lcoe_margin_pct / 100.0)

    P_grid_max = float(scenario.get("P_grid_max", 0.0))
    if P_grid_max <= 0:
        P_grid_max = 1e9  # Sem limite prático

    # ----- Problema de otimização: maximizar EBITDA -----
    prob = pulp.LpProblem("BESS_size_and_operation_LCOE", pulp.LpMaximize)

    # Variáveis globais (tamanho da bateria)
    E_cap = pulp.LpVariable("E_cap_MWh", lowBound=0)
    P_cap = pulp.LpVariable("P_cap_MW", lowBound=0)

    # Variáveis por período
    soc = pulp.LpVariable.dicts("soc", range(T), lowBound=0)
    ch_ren = pulp.LpVariable.dicts("charge_from_ren", range(T), lowBound=0)
    ch_grid = pulp.LpVariable.dicts("charge_from_grid", range(T), lowBound=0)
    dis = pulp.LpVariable.dicts("discharge", range(T), lowBound=0)
    sold_gen = pulp.LpVariable.dicts("sold_from_gen", range(T), lowBound=0)
    spill = pulp.LpVariable.dicts("spill", range(T), lowBound=0)

    # ----- Restrições operacionais -----
    for t in range(T):
        g_t = df.loc[t, "gen_MWh"]

        # Balanço da geração
        prob += sold_gen[t] + ch_ren[t] + spill[t] == g_t, "gen_balance_%d" % t

        # Limites de potência da bateria (MWh/intervalo)
        prob += ch_ren[t] + ch_grid[t] <= P_cap * dt_hours, "ch_power_%d" % t
        prob += dis[t] <= P_cap * dt_hours, "dis_power_%d" % t

        # Limite de exportação para a rede (MW -> MWh no intervalo)
        prob += sold_gen[t] + dis[t] <= P_grid_max * dt_hours, "grid_export_%d" % t

        # Proibir carga da rede, se configurado
        if not allow_grid_charging:
            prob += ch_grid[t] == 0, "no_grid_charge_%d" % t

        # Capacidade de energia da bateria
        prob += soc[t] <= E_cap, "soc_cap_%d" % t

        # Dinâmica do SOC (cíclica)
        prev = T - 1 if t == 0 else t - 1
        prob += (
            soc[t]
            == soc[prev] + eta_c * (ch_ren[t] + ch_grid[t]) - dis[t] / eta_d
        ), "soc_dyn_%d" % t

    # ----- Termos de receita / custo / energia -----
    revenue_terms = []
    grid_cost_terms = []
    energy_terms = []

    for t in range(T):
        price_t = df.loc[t, "price_EUR_per_MWh"]
        revenue_terms.append(price_t * (sold_gen[t] + dis[t]))
        grid_cost_terms.append(price_t * ch_grid[t])
        energy_terms.append(sold_gen[t] + dis[t])

    # Anualização
    revenue_expr = annual_factor * pulp.lpSum(revenue_terms)
    grid_cost_expr = annual_factor * pulp.lpSum(grid_cost_terms)
    energy_annual_expr = annual_factor * pulp.lpSum(energy_terms)

    # CAPEX BESS e custo anualizado
    bess_capex_expr = c_E_capex * 1000.0 * E_cap + c_P_capex * 1000.0 * P_cap
    capex_total_expr = capex_gen + bess_capex_expr
    cost_annual_expr = crf(discount_rate, lifetime_years) * capex_total_expr
    bess_annual_cost_expr = crf(discount_rate, lifetime_years) * bess_capex_expr

    # EBITDA anual (simplificado)
    ebitda_expr = revenue_expr - grid_cost_expr - bess_annual_cost_expr

    # ----- Restrição de LCOE -----
    # LCOE_total = Cost_annual / E_annual <= LCOE_target
    # => Cost_annual <= LCOE_target * E_annual
    prob += cost_annual_expr <= lcoe_target * energy_annual_expr, "lcoe_constraint"

    # ----- Objetivo: maximizar EBITDA anual -----
    prob += ebitda_expr

    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return {
            "status_text": "Modelo não encontrou solução ótima (status: %s)" % status,
            "E_cap_opt_MWh": 0.0,
            "P_cap_opt_MW": 0.0,
            "ROI_percent": 0.0,
            "EBITDA_annual_EUR": 0.0,
            "LCOE_target_EUR_per_MWh": lcoe_target,
            "LCOE_actual_EUR_per_MWh": None,
            "BESS_required": False,
            "BESS_required_text": "Indeterminado (problema sem solução ótima)",
            "schedule": df,
        }

    # ----- Extrair valores escalares -----
    E_cap_opt = E_cap.value()
    P_cap_opt = P_cap.value()
    bess_capex_val = bess_capex_eur(
        E_cap_opt, P_cap_opt, c_E_capex, c_P_capex
    )
    bess_annual_cost_val = annualized_bess_cost_eur(
        bess_capex_val, discount_rate, lifetime_years
    )

    revenue_val = pulp.value(revenue_expr)
    grid_cost_val = pulp.value(grid_cost_expr)
    ebitda_val = ebitda_annual_eur(
        revenue_val, grid_cost_val, bess_annual_cost_val
    )
    capex_total_val = capex_gen + bess_capex_val
    roi_val = roi_from_ebitda(ebitda_val, capex_total_val)

    energy_annual_val = pulp.value(energy_annual_expr)
    cost_annual_val = pulp.value(cost_annual_expr)
    if energy_annual_val > 0:
        lcoe_actual = cost_annual_val / energy_annual_val
    else:
        lcoe_actual = None

    # BESS é necessário?
    bess_required = (E_cap_opt > 1e-6) or (P_cap_opt > 1e-6)
    if bess_required:
        bess_required_text = "Sim, BESS diferente de zero é ótimo."
    else:
        bess_required_text = "Não: a planta atende o target de LCOE sem BESS."

    # Montar schedule
    sched = df.copy()
    sched["soc_MWh"] = [soc[t].value() for t in range(T)]
    sched["charge_from_ren_MWh"] = [ch_ren[t].value() for t in range(T)]
    sched["charge_from_grid_MWh"] = [ch_grid[t].value() for t in range(T)]
    sched["discharge_MWh"] = [dis[t].value() for t in range(T)]
    sched["sold_from_gen_MWh"] = [sold_gen[t].value() for t in range(T)]
    sched["spill_MWh"] = [spill[t].value() for t in range(T)]

    result = {
        "status_text": "Solução ótima encontrada com restrição de LCOE",
        "E_cap_opt_MWh": E_cap_opt,
        "P_cap_opt_MW": P_cap_opt,
        "EBITDA_annual_EUR": ebitda_val,
        "ROI_percent": roi_val * 100.0,
        "BESS_CAPEX_EUR": bess_capex_val,
        "BESS_annual_cost_EUR": bess_annual_cost_val,
        "Revenue_EUR": revenue_val,
        "Grid_energy_cost_EUR": grid_cost_val,
        "LCOE_target_EUR_per_MWh": lcoe_target,
        "LCOE_actual_EUR_per_MWh": lcoe_actual,
        "BESS_required": bess_required,
        "BESS_required_text": bess_required_text,
        "schedule": sched,
    }
    return result
