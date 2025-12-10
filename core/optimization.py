import numpy as np
import pandas as pd
import pulp

from core.finance import crf, annualized_cost, lcoe_annual, bess_capex_eur


# ==============================
# Utilitários tempo e merge
# ==============================
def _infer_dt_hours(dt_series: pd.Series) -> float:
    s = pd.to_datetime(dt_series)
    if len(s) < 2:
        raise ValueError("Série temporal curta para inferir Δt.")
    dt = (s.iloc[1] - s.iloc[0]).total_seconds() / 3600.0
    if dt <= 0:
        raise ValueError("Δt inválido.")
    return dt


def _annual_factor(n_steps: int, dt_h: float) -> float:
    hours = n_steps * dt_h
    if hours <= 0:
        raise ValueError("Duração inválida.")
    return 8760.0 / hours


def _merge_all(price_df: pd.DataFrame,
               pv_df: pd.DataFrame | None,
               load_df: pd.DataFrame | None) -> pd.DataFrame:
    if "datetime" not in price_df or "price_EUR_per_MWh" not in price_df:
        raise ValueError("Preço: colunas necessárias 'datetime, price_EUR_per_MWh'.")

    p = price_df.copy()
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p.sort_values("datetime").reset_index(drop=True)

    df = p.copy()
    if pv_df is not None:
        g = pv_df.copy()
        if "datetime" not in g or "pv_MW" not in g:
            raise ValueError("PV: colunas necessárias 'datetime, pv_MW'.")
        g["datetime"] = pd.to_datetime(g["datetime"])
        g = g.sort_values("datetime").reset_index(drop=True)
        df = pd.merge_asof(df, g, on="datetime")
    else:
        df["pv_MW"] = 0.0

    if load_df is not None:
        l = load_df.copy()
        if "datetime" not in l or "load_MW" not in l:
            raise ValueError("Carga: colunas necessárias 'datetime, load_MW'.")
        l["datetime"] = pd.to_datetime(l["datetime"])
        l = l.sort_values("datetime").reset_index(drop=True)
        df = pd.merge_asof(df, l, on="datetime")
    else:
        df["load_MW"] = 0.0

    df[["pv_MW", "load_MW"]] = df[["pv_MW", "load_MW"]].fillna(0.0)
    return df


# ==========================================
# Baselines (sem BESS) para relatórios
# ==========================================
def baselines_mw(price_df, pv_df=None, load_df=None,
                 import_fee=0.0, export_fee=0.0) -> dict:
    """
    Tudo em MW/15s ⇒ convertemos para MWh/step com dt_h.
    - Custo consumo: (preço + import_fee) * load
    - Receita só solar: (preço - export_fee) * pv  (toda PV exportada)
    """
    df = _merge_all(price_df, pv_df, load_df)
    dt_h = _infer_dt_hours(df["datetime"])
    af = _annual_factor(len(df), dt_h)

    price = df["price_EUR_per_MWh"].to_numpy(float)
    load_MWh = df["load_MW"].to_numpy(float) * dt_h
    pv_MWh = df["pv_MW"].to_numpy(float) * dt_h

    cost_consumption_series = ((price + import_fee) * load_MWh).sum()
    revenue_solar_only_series = ((price - export_fee) * pv_MWh).sum()

    return {
        "dt_hours": dt_h,
        "annual_factor": af,
        "Cost_consumption_annual_EUR": af * cost_consumption_series,
        "Revenue_solar_only_annual_EUR": af * revenue_solar_only_series,
    }


# ==========================================
# MILP (em MW → MWh/step) – cenário geral
# ==========================================
def run_site_bess_mw(price_df: pd.DataFrame,
                     pv_df: pd.DataFrame | None,
                     load_df: pd.DataFrame | None,
                     params: dict,
                     P_bess_MW: float,
                     c_rate_per_hour: float,
                     return_schedule: bool = True) -> dict:
    """
    Modelo único que cobre:
      - arbitragem pura (sem PV, sem carga)
      - consumo + arbitragem (sem PV)
      - PV + arbitragem (sem carga)
      - PV + consumo + arbitragem

    Entradas:
      • P_bess_MW (potência nominal)
      • c_rate (1/h). Energia nominal: E_cap_MWh = P / c
      • Eficiências, SoC min/max, limites de import/export, tarifas import/export.

    Variáveis por passo (MWh no passo):
      c_grid, c_pv, d_grid, d_load, pv_load, pv_export, grid_load, soc.
      Exclusividade carga/descarga via binária y_t e Big-M = P*dt.
    """
    df = _merge_all(price_df, pv_df, load_df)
    dt = _infer_dt_hours(df["datetime"])
    af = _annual_factor(len(df), dt)
    T = len(df)

    price = df["price_EUR_per_MWh"].to_numpy(float)
    pv_MWh = df["pv_MW"].to_numpy(float) * dt
    load_MWh = df["load_MW"].to_numpy(float) * dt

    # Parâmetros
    P_cap = float(P_bess_MW)
    c_rate = max(1e-6, float(c_rate_per_hour))
    E_cap = P_cap / c_rate  # MWh
    eta_c = float(params.get("eta_charge", 0.95))
    eta_d = float(params.get("eta_discharge", 0.95))
    soc_min = float(params.get("soc_min", 0.0)) * E_cap
    soc_max = float(params.get("soc_max", 1.0)) * E_cap

    import_fee = float(params.get("import_fee_eur_per_MWh", 0.0))
    export_fee = float(params.get("export_fee_eur_per_MWh", 0.0))
    allow_grid_charging = bool(params.get("allow_grid_charging", True))

    imp_cap_MWh = float(params.get("P_grid_import_max", 1e9)) * dt
    exp_cap_MWh = float(params.get("P_grid_export_max", 1e9)) * dt
    M_step = max(1e-6, P_cap * dt)

    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0))
    opex_fix_gen = float(params.get("opex_fix_gen", 0.0))
    opex_fix_bess = float(params.get("opex_fix_bess", 0.0))
    opex_var_trade = float(params.get("opex_var_trade_eur_per_MWh", 0.0))
    opex_var_gen = float(params.get("opex_var_gen_eur_per_mwh", 0.0))

    # ---------------- MILP -----------------
    prob = pulp.LpProblem("BESS_MW_MILP", pulp.LpMaximize)

    # Variáveis (todas MWh por passo; soc em MWh)
    c_grid   = pulp.LpVariable.dicts("c_grid",   range(T), lowBound=0)  # carga da rede
    c_pv     = pulp.LpVariable.dicts("c_pv",     range(T), lowBound=0)  # carga do PV
    d_grid   = pulp.LpVariable.dicts("d_grid",   range(T), lowBound=0)  # descarga para rede
    d_load   = pulp.LpVariable.dicts("d_load",   range(T), lowBound=0)  # descarga para carga local
    pv_load  = pulp.LpVariable.dicts("pv_load",  range(T), lowBound=0)  # PV → carga
    pv_exp   = pulp.LpVariable.dicts("pv_exp",   range(T), lowBound=0)  # PV → rede
    g_load   = pulp.LpVariable.dicts("g_load",   range(T), lowBound=0)  # rede → carga
    soc      = pulp.LpVariable.dicts("soc",      range(T), lowBound=soc_min, upBound=soc_max)
    y        = pulp.LpVariable.dicts("y_charging", range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Restrições por passo
    for t in range(T):
        # Balanço PV
        prob += pv_load[t] + c_pv[t] + pv_exp[t] == pv_MWh[t], f"pv_balance_{t}"
        # Balanço carga
        prob += g_load[t] + pv_load[t] + d_load[t] == load_MWh[t], f"load_balance_{t}"
        # Potência de carga/descarga (P*dt)
        prob += c_grid[t] + c_pv[t] <= M_step, f"charge_power_{t}"
        prob += d_grid[t] + d_load[t] <= M_step, f"discharge_power_{t}"
        # Exclusividade (Big-M)
        prob += c_grid[t] + c_pv[t] <= M_step * y[t], f"switch_c_{t}"
        prob += d_grid[t] + d_load[t] <= M_step * (1 - y[t]), f"switch_d_{t}"
        # Limites de rede
        prob += g_load[t] + c_grid[t] <= imp_cap_MWh, f"import_cap_{t}"
        prob += pv_exp[t] + d_grid[t] <= exp_cap_MWh, f"export_cap_{t}"
        # Proibir carga pela rede se não permitido
        if not allow_grid_charging:
            prob += c_grid[t] == 0, f"no_grid_charge_{t}"
        # Dinâmica SOC
        prev = T - 1 if t == 0 else t - 1
        prob += soc[t] == soc[prev] + eta_c * (c_grid[t] + c_pv[t]) - (d_grid[t] + d_load[t]) / eta_d, f"soc_dyn_{t}"

    # Objetivo: receita de export + economia de autoconsumo − custo import para carga − degradação
    # Receita export:
    revenue_export = pulp.lpSum([(price[t] - export_fee) * (pv_exp[t] + d_grid[t]) for t in range(T)])
    # Economia autoconsumo: evita pagar (preço + import_fee) na parte atendida por PV ou BESS
    savings_self = pulp.lpSum([(price[t] + import_fee) * (pv_load[t] + d_load[t]) for t in range(T)])
    # Custo de importar para carregar bateria
    cost_charge_grid = pulp.lpSum([(price[t] + import_fee) * c_grid[t] for t in range(T)])
    # Custo de degradação
    throughput = pulp.lpSum([d_grid[t] + d_load[t] for t in range(T)])
    deg_series_cost = deg_cost * throughput

    gross_margin_series = revenue_export + savings_self - cost_charge_grid - deg_series_cost

    prob += gross_margin_series
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(params.get("solver_time_limit_s", 120)))
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    # Extrair série
    val = lambda expr: float(pulp.value(expr))
    revenue_series = val(revenue_export)
    savings_series = val(savings_self)
    cost_charge_series = val(cost_charge_grid)
    throughput_series = val(throughput)
    margin_series = val(gross_margin_series)

    # Anualização
    revenue_annual = af * revenue_series
    savings_annual = af * savings_series
    cost_charge_annual = af * cost_charge_series
    throughput_annual = af * throughput_series
    gross_margin_annual = af * margin_series

    # Custos fixos e variáveis anuais
    bess_capex = bess_capex_eur(E_cap, P_cap,
                                float(params.get("c_E_capex", 250.0)),
                                float(params.get("c_P_capex", 150.0)))
    bess_ann_cost = annualized_cost(bess_capex, float(params.get("discount_rate", 8.0)),
                                    int(params.get("lifetime_years", 15)))
    opex_var_trade_annual = float(opex_var_trade) * (af * pulp.value(pulp.lpSum([g_load[t] + pv_exp[t] + d_grid[t] for t in range(T)])))
    opex_var_gen_annual = float(opex_var_gen) * (af * pulp.value(pulp.lpSum([pv_load[t]] for _ in [0])) if load_df is not None else 0.0)

    ebitda_annual = gross_margin_annual - bess_ann_cost - opex_fix_bess - opex_fix_gen - opex_var_trade_annual - opex_var_gen_annual

    # LCOE total de energia **entregue** (para referência)
    energy_delivered_annual = af * pulp.value(pulp.lpSum([d_grid[t] + d_load[t] + pv_load[t] + pv_exp[t] for t in range(T)]))
    lcoe_total = lcoe_annual(
        capex_total_eur=float(params.get("capex_gen", 0.0)) + bess_capex,
        rate_percent=float(params.get("discount_rate", 8.0)),
        n_years=int(params.get("lifetime_years", 15)),
        energy_annual_MWh=max(1e-9, energy_delivered_annual),
        opex_fix_eur_per_year=opex_fix_gen + opex_fix_bess,
        opex_var_eur_per_MWh=opex_var_gen + opex_var_trade,
    )

    result = {
        "status_text": status,
        "dt_hours": dt,
        "annual_factor": af,
        "P_cap_MW": P_cap,
        "E_cap_MWh": E_cap,
        "Revenue_export_annual_EUR": revenue_annual,
        "Savings_selfcons_annual_EUR": savings_annual,
        "Cost_charge_grid_annual_EUR": cost_charge_annual,
        "Throughput_annual_MWh": throughput_annual,
        "Gross_margin_annual_EUR": gross_margin_annual,
        "BESS_CAPEX_EUR": bess_capex,
        "BESS_annual_cost_EUR": bess_ann_cost,
        "EBITDA_project_annual_EUR": ebitda_annual,
        "LCOE_total_EUR_per_MWh": lcoe_total,
    }

    if return_schedule:
        data = {
            "datetime": df["datetime"],
            "price_EUR_per_MWh": price,
            "pv_MWh": pv_MWh,
            "load_MWh": load_MWh,
            "c_grid_MWh": [pulp.value(c_grid[t]) for t in range(T)],
            "c_pv_MWh": [pulp.value(c_pv[t]) for t in range(T)],
            "d_grid_MWh": [pulp.value(d_grid[t]) for t in range(T)],
            "d_load_MWh": [pulp.value(d_load[t]) for t in range(T)],
            "pv_load_MWh": [pulp.value(pv_load[t]) for t in range(T)],
            "pv_exp_MWh": [pulp.value(pv_exp[t]) for t in range(T)],
            "g_load_MWh": [pulp.value(g_load[t]) for t in range(T)],
            "soc_MWh": [pulp.value(soc[t]) for t in range(T)],
        }
        result["schedule"] = pd.DataFrame(data)

    # ROI e Payback (simples)
    capex = bess_capex
    ebitda = ebitda_annual
    result["ROI_annual_%"] = 100.0 * (ebitda / capex) if capex > 0 else None
    result["Payback_years"] = (capex / ebitda) if ebitda > 1e-9 else None

    return result


# ==========================================
# Otimização por grade (P em MW, C-rate)
# ==========================================
def optimize_site_bess_mw(price_df, pv_df, load_df, params,
                           P_values_MW, C_values_per_hour,
                           objective="ROI") -> dict:
    """
    Faz uma varredura leve e escolhe o par (P, C) que maximiza:
      • objective = "ROI" (padrão)  OU  "EBITDA"
    """
    best = None
    for P in P_values_MW:
        for C in C_values_per_hour:
            res = run_site_bess_mw(price_df, pv_df, load_df, params, P, C, return_schedule=False)
            score = (res["ROI_annual_%"] if objective == "ROI" else res["EBITDA_project_annual_EUR"])
            if (best is None) or (score is not None and score > (best["score"] if best else -1e30)):
                best = {"score": score, "res": res, "P": P, "C": C}
    # roda de novo para pegar o schedule
    final = run_site_bess_mw(price_df, pv_df, load_df, params, best["P"], best["C"], return_schedule=True)
    final["status_text"] = f"Melhor par encontrado: P={best['P']} MW, C={best['C']}  (objetivo: {objective})"
    return final
