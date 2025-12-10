import numpy as np
import pandas as pd
import pulp

from core.finance import (
    crf,
    bess_capex_eur,
    annualized_cost,
    lcoe_annual,
)

# =========================================================
# Helpers de preparação
# =========================================================
def _infer_dt_hours(df_time: pd.Series) -> float:
    if len(df_time) < 2:
        raise ValueError("Série temporal muito curta para inferir Δt.")
    dt = (pd.to_datetime(df_time.iloc[1]) - pd.to_datetime(df_time.iloc[0])).total_seconds() / 3600.0
    if dt <= 0:
        raise ValueError("Δt inválido.")
    return dt


def _annual_factor(n_steps: int, dt_hours: float) -> float:
    hours = n_steps * dt_hours
    if hours <= 0:
        raise ValueError("Duração da série inválida.")
    return 8760.0 / hours


def _prepare_generation(gen_df: pd.DataFrame,
                        availability_pct: float,
                        P_grid_export_max_MW: float,
                        dt_hours: float) -> pd.DataFrame:
    """
    Espera colunas: datetime, gen_MWh
    Aplica disponibilidade (escala) e cria coluna de limite de exportação (MWh/intervalo).
    """
    if "datetime" not in gen_df.columns or "gen_MWh" not in gen_df.columns:
        raise ValueError("gen_df deve conter colunas 'datetime' e 'gen_MWh'.")
    df = gen_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    avail = max(0.0, min(100.0, availability_pct)) / 100.0
    df["gen_MWh"] = df["gen_MWh"] * avail

    export_cap_MWh = (P_grid_export_max_MW if P_grid_export_max_MW > 0 else 1e12) * dt_hours
    df["gen_export_cap_MWh"] = export_cap_MWh
    return df


def _merge_prices_generation(price_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in price_df.columns or "price_EUR_per_MWh" not in price_df.columns:
        raise ValueError("price_df deve conter colunas 'datetime' e 'price_EUR_per_MWh'.")
    p = price_df.copy()
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p.sort_values("datetime").reset_index(drop=True)
    g = gen_df.copy()
    g["datetime"] = pd.to_datetime(g["datetime"])
    g = g.sort_values("datetime").reset_index(drop=True)
    df = pd.merge_asof(g, p, on="datetime")
    return df

# =========================================================
# Baseline (sem BESS)
# =========================================================
def run_baseline(price_df: pd.DataFrame,
                 gen_df: pd.DataFrame,
                 params: dict) -> dict:
    """
    Calcula baseline sem BESS: energia anual, receita anual e LCOE base.
    Params requer:
      - capex_gen (EUR)
      - opex_fix_gen (EUR/ano)
      - opex_var_eur_per_mwh
      - discount_rate (%)
      - lifetime_years
      - availability_pct (%)
      - P_grid_export_max (MW)
    """
    dt_hours = _infer_dt_hours(gen_df["datetime"])
    af = _annual_factor(len(gen_df), dt_hours)

    gen_prepared = _prepare_generation(
        gen_df, params.get("availability_pct", 100.0),
        params.get("P_grid_export_max", 0.0),
        dt_hours
    )
    df = _merge_prices_generation(price_df, gen_prepared)

    # Sem BESS: venda direta limitada pela exportação (spill = excesso)
    sold_direct = np.minimum(df["gen_MWh"].values, df["gen_export_cap_MWh"].values)
    spill = df["gen_MWh"].values - sold_direct
    df["sold_direct_MWh"] = sold_direct
    df["spill_MWh"] = spill.clip(min=0.0)

    E_series = sold_direct.sum()
    E_annual = af * E_series
    revenue_series = (df["price_EUR_per_MWh"].values * sold_direct).sum()
    revenue_annual = af * revenue_series

    lcoe_base = lcoe_annual(
        capex_total_eur=params["capex_gen"],
        rate_percent=params["discount_rate"],
        n_years=int(params["lifetime_years"]),
        energy_annual_MWh=E_annual,
        opex_fix_eur_per_year=params.get("opex_fix_gen", 0.0),
        opex_var_eur_per_MWh=params.get("opex_var_eur_per_mwh", 0.0),
    )

    return {
        "E_series_MWh": float(E_series),
        "E_annual_MWh": float(E_annual),
        "Revenue_series_EUR": float(revenue_series),
        "Revenue_annual_EUR": float(revenue_annual),
        "LCOE_base_EUR_per_MWh": float(lcoe_base),
        "schedule_baseline": df[["datetime", "gen_MWh", "sold_direct_MWh", "spill_MWh", "price_EUR_per_MWh"]],
        "dt_hours": dt_hours,
        "annual_factor": af,
    }

# =========================================================
# Otimização com BESS (MILP) – Big-M linearization
# =========================================================
def run_with_bess(price_df: pd.DataFrame,
                  gen_df: pd.DataFrame,
                  params: dict) -> dict:
    """
    Maximiza o EBITDA anual com MILP:
      - Proíbe carga e descarga simultâneas via Big-M (sem produto de variáveis).
      - Permite carga da rede (opcional) com limite de importação.
      - Degradação via custo por throughput (€/MWh_throughput).
      - Limite de ciclos/ano: throughput_annual <= 2 * E_cap * cycles_per_year_max
    """
    # --- Preparação ---
    dt = _infer_dt_hours(gen_df["datetime"])
    af = _annual_factor(len(gen_df), dt)

    gen_prep = _prepare_generation(
        gen_df,
        params.get("availability_pct", 100.0),
        params.get("P_grid_export_max", 0.0),
        dt
    )
    df = _merge_prices_generation(price_df, gen_prep)
    T = len(df)

    # Limites de rede (MWh/step)
    exp_cap = (params.get("P_grid_export_max", 0.0) if params.get("P_grid_export_max", 0.0) > 0 else 1e12) * dt
    imp_cap = (params.get("P_grid_import_max", 0.0) if params.get("P_grid_import_max", 0.0) > 0 else 1e12) * dt

    # Big-M (constante por passo) para alternância carga/descarga
    g_max = float(df["gen_MWh"].max()) if T > 0 else 0.0
    M_step = max(exp_cap, imp_cap, g_max + imp_cap) + 1.0

    # Parâmetros do BESS
    eta_c = float(params.get("eta_charge", 0.95))
    eta_d = float(params.get("eta_discharge", 0.95))
    allow_grid = bool(params.get("allow_grid_charging", True))
    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0))
    cycles_max = float(params.get("cycles_per_year_max", 0.0))  # 0 => sem limite

    # Variáveis globais (dimensionamento)
    E_cap = pulp.LpVariable("E_cap_MWh", lowBound=0)
    P_cap = pulp.LpVariable("P_cap_MW", lowBound=0)

    # Variáveis por período
    soc = pulp.LpVariable.dicts("soc", range(T), lowBound=0)
    c_ren = pulp.LpVariable.dicts("charge_from_ren", range(T), lowBound=0)
    c_grid = pulp.LpVariable.dicts("charge_from_grid", range(T), lowBound=0)
    d = pulp.LpVariable.dicts("discharge", range(T), lowBound=0)
    s_dir = pulp.LpVariable.dicts("sold_from_gen", range(T), lowBound=0)
    spill = pulp.LpVariable.dicts("spill", range(T), lowBound=0)
    y = pulp.LpVariable.dicts("is_charging_mode", range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)

    prob = pulp.LpProblem("BESS_Arbitrage_MILP", pulp.LpMaximize)

    # Restrições físicas
    for t in range(T):
        g_t = float(df.loc[t, "gen_MWh"])

        # Balanço da geração
        prob += s_dir[t] + c_ren[t] + spill[t] == g_t, f"gen_balance_{t}"

        # --- BIG-M linearization ---
        # Limites de potência do BESS por passo (sem binária)
        prob += c_ren[t] + c_grid[t] <= P_cap * dt, f"charge_cap_{t}"
        prob += d[t] <= P_cap * dt, f"discharge_cap_{t}"
        # Alternância carga/descarga com Big-M (sem produto de variáveis)
        prob += c_ren[t] + c_grid[t] <= M_step * y[t], f"charge_switch_{t}"
        prob += d[t] <= M_step * (1 - y[t]), f"discharge_switch_{t}"

        # Limites de rede
        prob += s_dir[t] + d[t] <= exp_cap, f"export_cap_{t}"
        if not allow_grid:
            prob += c_grid[t] == 0, f"no_grid_{t}"
        else:
            prob += c_grid[t] <= imp_cap, f"import_cap_{t}"

        # Capacidade de energia
        prob += soc[t] <= E_cap, f"soc_cap_{t}"

        # Dinâmica do SOC (cíclica)
        prev = T - 1 if t == 0 else t - 1
        prob += soc[t] == soc[prev] + eta_c * (c_ren[t] + c_grid[t]) - d[t] / eta_d, f"soc_dyn_{t}"

    # Termos econômicos (anualizados)
    price = df["price_EUR_per_MWh"].values
    revenue_series = pulp.lpSum([price[t] * (s_dir[t] + d[t]) for t in range(T)])
    grid_cost_series = pulp.lpSum([price[t] * c_grid[t] for t in range(T)])
    energy_series = pulp.lpSum([s_dir[t] + d[t] for t in range(T)])

    revenue_annual = _annual_factor(T, dt) * revenue_series
    grid_cost_annual = _annual_factor(T, dt) * grid_cost_series
    energy_annual = _annual_factor(T, dt) * energy_series

    # Throughput anual aproximado (carga + descarga)
    throughput_series = pulp.lpSum([(c_ren[t] + c_grid[t]) + d[t] for t in range(T)])
    throughput_annual = _annual_factor(T, dt) * throughput_series

    # Limite de ciclos/ano
    if cycles_max > 0:
        prob += throughput_annual <= 2.0 * E_cap * cycles_max, "cycles_per_year_limit"

    # Custos do BESS
    bess_capex_expr = 1000.0 * params["c_E_capex"] * E_cap + 1000.0 * params["c_P_capex"] * P_cap
    bess_annual_cost = crf(params["discount_rate"], int(params["lifetime_years"])) * bess_capex_expr

    # OPEX
    opex_var = float(params.get("opex_var_eur_per_mwh", 0.0))
    opex_var_annual = opex_var * energy_annual
    opex_fix_bess = float(params.get("opex_fix_bess", 0.0))

    # Degradação (€/MWh_throughput)
    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0))
    deg_cost_annual = deg_cost * throughput_annual

    # EBITDA anual
    ebitda_annual = revenue_annual - grid_cost_annual - bess_annual_cost - opex_fix_bess - opex_var_annual - deg_cost_annual

    # Objetivo
    prob += ebitda_annual

    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {
            "status_text": f"Modelo não encontrou solução ótima (status: {status})",
            "schedule": df,
        }

    # Extrair resultados
    E_cap_opt = E_cap.value()
    P_cap_opt = P_cap.value()

    df_out = df.copy()
    df_out["soc_MWh"] = [soc[t].value() for t in range(T)]
    df_out["charge_from_ren_MWh"] = [c_ren[t].value() for t in range(T)]
    df_out["charge_from_grid_MWh"] = [c_grid[t].value() for t in range(T)]
    df_out["discharge_MWh"] = [d[t].value() for t in range(T)]
    df_out["sold_from_gen_MWh"] = [s_dir[t].value() for t in range(T)]
    df_out["spill_MWh"] = [spill[t].value() for t in range(T)]
    df_out["is_charging_mode"] = [y[t].value() for t in range(T)]

    revenue_val = pulp.value(revenue_annual)
    grid_cost_val = pulp.value(grid_cost_annual)
    energy_annual_val = pulp.value(energy_annual)
    throughput_annual_val = pulp.value(throughput_annual)
    deg_cost_annual_val = pulp.value(deg_cost_annual)

    bess_capex_val = bess_capex_eur(E_cap_opt, P_cap_opt, params["c_E_capex"], params["c_P_capex"])
    bess_annual_cost_val = annualized_cost(bess_capex_val, params["discount_rate"], int(params["lifetime_years"]))
    ebitda_val = pulp.value(ebitda_annual)

    # LCOE com BESS (inclui CAPEX da usina + BESS)
    lcoe_with_bess = lcoe_annual(
        capex_total_eur=params["capex_gen"] + bess_capex_val,
        rate_percent=params["discount_rate"],
        n_years=int(params["lifetime_years"]),
        energy_annual_MWh=energy_annual_val,
        opex_fix_eur_per_year=float(params.get("opex_fix_gen", 0.0)) + float(params.get("opex_fix_bess", 0.0)),
        opex_var_eur_per_MWh=opex_var,
    )

    return {
        "status_text": "Solução ótima encontrada (MILP)",
        "E_cap_opt_MWh": float(E_cap_opt),
        "P_cap_opt_MW": float(P_cap_opt),
        "Energy_annual_MWh": float(energy_annual_val),
        "Throughput_annual_MWh": float(throughput_annual_val),
        "Deg_cost_annual_EUR": float(deg_cost_annual_val),
        "Revenue_annual_EUR": float(revenue_val),
        "Grid_energy_cost_annual_EUR": float(grid_cost_val),
        "BESS_CAPEX_EUR": float(bess_capex_val),
        "BESS_annual_cost_EUR": float(bess_annual_cost_val),
        "EBITDA_annual_EUR": float(ebitda_val),
        "LCOE_with_BESS_EUR_per_MWh": float(lcoe_with_bess),
        "schedule": df_out,
    }

# =========================================================
# Sensibilidades (±10/±20%) em c_E e c_P
# =========================================================
def run_sensitivities(price_df: pd.DataFrame,
                      gen_df: pd.DataFrame,
                      base_params: dict) -> pd.DataFrame:
    factors = [-0.2, -0.1, 0.0, 0.1, 0.2]
    rows = []
    for fE in factors:
        for fP in factors:
            params = dict(base_params)
            params["c_E_capex"] = base_params["c_E_capex"] * (1.0 + fE)
            params["c_P_capex"] = base_params["c_P_capex"] * (1.0 + fP)
            res = run_with_bess(price_df, gen_df, params)
            if res.get("status_text", "").startswith("Solução"):
                rows.append({
                    "delta_cE_%": int(fE * 100),
                    "delta_cP_%": int(fP * 100),
                    "E_cap_MWh": res["E_cap_opt_MWh"],
                    "P_cap_MW": res["P_cap_opt_MW"],
                    "EBITDA_annual_EUR": res["EBITDA_annual_EUR"],
                    "LCOE_with_BESS": res["LCOE_with_BESS_EUR_per_MWh"],
                })
    return pd.DataFrame(rows)

# =========================================================
# Lote de cenários (P50/P90 etc.)
# =========================================================
def run_batch_scenarios(price_dfs, gen_dfs, labels, params):
    results = []
    rows = []
    for (p, g, name) in zip(price_dfs, gen_dfs, labels):
        base = run_baseline(p, g, params)
        withb = run_with_bess(p, g, params)
        results.append({"label": name, "baseline": base, "with_bess": withb})

        if withb.get("status_text", "").startswith("Solução"):
            rows.append({
                "Cenário": name,
                "E_cap_MWh": withb["E_cap_opt_MWh"],
                "P_cap_MW": withb["P_cap_opt_MW"],
                "EBITDA_EUR_ano": withb["EBITDA_annual_EUR"],
                "LCOE_base": base["LCOE_base_EUR_per_MWh"],
                "LCOE_com_BESS": withb["LCOE_with_BESS_EUR_per_MWh"],
                "Receita_base_EUR_ano": base["Revenue_annual_EUR"],
                "Receita_BESS_EUR_ano": withb["Revenue_annual_EUR"],
                "Throughput_MWh_ano": withb["Throughput_annual_MWh"],
            })
        else:
            rows.append({
                "Cenário": name,
                "E_cap_MWh": None,
                "P_cap_MW": None,
                "EBITDA_EUR_ano": None,
                "LCOE_base": base["LCOE_base_EUR_per_MWh"],
                "LCOE_com_BESS": None,
                "Receita_base_EUR_ano": base["Revenue_annual_EUR"],
                "Receita_BESS_EUR_ano": None,
                "Throughput_MWh_ano": None,
            })
    return results, pd.DataFrame(rows)
