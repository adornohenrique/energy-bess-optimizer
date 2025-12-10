# core/optimization.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import pulp
from core.finance import annualized_cost, lcoe_annual, bess_capex_eur


def _infer_dt_hours(dt_series: pd.Series) -> float:
    s = pd.to_datetime(dt_series, utc=True)
    if len(s) < 2:
        raise ValueError("Série temporal curta.")
    dt_h = (s.iloc[1] - s.iloc[0]).total_seconds() / 3600.0
    if dt_h <= 0:
        raise ValueError("Δt inválido.")
    return dt_h

def _annual_factor(n_steps: int, dt_h: float) -> float:
    total_h = n_steps * dt_h
    if total_h <= 0:
        raise ValueError("Duração inválida.")
    return 8760.0 / total_h

def _merge_all(price_df: pd.DataFrame,
               pv_df: Optional[pd.DataFrame],
               load_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if "datetime" not in price_df or "price_EUR_per_MWh" not in price_df:
        raise ValueError("Preços: 'datetime, price_EUR_per_MWh'.")
    df = price_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if pv_df is not None:
        pv = pv_df.copy()
        if "datetime" not in pv or "pv_MW" not in pv:
            raise ValueError("PV: 'datetime, pv_MW'.")
        pv["datetime"] = pd.to_datetime(pv["datetime"], utc=True)
        pv = pv.sort_values("datetime").reset_index(drop=True)
        df = pd.merge_asof(df, pv, on="datetime")
    else:
        df["pv_MW"] = 0.0
    if load_df is not None:
        ld = load_df.copy()
        if "datetime" not in ld or "load_MW" not in ld:
            raise ValueError("Carga: 'datetime, load_MW'.")
        ld["datetime"] = pd.to_datetime(ld["datetime"], utc=True)
        ld = ld.sort_values("datetime").reset_index(drop=True)
        df = pd.merge_asof(df, ld, on="datetime")
    else:
        df["load_MW"] = 0.0
    df[["pv_MW", "load_MW"]] = df[["pv_MW", "load_MW"]].fillna(0.0)
    return df


def baselines_mw(price_df: pd.DataFrame,
                 pv_df: Optional[pd.DataFrame] = None,
                 load_df: Optional[pd.DataFrame] = None,
                 import_fee_const: float = 0.0,
                 export_fee_const: float = 0.0,
                 import_fee_series: Optional[np.ndarray] = None,
                 export_fee_series: Optional[np.ndarray] = None) -> dict:
    df = _merge_all(price_df, pv_df, load_df)
    dt = _infer_dt_hours(df["datetime"])
    af = _annual_factor(len(df), dt)

    price = df["price_EUR_per_MWh"].to_numpy(float)
    pv_MWh = df["pv_MW"].to_numpy(float) * dt
    load_MWh = df["load_MW"].to_numpy(float) * dt

    if import_fee_series is not None and len(import_fee_series) == len(df):
        imp_fee = np.asarray(import_fee_series, dtype=float)
    else:
        imp_fee = np.full(len(df), float(import_fee_const))
    if export_fee_series is not None and len(export_fee_series) == len(df):
        exp_fee = np.asarray(export_fee_series, dtype=float)
    else:
        exp_fee = np.full(len(df), float(export_fee_const))

    cost_consumption_series = np.sum((price + imp_fee) * load_MWh)
    revenue_solar_only_series = np.sum((price - exp_fee) * pv_MWh)

    return {
        "dt_hours": dt,
        "annual_factor": af,
        "Cost_consumption_annual_EUR": af * float(cost_consumption_series),
        "Revenue_solar_only_annual_EUR": af * float(revenue_solar_only_series),
    }


def run_site_bess_mw(price_df: pd.DataFrame,
                     pv_df: Optional[pd.DataFrame],
                     load_df: Optional[pd.DataFrame],
                     params: dict,
                     P_bess_MW: float,
                     c_rate_per_hour: float,
                     return_schedule: bool = True) -> dict:

    df = _merge_all(price_df, pv_df, load_df)
    dt = _infer_dt_hours(df["datetime"])
    af = _annual_factor(len(df), dt)
    T = len(df)

    price = df["price_EUR_per_MWh"].to_numpy(float)
    pv_MWh = df["pv_MW"].to_numpy(float) * dt
    load_MWh = df["load_MW"].to_numpy(float) * dt

    # séries de tarifas (por passo)
    if params.get("import_fee_series") is not None and len(params["import_fee_series"]) == T:
        imp_fee = np.asarray(params["import_fee_series"], dtype=float)
    else:
        imp_fee = np.full(T, float(params.get("import_fee_eur_per_MWh", 0.0)))
    if params.get("export_fee_series") is not None and len(params["export_fee_series"]) == T:
        exp_fee = np.asarray(params["export_fee_series"], dtype=float)
    else:
        exp_fee = np.full(T, float(params.get("export_fee_eur_per_MWh", 0.0)))

    # BESS
    P_cap = float(P_bess_MW)
    c_rate = max(1e-6, float(c_rate_per_hour))
    E_cap = P_cap / c_rate

    # Eficiências: η_charge_total = η_charge * η_AC→DC ; η_discharge_total = η_discharge * η_DC→AC
    eta_c  = float(params.get("eta_charge", 0.95))
    eta_d  = float(params.get("eta_discharge", 0.95))
    eta_ac2dc = float(params.get("eta_ac2dc", 1.0))
    eta_dc2ac = float(params.get("eta_dc2ac", 1.0))
    eta_charge_total    = eta_c  * eta_ac2dc
    eta_discharge_total = eta_d  * eta_dc2ac

    soc_min_frac = float(params.get("soc_min", 0.0))
    soc_max_frac = float(params.get("soc_max", 1.0))
    soc_init_frac = float(params.get("soc_init", 0.5))
    soc_final_min_frac = float(params.get("soc_final_min", 0.0))
    enforce_equal_terminal = bool(params.get("enforce_terminal_equals_init", False))

    soc_min = soc_min_frac * E_cap
    soc_max = soc_max_frac * E_cap
    soc_init = soc_init_frac * E_cap
    soc_final_min = soc_final_min_frac * E_cap

    # Rede e solver
    allow_grid_charging = bool(params.get("allow_grid_charging", True))
    imp_cap = float(params.get("P_grid_import_max", 1e12)) * dt
    exp_cap = float(params.get("P_grid_export_max", 1e12)) * dt
    M_step = max(1e-6, P_cap * dt)

    # Custos
    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0))
    opex_fix_bess = float(params.get("opex_fix_bess", 0.0))
    opex_fix_gen  = float(params.get("opex_fix_gen", 0.0))
    opex_var_trade = float(params.get("opex_var_trade_eur_per_MWh", 0.0))
    opex_var_gen   = float(params.get("opex_var_gen_eur_per_mwh", 0.0))

    discount = float(params.get("discount_rate", 8.0))
    lifetime = int(params.get("lifetime_years", 15))
    c_E = float(params.get("c_E_capex", 250.0))
    c_P = float(params.get("c_P_capex", 150.0))
    capex_gen = float(params.get("capex_gen", 0.0))

    # Modelo
    prob = pulp.LpProblem("BESS_MW", pulp.LpMaximize)

    c_grid  = pulp.LpVariable.dicts("c_grid",  range(T), lowBound=0)  # carga via rede
    c_pv    = pulp.LpVariable.dicts("c_pv",    range(T), lowBound=0)  # carga via PV
    d_grid  = pulp.LpVariable.dicts("d_grid",  range(T), lowBound=0)  # descarga p/ rede
    d_load  = pulp.LpVariable.dicts("d_load",  range(T), lowBound=0)  # descarga p/ carga
    pv_load = pulp.LpVariable.dicts("pv_load", range(T), lowBound=0)  # PV direto p/ carga
    pv_exp  = pulp.LpVariable.dicts("pv_exp",  range(T), lowBound=0)  # PV p/ rede
    g_load  = pulp.LpVariable.dicts("g_load",  range(T), lowBound=0)  # rede p/ carga
    soc     = pulp.LpVariable.dicts("soc",     range(T), lowBound=soc_min, upBound=soc_max)
    y       = pulp.LpVariable.dicts("y_chg",   range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Inicial
    prob += soc[0] == soc_init + eta_charge_total*(c_grid[0] + c_pv[0]) - (d_grid[0] + d_load[0])/eta_discharge_total, "soc_init_step"

    for t in range(T):
        # Balanços
        prob += pv_load[t] + c_pv[t] + pv_exp[t] == pv_MWh[t], f"pv_bal_{t}"
        prob += g_load[t] + pv_load[t] + d_load[t] == load_MWh[t], f"load_bal_{t}"

        # Potência por passo (MWh no passo)
        prob += c_grid[t] + c_pv[t] <= M_step, f"lim_charge_{t}"
        prob += d_grid[t] + d_load[t] <= M_step, f"lim_discharge_{t}"

        # Exclusividade
        prob += c_grid[t] + c_pv[t] <= M_step * y[t],         f"switch_c_{t}"
        prob += d_grid[t] + d_load[t] <= M_step * (1 - y[t]), f"switch_d_{t}"

        # Limites de rede
        prob += g_load[t] + c_grid[t] <= imp_cap, f"imp_cap_{t}"
        prob += pv_exp[t] + d_grid[t] <= exp_cap, f"exp_cap_{t}"
        if not allow_grid_charging:
            prob += c_grid[t] == 0,               f"no_grid_charge_{t}"

        # Dinâmica SOC (a partir do 2º passo)
        if t >= 1:
            prob += soc[t] == soc[t-1] + eta_charge_total*(c_grid[t] + c_pv[t]) - (d_grid[t] + d_load[t])/eta_discharge_total, f"soc_{t}"

    # Terminal
    if enforce_equal_terminal:
        prob += soc[T-1] == soc_init, "terminal_equal_init"
    else:
        prob += soc[T-1] >= soc_final_min, "terminal_min"

    # Objetivo
    revenue_export = pulp.lpSum([(price[t] - exp_fee[t]) * (pv_exp[t] + d_grid[t]) for t in range(T)])
    savings_self   = pulp.lpSum([(price[t] + imp_fee[t]) * (pv_load[t] + d_load[t]) for t in range(T)])
    cost_charge    = pulp.lpSum([(price[t] + imp_fee[t]) * c_grid[t] for t in range(T)])
    throughput     = pulp.lpSum([d_grid[t] + d_load[t] for t in range(T)])
    deg_cost_series = float(deg_cost) * throughput

    gross_margin_series = revenue_export + savings_self - cost_charge - deg_cost_series
    prob += gross_margin_series

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=int(params.get("solver_time_limit_s", 120))))
    status = pulp.LpStatus[prob.status]

    val = lambda x: float(pulp.value(x))
    revenue_series  = val(revenue_export)
    savings_series  = val(savings_self)
    charge_series   = val(cost_charge)
    thr_series      = val(throughput)
    margin_series   = val(gross_margin_series)

    revenue_annual = float(_annual_factor(T, dt) * revenue_series)
    savings_annual = float(_annual_factor(T, dt) * savings_series)
    charge_annual  = float(_annual_factor(T, dt) * charge_series)
    thr_annual     = float(_annual_factor(T, dt) * thr_series)
    margin_annual  = float(_annual_factor(T, dt) * margin_series)

    bess_capex = bess_capex_eur(E_cap, P_cap, float(params.get("c_E_capex", 250.0)), float(params.get("c_P_capex", 150.0)))
    bess_ann   = annualized_cost(bess_capex, float(params.get("discount_rate", 8.0)), int(params.get("lifetime_years", 15)))

    # energia transacionada com o mercado (para opex_var_trade)
    energy_market_series = val(pulp.lpSum([g_load[t] + c_grid[t] + pv_exp[t] + d_grid[t] for t in range(T)]))
    energy_market_annual = float(_annual_factor(T, dt) * energy_market_series)
    opex_var_trade_annual = float(params.get("opex_var_trade_eur_per_MWh", 0.0)) * energy_market_annual

    # energia PV (para opex_var_gen)
    energy_pv_series = val(pulp.lpSum([pv_load[t] + c_pv[t] + pv_exp[t] for t in range(T)]))
    energy_pv_annual = float(_annual_factor(T, dt) * energy_pv_series)
    opex_var_gen_annual = float(params.get("opex_var_gen_eur_per_mwh", 0.0)) * energy_pv_annual

    ebitda_annual = margin_annual - bess_ann - float(params.get("opex_fix_bess", 0.0)) - float(params.get("opex_fix_gen", 0.0)) - opex_var_trade_annual - opex_var_gen_annual

    energy_delivered_annual = float(_annual_factor(T, dt) * val(pulp.lpSum([pv_load[t] + pv_exp[t] + d_grid[t] + d_load[t] for t in range(T)])))
    lcoe_total = lcoe_annual(capex_total_eur=float(params.get("capex_gen", 0.0)) + bess_capex,
                             rate_percent=float(params.get("discount_rate", 8.0)),
                             n_years=int(params.get("lifetime_years", 15)),
                             energy_annual_MWh=max(1e-9, energy_delivered_annual),
                             opex_fix_eur_per_year=float(params.get("opex_fix_bess", 0.0)) + float(params.get("opex_fix_gen", 0.0)),
                             opex_var_eur_per_MWh=float(params.get("opex_var_trade_eur_per_MWh", 0.0)) + float(params.get("opex_var_gen_eur_per_mwh", 0.0)))

    result = {
        "status_text": status,
        "dt_hours": dt,
        "annual_factor": _annual_factor(T, dt),
        "P_cap_MW": P_cap,
        "E_cap_MWh": E_cap,
        "Revenue_export_annual_EUR": revenue_annual,
        "Savings_selfcons_annual_EUR": savings_annual,
        "Cost_charge_grid_annual_EUR": charge_annual,
        "Throughput_annual_MWh": thr_annual,
        "Gross_margin_annual_EUR": margin_annual,
        "BESS_CAPEX_EUR": bess_capex,
        "BESS_annual_cost_EUR": bess_ann,
        "EBITDA_project_annual_EUR": ebitda_annual,
        "LCOE_total_EUR_per_MWh": lcoe_total,
        "ROI_annual_%": (100.0 * ebitda_annual / bess_capex) if bess_capex > 0 else None,
        "Payback_years": (bess_capex / ebitda_annual) if ebitda_annual > 1e-9 else None,
    }

    if return_schedule:
        data = {
            "datetime": df["datetime"],
            "price_EUR_per_MWh": price,
            "pv_MWh": pv_MWh,
            "load_MWh": load_MWh,
            "import_fee_EUR_per_MWh": imp_fee,
            "export_fee_EUR_per_MWh": exp_fee,
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

    return result


def optimize_site_bess_mw(price_df: pd.DataFrame,
                           pv_df: Optional[pd.DataFrame],
                           load_df: Optional[pd.DataFrame],
                           params: dict,
                           P_values_MW,
                           C_values_per_hour,
                           objective: str = "ROI") -> dict:
    best = None
    for P in P_values_MW:
        for C in C_values_per_hour:
            r = run_site_bess_mw(price_df, pv_df, load_df, params, float(P), float(C), return_schedule=False)
            score = r["ROI_annual_%"] if objective.upper() == "ROI" else r["EBITDA_project_annual_EUR"]
            if score is None:
                continue
            if (best is None) or (score > best["score"]):
                best = {"score": score, "P": float(P), "C": float(C), "res": r}
    final = run_site_bess_mw(price_df, pv_df, load_df, params, best["P"], best["C"], return_schedule=True)
    final["status_text"] = f"Melhor par: P={best['P']} MW, C={best['C']}  (objetivo: {objective})"
    return final
