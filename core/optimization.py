# core/optimization.py
from typing import Optional
import numpy as np
import pandas as pd
import pulp

from core.finance import annualized_cost, lcoe_annual, bess_capex_eur


# ----------------------- utilitários -----------------------
def _infer_dt_hours(dt_series: pd.Series) -> float:
    s = pd.to_datetime(dt_series)
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
        raise ValueError("Preços: use colunas 'datetime, price_EUR_per_MWh'.")

    df = price_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if pv_df is not None:
        pv = pv_df.copy()
        if "datetime" not in pv or "pv_MW" not in pv:
            raise ValueError("PV: use colunas 'datetime, pv_MW'.")
        pv["datetime"] = pd.to_datetime(pv["datetime"])
        pv = pv.sort_values("datetime").reset_index(drop=True)
        df = pd.merge_asof(df, pv, on="datetime")
    else:
        df["pv_MW"] = 0.0

    if load_df is not None:
        ld = load_df.copy()
        if "datetime" not in ld or "load_MW" not in ld:
            raise ValueError("Carga: use colunas 'datetime, load_MW'.")
        ld["datetime"] = pd.to_datetime(ld["datetime"])
        ld = ld.sort_values("datetime").reset_index(drop=True)
        df = pd.merge_asof(df, ld, on="datetime")
    else:
        df["load_MW"] = 0.0

    df[["pv_MW", "load_MW"]] = df[["pv_MW", "load_MW"]].fillna(0.0)
    return df


# ----------------------- baselines -------------------------
def baselines_mw(price_df: pd.DataFrame,
                 pv_df: Optional[pd.DataFrame] = None,
                 load_df: Optional[pd.DataFrame] = None,
                 import_fee: float = 0.0,
                 export_fee: float = 0.0) -> dict:
    df = _merge_all(price_df, pv_df, load_df)
    dt = _infer_dt_hours(df["datetime"])
    af = _annual_factor(len(df), dt)

    price = df["price_EUR_per_MWh"].to_numpy(float)
    pv_MWh = df["pv_MW"].to_numpy(float) * dt
    load_MWh = df["load_MW"].to_numpy(float) * dt

    cost_consumption_series = ((price + float(import_fee)) * load_MWh).sum()
    revenue_solar_only_series = ((price - float(export_fee)) * pv_MWh).sum()

    return {
        "dt_hours": dt,
        "annual_factor": af,
        "Cost_consumption_annual_EUR": af * cost_consumption_series,
        "Revenue_solar_only_annual_EUR": af * revenue_solar_only_series,
    }


# ------------------------- MILP ----------------------------
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

    # parâmetros BESS
    P_cap = float(P_bess_MW)
    c_rate = max(1e-6, float(c_rate_per_hour))
    E_cap = P_cap / c_rate
    eta_c = float(params.get("eta_charge", 0.95))
    eta_d = float(params.get("eta_discharge", 0.95))
    soc_min = float(params.get("soc_min", 0.0)) * E_cap
    soc_max = float(params.get("soc_max", 1.0)) * E_cap

    # rede e tarifas
    import_fee = float(params.get("import_fee_eur_per_MWh", 0.0))
    export_fee = float(params.get("export_fee_eur_per_MWh", 0.0))
    allow_grid_charging = bool(params.get("allow_grid_charging", True))
    imp_cap = float(params.get("P_grid_import_max", 1e12)) * dt
    exp_cap = float(params.get("P_grid_export_max", 1e12)) * dt
    M_step = max(1e-6, P_cap * dt)

    # custos
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

    # modelo
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

    for t in range(T):
        # balanços
        prob += pv_load[t] + c_pv[t] + pv_exp[t] == pv_MWh[t], f"pv_bal_{t}"
        prob += g_load[t] + pv_load[t] + d_load[t] == load_MWh[t], f"load_bal_{t}"

        # potência (P*dt)
        prob += c_grid[t] + c_pv[t] <= M_step, f"lim_charge_{t}"
        prob += d_grid[t] + d_load[t] <= M_step, f"lim_discharge_{t}"

        # exclusividade
        prob += c_grid[t] + c_pv[t] <= M_step * y[t],         f"switch_c_{t}"
        prob += d_grid[t] + d_load[t] <= M_step * (1 - y[t]), f"switch_d_{t}"

        # limites de rede
        prob += g_load[t] + c_grid[t] <= imp_cap,       f"imp_cap_{t}"
        prob += pv_exp[t] + d_grid[t] <= exp_cap,       f"exp_cap_{t}"
        if not allow_grid_charging:
            prob += c_grid[t] == 0,                     f"no_grid_charge_{t}"

        # dinâmica SOC (cíclico)
        prev = T - 1 if t == 0 else t - 1
        prob += soc[t] == soc[prev] + eta_c*(c_grid[t] + c_pv[t]) - (d_grid[t] + d_load[t])/eta_d, f"soc_{t}"

    # objetivo
    revenue_export = pulp.lpSum([(price[t] - export_fee)*(pv_exp[t] + d_grid[t]) for t in range(T)])
    savings_self   = pulp.lpSum([(price[t] + import_fee)*(pv_load[t] + d_load[t]) for t in range(T)])
    cost_charge    = pulp.lpSum([(price[t] + import_fee)*c_grid[t] for t in range(T)])
    throughput     = pulp.lpSum([d_grid[t] + d_load[t] for t in range(T)])
    deg_cost_series = deg_cost * throughput

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

    # anualização
    revenue_annual = af * revenue_series
    savings_annual = af * savings_series
    charge_annual  = af * charge_series
    thr_annual     = af * thr_series
    margin_annual  = af * margin_series

    # capex/opex
    bess_capex = bess_capex_eur(E_cap, P_cap, c_E, c_P)
    bess_ann   = annualized_cost(bess_capex, discount, lifetime)

    # energia transacionada com rede para opex_var_trade
    energy_market_series = val(pulp.lpSum([g_load[t] + c_grid[t] + pv_exp[t] + d_grid[t] for t in range(T)]))
    energy_market_annual = af * energy_market_series
    opex_var_trade_annual = opex_var_trade * energy_market_annual

    # energia PV (para opex_var_gen)
    energy_pv_series = val(pulp.lpSum([pv_load[t] + c_pv[t] + pv_exp[t] for t in range(T)]))
    energy_pv_annual = af * energy_pv_series
    opex_var_gen_annual = opex_var_gen * energy_pv_annual

    ebitda_annual = margin_annual - bess_ann - opex_fix_bess - opex_fix_gen - opex_var_trade_annual - opex_var_gen_annual

    # LCOE total (referência)
    energy_delivered_annual = af * val(pulp.lpSum([pv_load[t] + pv_exp[t] + d_grid[t] + d_load[t] for t in range(T)]))
    lcoe_total = lcoe_annual(capex_total_eur=capex_gen + bess_capex,
                             rate_percent=discount,
                             n_years=lifetime,
                             energy_annual_MWh=max(1e-9, energy_delivered_annual),
                             opex_fix_eur_per_year=opex_fix_bess + opex_fix_gen,
                             opex_var_eur_per_MWh=opex_var_trade + opex_var_gen)

    result = {
        "status_text": status,
        "dt_hours": dt,
        "annual_factor": af,
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


# ------------- varredura (P em MW, C-rate em 1/h) -------------
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
