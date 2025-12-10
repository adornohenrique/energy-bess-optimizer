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
# Helpers básicos
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
# Baseline (sem BESS) + EBITDA_base
# =========================================================
def run_baseline(price_df: pd.DataFrame,
                 gen_df: pd.DataFrame,
                 params: dict) -> dict:
    """
    Baseline sem BESS. Vende geração até o cap de exportação.
    Calcula LCOE_base e EBITDA_base (= Receita − OPEX_fix_gen − OPEX_var_gen·E_annual).
    """
    dt_hours = _infer_dt_hours(gen_df["datetime"])
    af = _annual_factor(len(gen_df), dt_hours)

    gen_prep = _prepare_generation(
        gen_df, params.get("availability_pct", 100.0),
        params.get("P_grid_export_max", 0.0),
        dt_hours
    )
    df = _merge_prices_generation(price_df, gen_prep)

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
        opex_var_eur_per_MWh=params.get("opex_var_gen_eur_per_mwh", 0.0),
    )

    ebitda_base = (
        revenue_annual
        - params.get("opex_fix_gen", 0.0)
        - params.get("opex_var_gen_eur_per_mwh", 0.0) * E_annual
    )

    return {
        "E_series_MWh": float(E_series),
        "E_annual_MWh": float(E_annual),
        "Revenue_series_EUR": float(revenue_series),
        "Revenue_annual_EUR": float(revenue_annual),
        "EBITDA_annual_EUR": float(ebitda_base),
        "LCOE_base_EUR_per_MWh": float(lcoe_base),
        "schedule_baseline": df[["datetime", "gen_MWh", "sold_direct_MWh", "spill_MWh", "price_EUR_per_MWh"]],
        "dt_hours": dt_hours,
        "annual_factor": af,
    }


# =========================================================
# Heurística “Preço > LCOE” (sem rede) pedida
# =========================================================
def run_price_threshold_dispatch(price_df: pd.DataFrame,
                                 gen_df: pd.DataFrame,
                                 baseline: dict,
                                 params: dict) -> dict:
    """
    Política:
    - Se preço > LCOE_base: vende geração (até cap de export) e pode descarregar BESS.
    - Se preço ≤ LCOE_base: carrega BESS com geração; se cheio, curtail.
    - Sem carga pela rede.
    """
    dt = _infer_dt_hours(gen_df["datetime"])
    af = _annual_factor(len(gen_df), dt)

    gen_prep = _prepare_generation(
        gen_df, params.get("availability_pct", 100.0),
        params.get("P_grid_export_max", 0.0), dt
    )
    df = _merge_prices_generation(price_df, gen_prep).copy()

    lcoe_base = float(baseline["LCOE_base_EUR_per_MWh"])
    E_cap = float(params.get("E_cap_max", 0.0))
    P_cap = float(params.get("P_cap_max", 0.0))
    eta_c = float(params.get("eta_charge", 0.95))
    eta_d = float(params.get("eta_discharge", 0.95))

    T = len(df)
    soc = 0.0
    soc_list, sell_gen, charge_ren, discharge, spill = [], np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    exp_cap = (params.get("P_grid_export_max", 0.0) if params.get("P_grid_export_max", 0.0) > 0 else 1e12) * dt

    for t in range(T):
        price = float(df.loc[t, "price_EUR_per_MWh"])
        g = float(df.loc[t, "gen_MWh"])

        if price > lcoe_base:
            sell = min(g, exp_cap)
            sell_gen[t] = sell
            g_left = g - sell
            d_max = min(P_cap * dt, soc * eta_d)
            discharge[t] = d_max
            soc -= d_max / eta_d
            if g_left > 0:
                c_room = (E_cap - soc) / eta_c
                c = min(g_left, P_cap * dt, c_room)
                charge_ren[t] = c
                soc += eta_c * c
                spill[t] = g_left - c
        else:
            c_room = (E_cap - soc) / eta_c
            c = min(g, P_cap * dt, c_room)
            charge_ren[t] = c
            soc += eta_c * c
            spill[t] = g - c

        soc = max(0.0, min(E_cap, soc))
        soc_list.append(soc)

    df_out = df.copy()
    df_out["sold_from_gen_MWh"] = sell_gen
    df_out["charge_from_ren_MWh"] = charge_ren
    df_out["discharge_MWh"] = discharge
    df_out["spill_MWh"] = spill
    df_out["soc_MWh"] = soc_list

    energy_series = sell_gen.sum() + discharge.sum()
    energy_annual = af * energy_series
    revenue_series = (df["price_EUR_per_MWh"].values * (sell_gen + discharge)).sum()
    revenue_annual = af * revenue_series

    return {
        "policy_name": "Heurística: preço > LCOE",
        "Energy_annual_MWh": float(energy_annual),
        "Revenue_annual_EUR": float(revenue_annual),
        "Throughput_annual_MWh": float(discharge.sum() * af),  # throughput = descarga
        "schedule": df_out,
        "LCOE_base_ref": lcoe_base,
    }


# =========================================================
# MILP (ótimo) — Big-M moderado, limites E/P, throughput = descarga
# =========================================================
def run_with_bess(price_df: pd.DataFrame,
                  gen_df: pd.DataFrame,
                  params: dict) -> dict:
    """
    Maximiza EBITDA anual do projeto com MILP:
      - Alternância carga/descarga via Big-M = P_cap_max·Δt.
      - Permite carga de rede (opcional).
      - Throughput = descarga anual; limite: discharge_annual ≤ E_cap·cycles.
      - EBITDA projeto = Receita − Custo rede − Anualização CAPEX BESS − OPEX fixos (usina+BESS)
                        − OPEX_var_gen·(venda direta) − OPEX_var_trade·(energia entregue total)
                        − custo degradação (€/MWh·descarga).
    """
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

    # Limites / Big-M
    E_cap_max = float(params.get("E_cap_max", 1e4))  # MWh
    P_cap_max = float(params.get("P_cap_max", 1e4))  # MW
    exp_cap = (params.get("P_grid_export_max", 0.0) if params.get("P_grid_export_max", 0.0) > 0 else 1e12) * dt
    imp_cap = (params.get("P_grid_import_max", 0.0) if params.get("P_grid_import_max", 0.0) > 0 else 1e12) * dt
    M_step = max(1e-6, P_cap_max * dt)

    # Parâmetros do BESS
    eta_c = float(params.get("eta_charge", 0.95))
    eta_d = float(params.get("eta_discharge", 0.95))
    allow_grid = bool(params.get("allow_grid_charging", True))
    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0))
    cycles_max = float(params.get("cycles_per_year_max", 0.0))
    time_limit = int(params.get("solver_time_limit_s", 180))

    # OPEX
    opex_fix_gen = float(params.get("opex_fix_gen", 0.0))
    opex_fix_bess = float(params.get("opex_fix_bess", 0.0))
    opex_var_gen = float(params.get("opex_var_gen_eur_per_mwh", 0.0))
    opex_var_trade = float(params.get("opex_var_trade_eur_per_mwh", 0.0))

    # Variáveis
    E_cap = pulp.LpVariable("E_cap_MWh", lowBound=0, upBound=E_cap_max)
    P_cap = pulp.LpVariable("P_cap_MW", lowBound=0, upBound=P_cap_max)

    soc = pulp.LpVariable.dicts("soc", range(T), lowBound=0, upBound=E_cap)
    c_ren = pulp.LpVariable.dicts("charge_from_ren", range(T), lowBound=0, upBound=P_cap * dt)
    c_grid = pulp.LpVariable.dicts("charge_from_grid", range(T), lowBound=0, upBound=P_cap * dt)
    d = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=P_cap * dt)
    s_dir = pulp.LpVariable.dicts("sold_from_gen", range(T), lowBound=0)
    spill = pulp.LpVariable.dicts("spill", range(T), lowBound=0)
    y = pulp.LpVariable.dicts("is_charging", range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)

    prob = pulp.LpProblem("BESS_Arbitrage_MILP", pulp.LpMaximize)

    for t in range(T):
        g_t = float(df.loc[t, "gen_MWh"])

        # Geração: vende + carrega + curtail = geração
        prob += s_dir[t] + c_ren[t] + spill[t] == g_t, f"gen_balance_{t}"

        # Alternância (Big-M)
        prob += c_ren[t] + c_grid[t] <= M_step * y[t], f"charge_switch_{t}"
        prob += d[t] <= M_step * (1 - y[t]), f"discharge_switch_{t}"

        # Limites de export/import
        prob += s_dir[t] + d[t] <= exp_cap, f"export_cap_{t}"
        if not allow_grid:
            prob += c_grid[t] == 0, f"no_grid_{t}"
        else:
            prob += c_grid[t] <= imp_cap, f"import_cap_{t}"

        # SOC dinâmico
        prev = T - 1 if t == 0 else t - 1
        prob += soc[t] == soc[prev] + eta_c * (c_ren[t] + c_grid[t]) - d[t] / eta_d, f"soc_dyn_{t}"

    # Termos econômicos
    price = df["price_EUR_per_MWh"].values
    sell_direct_series = pulp.lpSum([s_dir[t] for t in range(T)])
    discharge_series = pulp.lpSum([d[t] for t in range(T)])
    energy_series = sell_direct_series + discharge_series

    revenue_annual = _annual_factor(T, dt) * pulp.lpSum([price[t] * (s_dir[t] + d[t]) for t in range(T)])
    grid_cost_annual = _annual_factor(T, dt) * pulp.lpSum([price[t] * c_grid[t] for t in range(T)])
    energy_annual = _annual_factor(T, dt) * energy_series
    discharge_annual = _annual_factor(T, dt) * discharge_series  # throughput = descarga

    # Limite de ciclos
    if cycles_max > 0:
        prob += discharge_annual <= E_cap * cycles_max, "cycles_per_year_limit"

    # CAPEX anualizado BESS
    bess_capex_expr = 1000.0 * params["c_E_capex"] * E_cap + 1000.0 * params["c_P_capex"] * P_cap
    bess_annual_cost = crf(params["discount_rate"], int(params["lifetime_years"])) * bess_capex_expr

    # OPEX variáveis
    opex_var_gen_annual = opex_var_gen * _annual_factor(T, dt) * sell_direct_series
    opex_var_trade_annual = opex_var_trade * energy_annual
    deg_cost_annual = params.get("deg_cost_eur_per_MWh_throughput", 0.0) * discharge_annual

    # EBITDA do projeto
    ebitda_annual = (
        revenue_annual - grid_cost_annual
        - bess_annual_cost - opex_fix_bess - opex_fix_gen
        - opex_var_gen_annual - opex_var_trade_annual
        - deg_cost_annual
    )

    prob += ebitda_annual

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(params.get("solver_time_limit_s", 180)))
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {"status_text": f"Solver terminou sem ótima (status: {status}).", "schedule": df}

    # Extrair
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
    discharge_annual_val = pulp.value(discharge_annual)
    deg_cost_annual_val = pulp.value(deg_cost_annual)
    bess_capex_val = bess_capex_eur(E_cap_opt, P_cap_opt, params["c_E_capex"], params["c_P_capex"])
    bess_annual_cost_val = annualized_cost(bess_capex_val, params["discount_rate"], int(params["lifetime_years"]))
    ebitda_val = pulp.value(ebitda_annual)

    # LCOEs
    lcoe_total_entregue = lcoe_annual(
        capex_total_eur=params["capex_gen"] + bess_capex_val,
        rate_percent=params["discount_rate"],
        n_years=int(params["lifetime_years"]),
        energy_annual_MWh=energy_annual_val,
        opex_fix_eur_per_year=float(params.get("opex_fix_gen", 0.0)) + float(params.get("opex_fix_bess", 0.0)),
        opex_var_eur_per_MWh=float(params.get("opex_var_gen_eur_per_mwh", 0.0)) + float(params.get("opex_var_trade_eur_per_mwh", 0.0)),
    )

    lcoe_geracao_pura = lcoe_annual(
        capex_total_eur=params["capex_gen"],
        rate_percent=params["discount_rate"],
        n_years=int(params["lifetime_years"]),
        energy_annual_MWh=float(params.get("baseline_E_annual_MWh", 1.0)),
        opex_fix_eur_per_year=float(params.get("opex_fix_gen", 0.0)),
        opex_var_eur_per_MWh=float(params.get("opex_var_gen_eur_per_mwh", 0.0)),
    )

    return {
        "status_text": "Solução ótima encontrada (MILP)",
        "E_cap_opt_MWh": float(E_cap_opt),
        "P_cap_opt_MW": float(P_cap_opt),
        "Energy_annual_MWh": float(energy_annual_val),
        "Throughput_annual_MWh": float(discharge_annual_val),
        "Deg_cost_annual_EUR": float(deg_cost_annual_val),
        "Revenue_annual_EUR": float(revenue_val),
        "Grid_energy_cost_annual_EUR": float(grid_cost_val),
        "BESS_CAPEX_EUR": float(bess_capex_val),
        "BESS_annual_cost_EUR": float(bess_annual_cost_val),
        "EBITDA_project_annual_EUR": float(ebitda_val),
        "LCOE_total_entregue_EUR_per_MWh": float(lcoe_total_entregue),
        "LCOE_geracao_pura_EUR_per_MWh": float(lcoe_geracao_pura),
        "schedule": df_out,
    }


# =========================================================
# Rolling-horizon (15 min) — sizing + despacho anual
# =========================================================
def choose_representative_weeks(price_df: pd.DataFrame, k: int = 4) -> list:
    """
    Escolhe k semanas representativas pela maior volatilidade de preço (std semanal).
    Retorna lista de timestamps (início de semana, normalizados para 00:00 de seg/semana).
    """
    p = price_df.copy()
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p.sort_values("datetime")
    p["week"] = p["datetime"].dt.to_period("W-MON")
    agg = p.groupby("week")["price_EUR_per_MWh"].std().sort_values(ascending=False)
    weeks = [pd.Period(str(w), freq="W-MON") for w in agg.index[:k]]
    starts = [w.start_time.normalize() for w in weeks]
    return sorted(starts)


def _concat_blocks(price_df, gen_df, starts, window_days=7):
    """Concatena janelas semanais em uma única série; retorna df e lista de (ini, fim) índices de cada bloco."""
    price = price_df.copy(); gen = gen_df.copy()
    price["datetime"] = pd.to_datetime(price["datetime"]).sort_values()
    gen["datetime"] = pd.to_datetime(gen["datetime"]).sort_values()
    dfs = []
    blocks = []
    base_idx = 0
    for s in starts:
        e = s + pd.Timedelta(days=window_days)
        p = price[(price["datetime"] >= s) & (price["datetime"] < e)]
        g = gen[(gen["datetime"] >= s) & (gen["datetime"] < e)]
        df = _merge_prices_generation(p, _prepare_generation(g, 100.0, 0.0, _infer_dt_hours(g["datetime"])))
        df = df.sort_values("datetime").reset_index(drop=True)
        n = len(df)
        if n == 0:
            continue
        df["block_id"] = len(blocks)
        dfs.append(df)
        blocks.append((base_idx, base_idx + n - 1))
        base_idx += n
    if not dfs:
        raise ValueError("Nenhuma janela válida encontrada para rolling-horizon.")
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all, blocks


def run_milp_size_on_windows(price_df, gen_df, params, starts, window_days=7):
    """
    Dimensiona E_cap e P_cap globalmente usando janelas concatenadas.
    SOC cíclico por bloco (início = fim em cada semana).
    """
    df_all, blocks = _concat_blocks(price_df, gen_df, starts, window_days)
    dt = _infer_dt_hours(df_all["datetime"])
    T = len(df_all)

    # Limites / Big-M
    E_cap_max = float(params.get("E_cap_max", 1e4))
    P_cap_max = float(params.get("P_cap_max", 1e4))
    M_step = max(1e-6, P_cap_max * dt)
    exp_cap = (params.get("P_grid_export_max", 0.0) if params.get("P_grid_export_max", 0.0) > 0 else 1e12) * dt
    imp_cap = (params.get("P_grid_import_max", 0.0) if params.get("P_grid_import_max", 0.0) > 0 else 1e12) * dt

    eta_c = float(params.get("eta_charge", 0.95))
    eta_d = float(params.get("eta_discharge", 0.95))
    allow_grid = bool(params.get("allow_grid_charging", True))
    opex_var_gen = float(params.get("opex_var_gen_eur_per_mwh", 0.0))
    opex_var_trade = float(params.get("opex_var_trade_eur_per_mwh", 0.0))
    opex_fix_gen = float(params.get("opex_fix_gen", 0.0))
    opex_fix_bess = float(params.get("opex_fix_bess", 0.0))
    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0))
    time_limit = int(params.get("solver_time_limit_s", 180))

    price = df_all["price_EUR_per_MWh"].values
    gen_ = df_all["gen_MWh"].values

    E_cap = pulp.LpVariable("E_cap_MWh", lowBound=0, upBound=E_cap_max)
    P_cap = pulp.LpVariable("P_cap_MW", lowBound=0, upBound=P_cap_max)

    soc = pulp.LpVariable.dicts("soc", range(T), lowBound=0, upBound=E_cap)
    c_ren = pulp.LpVariable.dicts("charge_from_ren", range(T), lowBound=0, upBound=P_cap * dt)
    c_grid = pulp.LpVariable.dicts("charge_from_grid", range(T), lowBound=0, upBound=P_cap * dt)
    d = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=P_cap * dt)
    s_dir = pulp.LpVariable.dicts("sold_from_gen", range(T), lowBound=0)
    spill = pulp.LpVariable.dicts("spill", range(T), lowBound=0)
    y = pulp.LpVariable.dicts("is_charging", range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)

    prob = pulp.LpProblem("Sizing_on_windows", pulp.LpMaximize)

    for t in range(T):
        # balanço da geração
        prob += s_dir[t] + c_ren[t] + spill[t] == gen_[t], f"gen_balance_{t}"
        # alternância
        prob += c_ren[t] + c_grid[t] <= M_step * y[t], f"charge_switch_{t}"
        prob += d[t] <= M_step * (1 - y[t]), f"discharge_switch_{t}"
        # limites grid
        prob += s_dir[t] + d[t] <= exp_cap, f"export_cap_{t}"
        if not allow_grid:
            prob += c_grid[t] == 0, f"no_grid_{t}"
        else:
            prob += c_grid[t] <= imp_cap, f"import_cap_{t}"
        # SOC
        prev = T - 1 if t == 0 else t - 1
        prob += soc[t] == soc[prev] + eta_c * (c_ren[t] + c_grid[t]) - d[t] / eta_d, f"soc_dyn_{t}"

    # SOC cíclico por bloco (sem carregar legado entre semanas representativas)
    for (i0, i1) in blocks:
        prob += soc[i1] == soc[i0], f"soc_cyclic_block_{i0}_{i1}"

    # objetivo: maximizar EBITDA "semanas representativas"
    sell_direct_series = pulp.lpSum([s_dir[t] for t in range(T)])
    discharge_series   = pulp.lpSum([d[t] for t in range(T)])
    energy_series      = sell_direct_series + discharge_series
    revenue            = pulp.lpSum([price[t] * (s_dir[t] + d[t]) for t in range(T)])
    grid_cost          = pulp.lpSum([price[t] * c_grid[t] for t in range(T)])
    opex_var_gen_sum   = opex_var_gen * sell_direct_series
    opex_var_trade_sum = opex_var_trade * energy_series
    deg_cost_sum       = deg_cost * discharge_series

    # anualização do CAPEX BESS (usa CRF da vida toda; aqui é função do sizing)
    bess_capex_expr = 1000.0 * params["c_E_capex"] * E_cap + 1000.0 * params["c_P_capex"] * P_cap
    bess_annual_cost = crf(params["discount_rate"], int(params["lifetime_years"])) * bess_capex_expr

    # EBITDA equivalente (sem fator anual; sizing relativo)
    ebitda_like = revenue - grid_cost - opex_var_gen_sum - opex_var_trade_sum - deg_cost_sum - bess_annual_cost - opex_fix_gen - opex_fix_bess
    prob += ebitda_like

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {"status_text": f"Sizing sem ótima (status: {status})."}

    return {
        "status_text": "Sizing ótimo em janelas",
        "E_cap_sized_MWh": float(E_cap.value()),
        "P_cap_sized_MW": float(P_cap.value()),
    }


def _milp_dispatch_fixed_EP(df, params, E_cap_fix, P_cap_fix, soc0=None, time_limit_s=15):
    """
    MILP para um bloco com E/P fixos e SOC inicial opcional.
    Retorna schedule e SOC final.
    """
    dt = _infer_dt_hours(df["datetime"])
    T = len(df)

    exp_cap = (params.get("P_grid_export_max", 0.0) if params.get("P_grid_export_max", 0.0) > 0 else 1e12) * dt
    imp_cap = (params.get("P_grid_import_max", 0.0) if params.get("P_grid_import_max", 0.0) > 0 else 1e12) * dt
    M_step = max(1e-6, float(P_cap_fix) * dt)

    eta_c = float(params.get("eta_charge", 0.95))
    eta_d = float(params.get("eta_discharge", 0.95))
    allow_grid = bool(params.get("allow_grid_charging", True))
    opex_var_gen = float(params.get("opex_var_gen_eur_per_mwh", 0.0))
    opex_var_trade = float(params.get("opex_var_trade_eur_per_mwh", 0.0))
    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0))

    price = df["price_EUR_per_MWh"].values
    gen_ = df["gen_MWh"].values

    # Variáveis
    soc = pulp.LpVariable.dicts("soc", range(T), lowBound=0, upBound=E_cap_fix)
    c_ren = pulp.LpVariable.dicts("charge_from_ren", range(T), lowBound=0, upBound=P_cap_fix * dt)
    c_grid = pulp.LpVariable.dicts("charge_from_grid", range(T), lowBound=0, upBound=P_cap_fix * dt)
    d = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=P_cap_fix * dt)
    s_dir = pulp.LpVariable.dicts("sold_from_gen", range(T), lowBound=0)
    spill = pulp.LpVariable.dicts("spill", range(T), lowBound=0)
    y = pulp.LpVariable.dicts("is_charging", range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)

    prob = pulp.LpProblem("Dispatch_fixed_EP", pulp.LpMaximize)

    for t in range(T):
        prob += s_dir[t] + c_ren[t] + spill[t] == gen_[t], f"gen_balance_{t}"
        prob += c_ren[t] + c_grid[t] <= M_step * y[t], f"charge_switch_{t}"
        prob += d[t] <= M_step * (1 - y[t]), f"discharge_switch_{t}"
        prob += s_dir[t] + d[t] <= exp_cap, f"export_cap_{t}"
        if not allow_grid:
            prob += c_grid[t] == 0, f"no_grid_{t}"
        else:
            prob += c_grid[t] <= imp_cap, f"import_cap_{t}"
        prev = T - 1 if t == 0 else t - 1
        prob += soc[t] == soc[prev] + eta_c * (c_ren[t] + c_grid[t]) - d[t] / eta_d, f"soc_dyn_{t}"

    # SOC inicial (se fornecido)
    if soc0 is not None:
        prob += soc[0] == max(0.0, min(E_cap_fix, float(soc0))), "soc_init"

    revenue = pulp.lpSum([price[t] * (s_dir[t] + d[t]) for t in range(T)])
    grid_cost = pulp.lpSum([price[t] * c_grid[t] for t in range(T)])
    opex_var_gen_sum = opex_var_gen * pulp.lpSum([s_dir[t] for t in range(T)])
    opex_var_trade_sum = opex_var_trade * pulp.lpSum([s_dir[t] + d[t] for t in range(T)])
    deg_cost_sum = deg_cost * pulp.lpSum([d[t] for t in range(T)])

    obj = revenue - grid_cost - opex_var_gen_sum - opex_var_trade_sum - deg_cost_sum
    prob += obj

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(time_limit_s))
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",):
        status = "Feasible"

    # Extrair
    df_out = df.copy()
    df_out["soc_MWh"] = [soc[t].value() for t in range(T)]
    df_out["charge_from_ren_MWh"] = [c_ren[t].value() for t in range(T)]
    df_out["charge_from_grid_MWh"] = [c_grid[t].value() for t in range(T)]
    df_out["discharge_MWh"] = [d[t].value() for t in range(T)]
    df_out["sold_from_gen_MWh"] = [s_dir[t].value() for t in range(T)]
    df_out["spill_MWh"] = [spill[t].value() for t in range(T)]
    df_out["status"] = status

    return {
        "schedule": df_out,
        "revenue_series": float(pulp.value(revenue)),
        "grid_cost_series": float(pulp.value(grid_cost)),
        "sell_direct_series": float(pulp.value(pulp.lpSum([s_dir[t] for t in range(T)]))),
        "discharge_series": float(pulp.value(pulp.lpSum([d[t] for t in range(T)]))),
        "soc_end": float(df_out["soc_MWh"].iloc[-1]),
        "status": status,
    }


def run_milp_rolling_year(price_df, gen_df, params,
                          sizing_weeks=4, window_days=7, step_days=7,
                          time_limit_s_sizing=180, time_limit_s_dispatch=15):
    """
    2 fases:
      1) Sizing global (E/P) em k semanas representativas (ótimo MILP).
      2) Despacho do ano inteiro em janelas 15 min com E/P fixos e SOC encadeado.
    Retorna KPIs anuais e E/P dimensionados.
    """
    # 1) escolher semanas e dimensionar
    starts = choose_representative_weeks(price_df, k=max(1, int(sizing_weeks)))
    sizing = run_milp_size_on_windows(price_df, gen_df, dict(params, solver_time_limit_s=time_limit_s_sizing),
                                      starts, window_days)
    if not sizing.get("status_text", "").startswith("Sizing ótimo"):
        return {"status_text": sizing.get("status_text", "Falha no sizing.")}

    E_cap = sizing["E_cap_sized_MWh"]
    P_cap = sizing["P_cap_sized_MW"]

    # 2) rolling
    p = price_df.copy(); g = gen_df.copy()
    p["datetime"] = pd.to_datetime(p["datetime"]).sort_values()
    g["datetime"] = pd.to_datetime(g["datetime"]).sort_values()

    start = max(p["datetime"].min(), g["datetime"].min())
    end   = min(p["datetime"].max(), g["datetime"].max())
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Faixa de datas inválida para rolling.")

    cur = start
    soc0 = 0.0
    rows = []
    energy_sum = 0.0
    revenue_sum = 0.0
    grid_cost_sum = 0.0
    sell_sum = 0.0
    discharge_sum = 0.0

    while cur < end:
        blk_end = min(cur + pd.Timedelta(days=window_days), end)
        p_blk = p[(p["datetime"] >= cur) & (p["datetime"] < blk_end)]
        g_blk = g[(g["datetime"] >= cur) & (g["datetime"] < blk_end)]
        if len(p_blk) == 0 or len(g_blk) == 0:
            cur = cur + pd.Timedelta(days=step_days)
            continue
        df_blk = _merge_prices_generation(p_blk, _prepare_generation(g_blk, params.get("availability_pct", 100.0),
                                                                     params.get("P_grid_export_max", 0.0),
                                                                     _infer_dt_hours(g_blk["datetime"])))
        res_blk = _milp_dispatch_fixed_EP(
            df_blk, params, E_cap, P_cap, soc0=soc0, time_limit_s=time_limit_s_dispatch
        )
        rows.append(res_blk["schedule"])
        soc0 = res_blk["soc_end"]
        revenue_sum += res_blk["revenue_series"]
        grid_cost_sum += res_blk["grid_cost_series"]
        sell_sum += res_blk["sell_direct_series"]
        discharge_sum += res_blk["discharge_series"]
        energy_sum += (res_blk["sell_direct_series"] + res_blk["discharge_series"])
        cur = cur + pd.Timedelta(days=step_days)

    df_year = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["datetime"])
    dt = _infer_dt_hours(df_year["datetime"]) if not df_year.empty else _infer_dt_hours(g["datetime"])
    af = _annual_factor(len(df_year), dt) if not df_year.empty else _annual_factor(len(g), _infer_dt_hours(g["datetime"]))

    revenue_annual = af * revenue_sum
    grid_cost_annual = af * grid_cost_sum
    energy_annual = af * energy_sum
    discharge_annual = af * discharge_sum

    # custos fixos e capex BESS
    bess_capex_val = bess_capex_eur(E_cap, P_cap, params["c_E_capex"], params["c_P_capex"])
    bess_annual_cost_val = annualized_cost(bess_capex_val, params["discount_rate"], int(params["lifetime_years"]))
    opex_fix_total = float(params.get("opex_fix_gen", 0.0)) + float(params.get("opex_fix_bess", 0.0))
    opex_var_trade = float(params.get("opex_var_trade_eur_per_mwh", 0.0)) * energy_annual
    opex_var_gen = float(params.get("opex_var_gen_eur_per_mwh", 0.0)) * (af * sell_sum)
    deg_cost = float(params.get("deg_cost_eur_per_MWh_throughput", 0.0)) * discharge_annual

    ebitda_project = revenue_annual - grid_cost_annual - bess_annual_cost_val - opex_fix_total - opex_var_trade - opex_var_gen - deg_cost

    lcoe_total_entregue = lcoe_annual(
        capex_total_eur=params["capex_gen"] + bess_capex_val,
        rate_percent=params["discount_rate"],
        n_years=int(params["lifetime_years"]),
        energy_annual_MWh=energy_annual,
        opex_fix_eur_per_year=opex_fix_total,
        opex_var_eur_per_MWh=float(params.get("opex_var_gen_eur_per_mwh", 0.0)) + float(params.get("opex_var_trade_eur_per_mwh", 0.0)),
    )

    return {
        "status_text": "Rolling 15-min concluído",
        "E_cap_opt_MWh": float(E_cap),
        "P_cap_opt_MW": float(P_cap),
        "Energy_annual_MWh": float(energy_annual),
        "Throughput_annual_MWh": float(discharge_annual),
        "Revenue_annual_EUR": float(revenue_annual),
        "Grid_energy_cost_annual_EUR": float(grid_cost_annual),
        "BESS_CAPEX_EUR": float(bess_capex_val),
        "BESS_annual_cost_EUR": float(bess_annual_cost_val),
        "EBITDA_project_annual_EUR": float(ebitda_project),
        "LCOE_total_entregue_EUR_per_MWh": float(lcoe_total_entregue),
        "schedule": df_year,
    }


# =========================================================
# Sensibilidades & batch (inalterado conceitualmente)
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
                    "EBITDA_project_EUR_ano": res["EBITDA_project_annual_EUR"],
                    "LCOE_total_entregue": res["LCOE_total_entregue_EUR_per_MWh"],
                })
    return pd.DataFrame(rows)


def run_batch_scenarios(price_dfs, gen_dfs, labels, params):
    results = []
    rows = []
    for (p, g, name) in zip(price_dfs, gen_dfs, labels):
        base = run_baseline(p, g, params)
        p2 = dict(params); p2["baseline_E_annual_MWh"] = base["E_annual_MWh"]
        withb = run_with_bess(p, g, p2)
        results.append({"label": name, "baseline": base, "with_bess": withb})

        if withb.get("status_text", "").startswith("Solução"):
            rows.append({
                "Cenário": name,
                "E_cap_MWh": withb["E_cap_opt_MWh"],
                "P_cap_MW": withb["P_cap_opt_MW"],
                "EBITDA_EUR_ano": withb["EBITDA_project_annual_EUR"],
                "LCOE_base": base["LCOE_base_EUR_per_MWh"],
                "LCOE_total_entregue": withb["LCOE_total_entregue_EUR_per_MWh"],
                "Receita_base_EUR_ano": base["Revenue_annual_EUR"],
                "Receita_BESS_EUR_ano": withb["Revenue_annual_EUR"],
                "Throughput_MWh_ano": withb["Throughput_annual_MWh"],
            })
        else:
            rows.append({
                "Cenário": name,
                "E_cap_MWh": None, "P_cap_MW": None, "EBITDA_EUR_ano": None,
                "LCOE_base": base["LCOE_base_EUR_per_MWh"],
                "LCOE_total_entregue": None,
                "Receita_base_EUR_ano": base["Revenue_annual_EUR"],
                "Receita_BESS_EUR_ano": None,
                "Throughput_MWh_ano": None,
            })
    return results, pd.DataFrame(rows)
