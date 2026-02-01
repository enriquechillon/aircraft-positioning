# -*- coding: utf-8 -*-
"""
model_func1_sh.py — continuo + fc29 + bloqueo de calle + precedencias por avión + ventanas por avión + delay avión/cliente
- Tiempo continuo: t_start[j], t_end[j] con duración D[j].
- Asignación única a posición: y[j,p].
- No-solape por posición (disyunción linealizada).
- BLOQUEO de calle (entrada/salida) con SEP_IN/OUT = 1e-3 días (≈86s) al estilo cpaste.py.
- Precedencias por avión (según 'task'): t_start[j2] ≥ t_end[j1].
- Ventanas por avión (EarlyStartOfPlane / LateFinishDeadline): se aplican a todos sus trabajos.
- Retraso por avión (último trabajo por avión) y por cliente (suma de retrasos de sus aviones).
- Política cliente→posición "soft" (z[c,p]) como antes.
- Objetivo fc29-like manteniendo métricas (vPresence, v01SwitchPlanes, v01JobInSlot, v01Alpha, vIdle) + ClientDelay via PlaneDelay.
- Gantt con posiciones ordenadas (position1..position5).
- REPORT ENRIQUECIDO (resumen por avión + detalle por trabajo + CSVs adicionales).

Requisitos:
  pip install pyomo pandas numpy plotly openpyxl
  (y Gurobi instalado/licenciado)
"""

import os, math
from datetime import date, timedelta
import pandas as pd
import numpy as np

from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Constraint, ConstraintList, Objective,
    NonNegativeReals, Reals, Any, Binary, minimize, value, SolverFactory
)
from pyomo.opt import TerminationCondition

# -----------------------
# POSICIONES / INTERFERENCIAS
# -----------------------
NO_POSITIONS = 5
POSITIONS = [f'position{i}' for i in range(1, NO_POSITIONS + 1)]

# Cadena frontal (direccional): (front, back)
BASE_INTERF = [("position3", "position4"),
               ("position3", "position5"),
               ("position4", "position5")]

INTERF_IN = list(BASE_INTERF)
INTERF_OUT = list(BASE_INTERF)

# Separaciones mínimas (días) para entrada/salida (≈86 s)
SEP_IN = 0.08
SEP_OUT = 0.08

# -----------------------
# CONFIG
# -----------------------
PLANNING_START = "2024-11-17"  # None => hoy; o "YYYY-MM-DD"

CLIENT_POS_POLICY = "hard"
W_CLIENT_POS = 5000000.0

# Pesos objetivo “fc29-like”
W_MAKESPAN = 1.0
W_JOBINSLOT = 1.0
W_ALPHA = 1.0
W_SWITCH = 0.8
W_PRESENCE = 1.0
W_CLIENT_DELAY = 1.0
W_IDLE = 1.0

# Solver
GAP = 0.05

# (global para report/validación)
data = {}

# -----------------------
# LECTURA
# -----------------------

from datetime import datetime

from datetime import datetime, date
import pandas as pd
import os

# ---------- Helpers de fecha robustos (elemento a elemento) ----------

from datetime import datetime, date
import pandas as pd


def _robust_base_date(ps):
    """Convierte planning_start a Timestamp robustamente."""
    if ps is None:
        return pd.Timestamp(date.today()).normalize()
    if isinstance(ps, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(ps).normalize()
    if isinstance(ps, str):
        s = ps.strip()
        if s.isdigit() and len(s) == 8:
            return pd.to_datetime(s, format="%Y%m%d").normalize()
        try:
            return pd.to_datetime(s, utc=False).normalize()
        except Exception:
            return pd.to_datetime(s, dayfirst=True, utc=False).normalize()
    if isinstance(ps, (int, float)):
        v = float(ps)
        s = str(int(v)) if float(v).is_integer() else None
        if s and len(s) == 8:
            return pd.to_datetime(s, format="%Y%m%d").normalize()
        # No intentamos interpretar serial Excel para planning_start; usa cadena/fecha.
        return pd.Timestamp(date.today()).normalize()
    return pd.Timestamp(date.today()).normalize()


def _to_days_from_base(val, base_date: pd.Timestamp) -> float:
    """
    Convierte un valor a DIAS desde base_date:
      - número → días (si está en 20000..80000 se interpreta como serial Excel → fecha → días)
      - string/fecha → se parsea a fecha y se pasa a días
    """
    if pd.isna(val):
        return 0.0
    # numérico
    try:
        v = float(val)
        if 20000 <= v <= 80000:  # serial Excel
            base_excel = pd.Timestamp("1899-12-30")
            d_int = int(v);
            frac = v - d_int
            dt = base_excel + pd.to_timedelta(d_int, unit="D") + pd.to_timedelta(frac * 86400, unit="s")
            return (dt - base_date).total_seconds() / 86400.0
        return v
    except Exception:
        pass
    # texto/fecha
    dt = pd.to_datetime(val, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return 0.0
    return (pd.Timestamp(dt).normalize() - base_date).total_seconds() / 86400.0


# ---------------------- Función principal pedida ----------------------

def read_input(xlsx_path: str, case_sheet: str, planning_start=PLANNING_START, planes_sheet: str = "Planes2"):
    print(f"Leyendo Excel: {os.path.basename(xlsx_path)} / {case_sheet}")
    base_date = _robust_base_date(planning_start)
    print(f"BASE_DATE: {base_date.date()}  (origen calendario)")

    # ---- Caso (jobs) ----
    df_case = pd.read_excel(xlsx_path, sheet_name=case_sheet)
    rename_map = {
        'plane': 'plane', 'task': 'task', 'job': 'job', 'date': 'date',
        'duration': 'duration', 'movable': 'movable', 'flexible': 'flexible', 'client': 'client'
    }
    df_case = df_case.rename(columns={c: rename_map.get(c, c) for c in df_case.columns})

    if 'plane' not in df_case or 'duration' not in df_case:
        raise ValueError("La hoja del caso debe contener columnas 'plane' y 'duration'.")
    df_case = df_case[pd.to_numeric(df_case['plane'], errors='coerce').notna()]
    df_case = df_case[pd.to_numeric(df_case['duration'], errors='coerce').notna()]
    df_case['plane'] = df_case['plane'].astype(int)
    df_case['duration'] = pd.to_numeric(df_case['duration'], errors='coerce').astype(float)
    if 'client' not in df_case: df_case['client'] = 'NA'
    if 'task' not in df_case:   df_case['task'] = 1
    df_case['task'] = pd.to_numeric(df_case['task'], errors='coerce').fillna(1).astype(int)

    # ---- Planes (SIN fallback) ----
    xls = pd.ExcelFile(xlsx_path)
    if planes_sheet not in xls.sheet_names:
        raise ValueError(f"La hoja '{planes_sheet}' no existe en el Excel.")
    df_planes = pd.read_excel(xlsx_path, sheet_name=planes_sheet)
    if 'plane' not in df_planes.columns:
        raise ValueError(f"La hoja '{planes_sheet}' debe tener columna 'plane'.")

    # early_start y late_finish por avión → DIAS desde BASE_DATE
    early_start_plane = {}
    late_finish_plane = {}
    if 'early_start' in df_planes.columns:
        tmp = df_planes[['plane', 'early_start']].dropna().copy()
        tmp['plane'] = tmp['plane'].astype(int)
        tmp['ES_days'] = tmp['early_start'].apply(lambda v: _to_days_from_base(v, base_date))
        early_start_plane = dict(zip(tmp['plane'], tmp['ES_days']))
    if 'late_finish' in df_planes.columns:
        tmp = df_planes[['plane', 'late_finish']].dropna().copy()
        tmp['plane'] = tmp['plane'].astype(int)
        tmp['LF_days'] = tmp['late_finish'].apply(lambda v: _to_days_from_base(v, base_date))
        late_finish_plane = dict(zip(tmp['plane'], tmp['LF_days']))

    # ---- Jobs (ids, orden) ----
    if 'job' not in df_case.columns or df_case['job'].isna().any():
        df_case['job'] = df_case.apply(lambda r: f"{int(r.plane)}-{int(r.task)}", axis=1)
    df_case = df_case.sort_values(['plane', 'task']).reset_index(drop=True)

    jobs = [str(r.job) for _, r in df_case.iterrows()]
    planes = [int(r.plane) for _, r in df_case.iterrows()]
    tasks = {jobs[i]: int(df_case.loc[i, 'task']) for i in range(len(jobs))}
    dur = {jobs[i]: float(df_case.loc[i, 'duration']) for i in range(len(jobs))}
    cli = {jobs[i]: str(df_case.loc[i, 'client']) for i in range(len(jobs))}
    plane_of = {jobs[i]: planes[i] for i in range(len(jobs))}
    clients = sorted(set(cli.values()))
    planes_set = sorted(set(planes))

    # ---- ES por job ----
    # Opción A (la que tenías activa): ES_j = early_start del avión
    ES_plane_days = {r: float(early_start_plane.get(r, 0.0)) for r in planes_set}
    es = {j: ES_plane_days[plane_of[j]] for j in jobs}

    # Opción B (si quieres que ES venga de case_261.date): DESCOMENTA estas 2 líneas y comenta la opción A
    # if 'date' in df_case.columns:
    #     es = {jobs[i]: float(_to_days_from_base(df_case.loc[i,'date'], base_date)) for i in range(len(jobs))}

    # ---- LF por job (prioridad: case.late > planes.late_finish > derivado) ----
    lf_row = None
    if 'late' in df_case.columns:
        lf_row = df_case['late'].apply(lambda v: _to_days_from_base(v, base_date)).tolist()

    big_pad = (max(dur.values()) if len(dur) > 0 else 1.0) * 10 + 100.0
    lf = {}
    for i, j in enumerate(jobs):
        r = plane_of[j]
        if lf_row is not None and not pd.isna(lf_row[i]):
            lf_j = float(lf_row[i])
        elif r in late_finish_plane:
            lf_j = float(late_finish_plane[r])
        else:
            lf_j = float(es[j] + dur[j] + big_pad)
        if es[j] + dur[j] > lf_j:  # garantizar ventana
            lf_j = es[j] + dur[j] + 1.0
        lf[j] = lf_j

    # ---- Ventanas por avión (para report/KPIs) ----
    ES_plane = {r: ES_plane_days.get(r, 0.0) for r in planes_set}
    LF_plane = {r: (late_finish_plane[r] if r in late_finish_plane else
                    (max(lf[j] for j in jobs if plane_of[j] == r) if any(plane_of[j] == r for j in jobs) else 0.0))
                for r in planes_set}

    # ---- Logs y checks ----
    print(f"Slots cargados: {len(jobs)}, Ejemplo: {[f'slot{i}' for i in range(min(3, len(jobs)))]}")
    print(f"Posiciones cargadas: {len(POSITIONS)}, Ejemplo: {POSITIONS[:3]}")
    ok_windows = all(es[j] + dur[j] <= lf[j] + 1e-9 for j in jobs)
    print("Todas las ventanas temporales son consistentes (early/late con holgura)." if ok_windows else
          "⚠️ ES+D > LF detectado; se ajustó LF al vuelo.")

    # ---- Precedencias ----
    preds = []
    for r in planes_set:
        jobs_r = [j for j in jobs if plane_of[j] == r]
        jobs_r.sort(key=lambda j: tasks[j])
        for k in range(len(jobs_r) - 1):
            j1, j2 = jobs_r[k], jobs_r[k + 1]
            if tasks[j1] < tasks[j2]:
                preds.append((j1, j2))

    # ---- Último job por avión ----
    last = {}
    for r in planes_set:
        jobs_r = [j for j in jobs if plane_of[j] == r]
        if not jobs_r: continue
        max_task = max(tasks[j] for j in jobs_r)
        for j in jobs_r:
            last[(j, r)] = 1 if tasks[j] == max_task else 0

    # ---- Cliente → aviones ----
    AOC = {}
    for c in clients:
        for r in planes_set:
            AOC[(c, r)] = 1 if any((plane_of[j] == r and cli[j] == c) for j in jobs) else 0

    # ---- Horizonte ----
    H_jobs = float(max(lf.values()) + max(dur.values()) + 10.0) if jobs else 100.0

    # ---- DEBUG puntual (útil para tu caso del avión 54) ----
    try:
        if any(j.startswith("54-") for j in jobs):
            j54 = [j for j in jobs if j.startswith("54-")][0]
            print(
                f"[DEBUG] BASE={base_date.date()}  ES(54)={es[j54]}  ES_date={(base_date + pd.to_timedelta(es[j54], 'D')).date()}  "
                f"LF(54)={lf[j54]}  LF_date={(base_date + pd.to_timedelta(lf[j54], 'D')).date()}")
    except Exception:
        pass

    # ---- Empaquetado ----
    global data
    data = {
        'JOBS': jobs,
        'PLANES': planes_set,
        'CLIENTS': clients,
        'POSITIONS': POSITIONS,
        'plane_of': plane_of,
        'client': cli,
        'task': tasks,
        'early': es, 'dur': dur, 'late': lf,  # ← días desde BASE_DATE
        'ES_plane': ES_plane, 'LF_plane': LF_plane,  # ← días desde BASE_DATE
        'AOC': AOC, 'LAST': last, 'PRED': preds,
        'H': H_jobs,
        'BASE_DATE': base_date,
        'INTERF_IN': INTERF_IN,
        'INTERF_OUT': INTERF_OUT
    }
    return data


# -----------------------
# MODELO
# -----------------------
def build_model(d,
                client_pos_policy: str = CLIENT_POS_POLICY,
                w_client_pos: float = W_CLIENT_POS,
                wms: float = W_MAKESPAN):
    m = ConcreteModel()

    # Conjuntos básicos
    m.J = Set(initialize=d['JOBS'], ordered=True)  # trabajos
    m.P = Set(initialize=d['POSITIONS'], ordered=True)  # posiciones
    m.C = Set(initialize=d['CLIENTS'], ordered=True)  # clientes
    m.R = Set(initialize=d['PLANES'], ordered=True)  # aviones (recursos)
    H = float(d['H'])

    # SLOTS (solo para métricas)
    SLOTS = int(min(max(1, math.ceil(H) + 10), 200))
    m.S = RangeSet(1, SLOTS)

    # Parámetros
    m.ES = Param(m.J, initialize=d['early'], within=Reals)
    m.D = Param(m.J, initialize=d['dur'], within=Reals)
    m.LF = Param(m.J, initialize=d['late'], within=Reals)
    m.H = Param(initialize=H, within=Reals)

    # per-plane windows
    m.ES_plane = Param(m.R, initialize=d['ES_plane'], within=Reals)
    m.LF_plane = Param(m.R, initialize=d['LF_plane'], within=Reals)

    # Mapeos
    m.clientOf = Param(m.J, initialize=d['client'], within=Any)
    m.planeOf = Param(m.J, initialize=d['plane_of'], within=Reals)

    # Precedencias y último trabajo
    m.PRED = Set(dimen=2, initialize=d['PRED'])  # (j1,j2)
    m.LAST = Param(m.J, m.R, initialize=d['LAST'], default=0, within=Reals)

    # Cliente → aviones
    m.AOC = Param(m.C, m.R, initialize=d['AOC'], default=0, within=Reals)

    # Variables de tiempo
    m.t_start = Var(m.J, within=NonNegativeReals, bounds=(0, H))
    m.t_end   = Var(m.J, within=NonNegativeReals, bounds=(0, H))

    # Asignación a posición
    m.y = Var(m.J, m.P, within=Binary)

    # Asignación única
    m.c_assign = Constraint(m.J, rule=lambda mdl, j: sum(mdl.y[j, p] for p in mdl.P) == 1)

    # Duración y ventanas por job
    m.c_dur    = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j] == mdl.t_start[j] + mdl.D[j])
    m.c_win_lo = Constraint(m.J, rule=lambda mdl, j: mdl.t_start[j] >= mdl.ES[j])
    m.c_win_hi = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j]   <= mdl.LF[j])

    # Ventanas por avión (aplicadas a todos sus trabajos)
    m.c_plane_es = Constraint(m.J, rule=lambda mdl, j: mdl.t_start[j] >= mdl.ES_plane[int(value(mdl.planeOf[j]))])
    m.c_plane_lf = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j]   <= mdl.LF_plane[int(value(mdl.planeOf[j]))])

    # Precedencias por avión (según tasks)
    m.c_pred = Constraint(m.PRED, rule=lambda mdl, j1, j2: mdl.t_start[j2] >= mdl.t_end[j1])

    # t_in/t_out por posición (para métricas/report)
    m.t_in  = Var(m.J, m.P, within=NonNegativeReals, bounds=(0, H))
    m.t_out = Var(m.J, m.P, within=NonNegativeReals, bounds=(0, H))

    bigM = H + max(value(m.D[j]) for j in m.J) + 100.0
    m.bigM = Param(initialize=float(bigM))

    def _link_in_lo(mdl, j, p):  return mdl.t_in[j, p]  >= mdl.t_start[j] - mdl.bigM * (1 - mdl.y[j, p])
    def _link_in_hi(mdl, j, p):  return mdl.t_in[j, p]  <= mdl.t_start[j] + mdl.bigM * (1 - mdl.y[j, p])
    def _link_out_lo(mdl, j, p): return mdl.t_out[j, p] >= mdl.t_end[j]   - mdl.bigM * (1 - mdl.y[j, p])
    def _link_out_hi(mdl, j, p): return mdl.t_out[j, p] <= mdl.t_end[j]   + mdl.bigM * (1 - mdl.y[j, p])

    m.c_link_in_lo  = Constraint(m.J, m.P, rule=_link_in_lo)
    m.c_link_in_hi  = Constraint(m.J, m.P, rule=_link_in_hi)
    m.c_link_out_lo = Constraint(m.J, m.P, rule=_link_out_lo)
    m.c_link_out_hi = Constraint(m.J, m.P, rule=_link_out_hi)

    # No SOLAPE por posición
    jobs = list(m.J); positions = list(m.P)
    idx_noover = [(j, k, p) for p in positions for j in jobs for k in jobs if j != k]
    m.IDX_NOOV = Set(dimen=3, initialize=idx_noover)
    m.w = Var(m.IDX_NOOV, within=Binary)  # w[j,k,p]=1 ⇒ j antes que k en p

    def c_noov1(mdl, j, k, p):
        return mdl.t_start[k] >= mdl.t_end[j] - mdl.bigM * (1 - mdl.w[j, k, p] + (1 - mdl.y[j, p]) + (1 - mdl.y[k, p]))
    def c_noov2(mdl, j, k, p):
        return mdl.t_start[j] >= mdl.t_end[k] - mdl.bigM * ( mdl.w[j, k, p] + (1 - mdl.y[j, p]) + (1 - mdl.y[k, p]))

    m.c_noov1 = Constraint(m.IDX_NOOV, rule=c_noov1)
    m.c_noov2 = Constraint(m.IDX_NOOV, rule=c_noov2)

    # -----------------------------
    # BLOQUEO (entrada/salida)
    # -----------------------------
    chain = {}
    for (pf, pb) in INTERF_IN:
        chain.setdefault(pb, []).append(pf)
    chain = {pb: [pf for pf in pfs if pf in positions] for pb, pfs in chain.items() if pb in positions}

    if chain:
        blk_idx = [(j, k, pb, pf) for pb, pfs in chain.items() for pf in pfs for j in m.J for k in m.J if j != k]
        m.IDX_BLK = Set(dimen=4, initialize=blk_idx)
        m.bEntry = Var(m.IDX_BLK, within=Binary)
        m.bExit  = Var(m.IDX_BLK, within=Binary)

        def _gating(mdl, j, k, pb, pf): return 2 - mdl.y[j, pb] - mdl.y[k, pf]

        def c_blk_entry_before(mdl, j, k, pb, pf):
            gating = _gating(mdl, j, k, pb, pf)
            return mdl.t_end[k] <= mdl.t_start[j] - SEP_IN + mdl.bigM * (1 - mdl.bEntry[j, k, pb, pf] + gating)
        def c_blk_entry_after(mdl, j, k, pb, pf):
            gating = _gating(mdl, j, k, pb, pf)
            return mdl.t_start[k] >= mdl.t_start[j] + SEP_IN - mdl.bigM * ( mdl.bEntry[j, k, pb, pf] + gating)
        def c_blk_exit_before(mdl, j, k, pb, pf):
            gating = _gating(mdl, j, k, pb, pf)
            return mdl.t_end[k] <= mdl.t_end[j] - SEP_OUT + mdl.bigM * (1 - mdl.bExit[j, k, pb, pf] + gating)
        def c_blk_exit_after(mdl, j, k, pb, pf):
            gating = _gating(mdl, j, k, pb, pf)
            return mdl.t_start[k] >= mdl.t_end[j] + SEP_OUT - mdl.bigM * ( mdl.bExit[j, k, pb, pf] + gating)

        m.c_blk_entry_before = Constraint(m.IDX_BLK, rule=c_blk_entry_before)
        m.c_blk_entry_after  = Constraint(m.IDX_BLK, rule=c_blk_entry_after)
        m.c_blk_exit_before  = Constraint(m.IDX_BLK, rule=c_blk_exit_before)
        m.c_blk_exit_after   = Constraint(m.IDX_BLK, rule=c_blk_exit_after)

    # -----------------------------
    # Variables “fc29-like” (métricas)
    # -----------------------------
    m.vPresence      = Var(m.S, m.P, m.R, within=Binary)  # (S,P,R)
    m.v01SwitchPlanes= Var(m.S, m.P, within=Binary)
    m.v01JobInSlot   = Var(m.S, m.P, m.J, within=Binary)
    m.v01Alpha       = Var(m.S, m.P, m.R, within=Binary)
    m.vIdle          = Var(m.S, m.R, within=Binary)

    # Retrasos avión/cliente
    m.vPlaneDelay  = Var(m.R, within=NonNegativeReals)
    m.vClientDelay = Var(m.C, within=NonNegativeReals)

    # Enlaces suaves métricos
    m.c_prs_hi = ConstraintList()
    m.c_prs_lo = ConstraintList()

    planes = list(m.R)
    for p in m.P:
        for r in m.R:
            Js = [j for j in m.J if int(value(m.planeOf[j])) == r]
            for s in m.S:
                m.c_prs_hi.add(m.vPresence[s, p, r] <= sum(m.y[j, p] for j in Js))
                for j in Js:
                    m.c_prs_lo.add(m.v01JobInSlot[s, p, j] <= m.y[j, p])

    first_s = value(m.S.first())
    for p in m.P:
        for s in m.S:
            if s == first_s: continue
            for r in m.R:
                m.add_component(f"c_alpha_pos_{p}_{s}_{r}_1",
                    Constraint(expr=m.v01Alpha[s, p, r] >= m.vPresence[s, p, r] - m.vPresence[s - 1, p, r]))
                m.add_component(f"c_alpha_pos_{p}_{s}_{r}_2",
                    Constraint(expr=m.v01Alpha[s, p, r] >= m.vPresence[s - 1, p, r] - m.vPresence[s, p, r]))
            m.add_component(f"c_sw_ge_alpha_{p}_{s}",
                Constraint(expr=m.v01SwitchPlanes[s, p] >=
                                sum(m.v01Alpha[s, p, r] for r in m.R) / max(1, len(planes))))

    for r in m.R:
        for s in m.S:
            m.add_component(f"c_idle_{r}_{s}",
                Constraint(expr=m.vIdle[s, r] >= 1 - sum(m.vPresence[s, p, r] for p in m.P)))

    # Retraso por avión (último trabajo)
    def _plane_delay_rule(mdl, r):
        return mdl.vPlaneDelay[r] >= sum((mdl.t_end[j] - mdl.LF_plane[r]) * mdl.LAST[j, r] for j in mdl.J)
    m.c_plane_delay = Constraint(m.R, rule=_plane_delay_rule)

    # Retraso por cliente = suma de retrasos de sus aviones
    def _client_delay_rule(mdl, c):
        return mdl.vClientDelay[c] == sum(mdl.vPlaneDelay[r] * mdl.AOC[c, r] for r in mdl.R)
    m.c_client_delay = Constraint(m.C, rule=_client_delay_rule)

    # -----------------------------
    # Política Cliente -> Posición (selector de dureza)
    # -----------------------------
    mode = str(client_pos_policy).lower().strip()

    # Variables necesarias según el modo
    if mode in {"soft12", "soft", "medium", "hard"}:
        m.z = Var(m.C, m.P, within=Binary)  # z[c,p] = 1 si c usa p
    else:
        m.z = None

    if mode in {"soft", "medium", "hard"}:
        m.viol_C = Var(m.C, within=NonNegativeReals)  # exceso de posiciones por cliente
    else:
        m.viol_C = None

    # Links fuertes (1.1 y 1.2) entre z y y
    if mode in {"soft12", "soft", "medium", "hard"}:
        m.c_link_yz = Constraint(m.J, m.P, rule=lambda mdl, j, p: mdl.y[j, p] <= mdl.z[mdl.clientOf[j], p])

        # sum_{j de c} y[j,p] >= z[c,p]   y   sum_{j de c} y[j,p] <= M_cp * z[c,p]
        m.c_z_le_sumy  = ConstraintList()
        m.c_sumy_le_Mz = ConstraintList()

        jobs_by_c = {c: [] for c in m.C}
        for j in m.J:
            cj_val = value(m.clientOf[j])
            for c in m.C:
                if value(c) == cj_val:
                    jobs_by_c[c].append(j)
                    break

        for c in m.C:
            Jc = jobs_by_c.get(c, [])
            if not Jc:
                continue
            M_cp = max(1, len(Jc))
            for p in m.P:
                m.c_z_le_sumy.add(sum(m.y[j, p] for j in Jc) >= m.z[c, p])
                m.c_sumy_le_Mz.add(sum(m.y[j, p] for j in Jc) <= M_cp * m.z[c, p])

    # Blando por exceso de posiciones por cliente
    if mode in {"soft", "medium", "hard"}:
        if not hasattr(m, "v_pos_over"):
            m.v_pos_over = Var(m.C, within=NonNegativeReals)
        def _pos_over(mdl, c):
            return mdl.v_pos_over[c] >= sum(mdl.z[c, p] for p in mdl.P) - 1
        m.c_pos_over = Constraint(m.C, rule=_pos_over)

    # Penalizar cambios de cliente dentro de una posición (índices relevantes solamente)
    if mode in {"medium", "hard"}:
        W_CLIENT_SWITCH_LOCAL = 1_000_000.0  # ajustable

        # Construir CHGPAIRS = {(j,k,p): j<k, clientes distintos, y (j,k,p) ∈ IDX_NOOV}
        w_index = set((j, k, p) for (j, k, p) in m.IDX_NOOV)
        valid_triplets = []
        for j in m.J:
            cj = value(m.clientOf[j])
            for k in m.J:
                if k <= j:
                    continue
                ck = value(m.clientOf[k])
                if cj == ck:
                    continue
                for p in m.P:
                    if (j, k, p) in w_index:
                        valid_triplets.append((j, k, p))

        if hasattr(m, "CHGPAIRS"):
            m.del_component(m.CHGPAIRS)
        m.CHGPAIRS = Set(dimen=3, initialize=valid_triplets, ordered=False)

        if hasattr(m, "client_change"):
            m.del_component(m.client_change)
        m.client_change = Var(m.CHGPAIRS, within=Binary)

        if hasattr(m, "c_link_change"):
            m.del_component(m.c_link_change)
        m.c_link_change = ConstraintList()
        for (j, k, p) in m.CHGPAIRS:
            m.c_link_change.add(m.client_change[j, k, p] >= m.w[j, k, p] - (1 - m.y[j, p]) - (1 - m.y[k, p]))
            m.c_link_change.add(m.client_change[j, k, p] <= m.w[j, k, p])
            m.c_link_change.add(m.client_change[j, k, p] <= m.y[j, p])
            m.c_link_change.add(m.client_change[j, k, p] <= m.y[k, p])
    else:
        W_CLIENT_SWITCH_LOCAL = 0.0

    # Exclusividad por posición con slack por posición (modo "hard")
    if mode == "hard":
        if hasattr(m, "u_pos"):
            m.del_component(m.u_pos)
        m.u_pos = Var(m.P, within=Binary)

        if hasattr(m, "c_pos_exclusive_soft"):
            m.del_component(m.c_pos_exclusive_soft)
        m.c_pos_exclusive_soft = Constraint(m.P, rule=lambda mdl, p:
            sum(mdl.z[c, p] for c in mdl.C) <= 1 + mdl.u_pos[p])

        W_POS_MIX_LOCAL = 20_000_000.0
    else:
        W_POS_MIX_LOCAL = 0.0

    # Makespan
    m.Tmax = Var(within=NonNegativeReals, bounds=(0, H))
    m.c_mks = Constraint(m.J, rule=lambda mdl, j: mdl.Tmax >= mdl.t_end[j])

    # Objetivo
    obj_terms = []
    obj_terms.append(W_JOBINSLOT * sum(m.v01JobInSlot[s, p, j] for s in m.S for p in m.P for j in m.J))
    obj_terms.append(W_ALPHA     * sum(m.v01Alpha[s, p, r] for s in m.S for p in m.P for r in m.R))
    obj_terms.append(W_SWITCH    * sum(m.v01SwitchPlanes[s, p] for s in m.S for p in m.P))
    obj_terms.append(W_PRESENCE  * sum(m.vPresence[s, p, r] for s in m.S for p in m.P for r in m.R))
    obj_terms.append(W_CLIENT_DELAY * sum(m.vClientDelay[c] for c in m.C))
    obj_terms.append(W_IDLE      * sum(m.vIdle[s, r] for s in m.S for r in m.R))

    # Costes de la política según modo
    if mode in {"soft", "medium", "hard"} and (m.viol_C is not None):
        obj_terms.append(w_client_pos * sum(m.viol_C[c] for c in m.C))
    if mode in {"medium", "hard"}:
        obj_terms.append(W_CLIENT_SWITCH_LOCAL * sum(m.client_change[j, k, p] for (j, k, p) in m.CHGPAIRS))
    if mode == "hard":
        obj_terms.append(W_POS_MIX_LOCAL * sum(m.u_pos[p] for p in m.P))

    if wms and wms > 0:
        obj_terms.append(wms * m.Tmax)

    m.OBJ = Objective(expr=sum(obj_terms), sense=minimize)
    return m

# -----------------------
# PLOT GANTT (ordenado)
# -----------------------
def plot_gantt_by_position(df_pos, base_date, color_by='client', html_path="schedule_enhanced.html"):
    import plotly.express as px
    d = df_pos.copy()
    d['Start'] = d['start_dt']
    d['Finish'] = d['finish_dt']
    d['Position'] = d['p']
    if color_by not in d.columns:
        color_by = 'client'
    fig = px.timeline(
        d, x_start="Start", x_end="Finish",
        y="Position", color=color_by, hover_data=["job", "plane", "client"]
    )
    # Orden ascendente: position1, position2, ... (NO invertido)
    order_positions = POSITIONS
    fig.update_yaxes(categoryorder="array", categoryarray=order_positions)
    fig.update_layout(title="Schedule por posición (ordenado)", legend_title="Cliente")
    fig.write_html(html_path)


# -----------------------
# REPORT ENRIQUECIDO
# -----------------------

def generate_report(df_pos: pd.DataFrame, d: dict, movimientos=None):
    """Reporte largo con resumen por avión y detalle por trabajo, reconstr. de movimientos.
    Exporta CSVs: report_resumen_por_avion.csv, report_detalle_trabajos.csv
    """
    base_date = d['BASE_DATE']
    ES = d['early'];
    D = d['dur'];
    LF = d['late'];
    CL = d['client'];
    plane_of = d['plane_of']
    ESpl = d.get('ES_plane', {})
    LFpl = d.get('LF_plane', {})

    df = df_pos.copy()
    if 'start_dt' not in df.columns:
        df['start_dt'] = pd.to_timedelta(df['start'], unit='D') + base_date
    if 'finish_dt' not in df.columns:
        df['finish_dt'] = pd.to_timedelta(df['finish'], unit='D') + base_date

    # --- RESUMEN POR AVIÓN ---
    resumen = []
    for plane, g in df.groupby('plane'):
        trabajos = ','.join(sorted(map(str, g['job'].unique())))
        posiciones = ','.join(sorted(map(str, g['p'].unique())))
        clientes = ','.join(sorted(map(str, g['client'].unique())))
        es_pl = ESpl.get(plane, 0.0)
        lf_pl = LFpl.get(plane, max(LF.values()) if len(LF) else 0.0)
        resumen.append({
            'Avión': plane,
            'Clientes': clientes if clientes else 'NA',
            'ES_plane': (base_date + pd.to_timedelta(es_pl, unit='D')).date(),
            'LF_plane': (base_date + pd.to_timedelta(lf_pl, unit='D')).date(),
            'Primer Inicio': g['start_dt'].min().date(),
            'Fin': g['finish_dt'].max().date(),
            'Trabajos': trabajos,
            'Posiciones': posiciones,
            'Movimientos(E/S)': 2 * len(g)
        })
    df_res = pd.DataFrame(resumen).sort_values('Avión')

    # --- DETALLE POR TRABAJO ---
    det = []
    for _, r in df.sort_values(['plane', 'start']).iterrows():
        j = r['job']
        esj = ES.get(j, 0.0);
        dj = D.get(j, r['finish'] - r['start']);
        lfj = LF.get(j, r['finish'])
        det.append({
            '✔': 'OK',
            'Avión': plane_of[j],
            'Trabajo': j,
            'Cliente': CL[j],
            'Posición': r['p'],
            'Fecha ES': (base_date + pd.to_timedelta(esj, unit='D')).date(),
            'Fecha LF': (base_date + pd.to_timedelta(lfj, unit='D')).date(),
            'Inicio real': r['start_dt'].date(),
            'Fin real': r['finish_dt'].date(),
            'Dur Est.(d)': dj,
            'Dur Real(d)': (r['finish'] - r['start']),
            'Retraso(d)': max(0.0, (r['finish'] - lfj)),
            'Holgura a ES(d)': max(0.0, r['start'] - esj),
            'Holgura a LF(d)': max(0.0, lfj - r['finish'])
        })
    df_det = pd.DataFrame(det).sort_values(['Avión', 'Inicio real'])

    # --- PRINT bonito ---
    print("" + "=" * 150)
    print("RESUMEN POR AVIÓN")
    print("=" * 150)
    if not df_res.empty:
        print(df_res.to_string(index=False, col_space=12))

    print("" + "=" * 170)
    print("DETALLE DE TODOS LOS TRABAJOS")
    print("=" * 170)
    if not df_det.empty:
        print(df_det.to_string(index=False, col_space=10))

    # --- CSVs del report ---
    try:
        df_res.to_csv("report_resumen_por_avion.csv", index=False)
        df_det.to_csv("report_detalle_trabajos.csv", index=False)
        print("Report CSVs: report_resumen_por_avion.csv, report_detalle_trabajos.csv")
    except Exception as e:
        print(f"No se pudieron guardar los CSV del report: {e}")


# -----------------------
# VALIDACIÓN
# -----------------------

def validate_solution(model, df_pos, INTERF_IN, INTERF_OUT):
    ok = True
    for p, grp in df_pos.groupby('p'):
        g = grp.sort_values('start');
        prev = -1e-9
        for _, r in g.iterrows():
            if r['start'] < prev - 1e-9:
                print(f"❌ Solape en {p}: {r['job']}")
                ok = False
            prev = r['finish']
    print("✅ Sanity check posiciones:", "sin solapes." if ok else "⚠ solapes")

    # Validación bloqueo (instantes E/S)
    issues = []
    df = df_pos.copy()
    pos_intervals = {
        p: df[df['p'] == p][['job', 'start', 'finish']].to_records(index=False)
        for p in df['p'].unique()
    }
    # Entrada
    for (pf, pb) in INTERF_IN:
        if pf not in pos_intervals or pb not in pos_intervals: continue
        for _, r in df[df['p'] == pb].iterrows():
            t_in = r['start']
            for (jobk, sk, fk) in pos_intervals[pf]:
                if sk - 1e-9 <= t_in <= fk + 1e-9:
                    issues.append(f"Bloqueo ENTRADA: {r['job']}@{pb} con {jobk}@{pf} (t_in dentro de [{sk},{fk}])")
    # Salida
    for (pf, pb) in INTERF_OUT:
        if pf not in pos_intervals or pb not in pos_intervals: continue
        for _, r in df[df['p'] == pb].iterrows():
            t_out = r['finish']
            for (jobk, sk, fk) in pos_intervals[pf]:
                if sk - 1e-9 <= t_out <= fk + 1e-9:
                    issues.append(f"Bloqueo SALIDA: {r['job']}@{pb} con {jobk}@{pf} (t_out dentro de [{sk},{fk}])")

    print("=== VALIDACIÓN POST-SOLUCIÓN ===")
    if issues:
        for s in sorted(set(issues)):
            print("❌ " + s)
        print("⚠️  Validación con incidencias")
    else:
        print("✅ Validación OK")


# -----------------------
# SOLVER + REPORT
# -----------------------

def solve_and_report(model, xlsx_path, case_sheet, timelimit=1500, mipgap=GAP):
    print("Iniciando resolución con Gurobi.")
    opt = SolverFactory('gurobi')
    base_opts = {
        "TimeLimit": timelimit, "MIPGap": mipgap, "Heuristics": 1, "RINS": 10,
        "MIPFocus": 3, "Cuts": 2, "Presolve": 2, "ImproveStartGap": 0.5,
        "NoRelHeurTime": 60, "VarBranch": 1, "BranchDir": -1, "MinRelNodes": 1000,
        "OutputFlag": 1, "LogToConsole": 1, "DisplayInterval": 1,
    }
    opt.options.update(base_opts)

    res = opt.solve(model, tee=True, load_solutions=True)
    if res.solver.termination_condition not in (TerminationCondition.optimal, TerminationCondition.feasible):
        print("WARNING:", res.solver.status, res.solver.termination_condition)
        model.write("case_conflict.mps")
        return None

    BASE = data['BASE_DATE']
    rows, movimientos = [], []
    for j in model.J:
        pos = next((p for p in model.P if value(model.y[j, p]) > 0.5), None)
        if pos is None: continue
        st = float(value(model.t_start[j]));
        en = float(value(model.t_end[j]))
        tin = float(value(model.t_in[j, pos]));
        tout = float(value(model.t_out[j, pos]))
        plane = int(value(model.planeOf[j]))
        client = str(value(model.clientOf[j]))
        rows.append({'plane': plane, 'job': j, 'p': pos, 'type': 'work', 'start': st, 'finish': en, 'client': client})
        movimientos.append((plane, pos, 'IN', tin));
        movimientos.append((plane, pos, 'OUT', tout))

    df_pos = pd.DataFrame(rows).sort_values(['p', 'start', 'plane']).reset_index(drop=True)
    df_pos['start_dt'] = df_pos['start'].map(lambda d: BASE + timedelta(days=float(d)))
    df_pos['finish_dt'] = df_pos['finish'].map(lambda d: BASE + timedelta(days=float(d)))

    ok = True
    for p, grp in df_pos.groupby('p'):
        g = grp.sort_values('start');
        prev = -1e9
        for _, r in g.iterrows():
            if r['start'] < prev - 1e-6: ok = False; break
            prev = r['finish']
    print("✅ Sanity check posiciones:", "sin solapes." if ok else "⚠ solapes")

    # CSV básicos
    df_pos.to_csv("solution_by_position.csv", index=False)
    df_pos[['plane', 'job', 'p', 'start', 'finish', 'client']].to_csv("solution_jobs.csv", index=False)
    kpis = []
    for p, grp in df_pos.groupby('p'):
        kpis.append(
            {'position': p, 'jobs': len(grp), 'start_min': grp['start'].min(), 'finish_max': grp['finish'].max()})
    pd.DataFrame(kpis).to_csv("solution_kpis.csv", index=False)
    print("CSV exportados: solution_by_position.csv, solution_jobs.csv, solution_kpis.csv")

    # Gantt
    try:
        plot_gantt_by_position(df_pos, base_date=BASE, color_by='client', html_path="schedule_enhanced.html")
        print("Plot HTML exportado: schedule_enhanced.html")
    except Exception as e:
        print(f"No se pudo generar el Gantt HTML: {e}")

    # Report + validación
    try:
        generate_report(df_pos, d=data, movimientos=movimientos)
    except Exception as e:
        print(f"Reporte parcial: {e}")
    try:
        validate_solution(model, df_pos, data['INTERF_IN'], data['INTERF_OUT'])
    except Exception as e:
        print(f"Validador: {e}")

    fact_check_solution(model, data, tol=1e-6)

    return df_pos


from pyomo.core import Constraint, Var
from pyomo.environ import value
import math
import pandas as pd


def _get_bound(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        try:
            return float(value(v)) if v is not None else None
        except Exception:
            return None


def fact_check_solution(model, data, tol=1e-6, csv_prefix="violations"):
    """
    Verifica:
      A) TODAS las restricciones activas de Pyomo (primal feasibility).
      B) Cotas de variables e integralidad de binarias.
      C) Chequeos derivados de la lógica del modelo (no-solape, bloqueos, etc.).

    Crea CSVs con los detalles y saca un resumen por consola.
    """
    print("\n==================== FACT CHECK: PRIMAL FEASIBILITY ====================")

    # ----------------------------------------------------------------------
    # A) Recorrido genérico sobre todas las restricciones activas
    # ----------------------------------------------------------------------
    con_viol = []
    total_cons = 0
    worst_violation = 0.0

    for comp in model.component_objects(Constraint, active=True):
        cname = comp.name
        for idx in comp:
            c = comp[idx]
            if (c.body is None):
                continue
            try:
                val = float(value(c.body))
            except Exception:
                # No evaluable (raro), lo marcamos
                con_viol.append({
                    "constraint": cname, "index": str(idx),
                    "lb": None, "value": None, "ub": None,
                    "type": "EVAL_ERROR", "violation": float("inf")
                })
                continue

            lb = _get_bound(c.lower) if c.has_lb() else -float("inf")
            ub = _get_bound(c.upper) if c.has_ub() else float("inf")

            v_lb = (lb - val) if (val < lb - tol) else 0.0
            v_ub = (val - ub) if (val > ub + tol) else 0.0

            if v_lb > 0 or v_ub > 0:
                viol = max(v_lb, v_ub)
                worst_violation = max(worst_violation, viol)
                con_viol.append({
                    "constraint": cname, "index": str(idx),
                    "lb": lb, "value": val, "ub": ub,
                    "type": ("LB" if v_lb > v_ub else "UB"),
                    "violation": viol
                })
            total_cons += 1

    # ----------------------------------------------------------------------
    # B) Cotas de variables e integralidad de binarias
    # ----------------------------------------------------------------------
    var_viol = []
    total_vars = 0
    worst_var_violation = 0.0

    for vcomp in model.component_objects(Var, active=True):
        vname = vcomp.name
        for v in vcomp.values():
            total_vars += 1
            try:
                x = float(value(v))
            except Exception:
                var_viol.append({
                    "var": vname, "index": str(v.index()),
                    "lb": None, "value": None, "ub": None,
                    "type": "EVAL_ERROR", "violation": float("inf")
                })
                continue

            lb = _get_bound(v.lb)
            ub = _get_bound(v.ub)

            if lb is not None and x < lb - tol:
                diff = (lb - x)
                worst_var_violation = max(worst_var_violation, diff)
                var_viol.append({
                    "var": vname, "index": str(v.index()),
                    "lb": lb, "value": x, "ub": ub,
                    "type": "LB", "violation": diff
                })

            if ub is not None and x > ub + tol:
                diff = (x - ub)
                worst_var_violation = max(worst_var_violation, diff)
                var_viol.append({
                    "var": vname, "index": str(v.index()),
                    "lb": lb, "value": x, "ub": ub,
                    "type": "UB", "violation": diff
                })

            # Integralidad aproximada para binarias
            try:
                if v.is_binary():
                    dist0 = abs(x - 0.0)
                    dist1 = abs(x - 1.0)
                    if min(dist0, dist1) > 1e-4:  # tolerancia de integridad
                        var_viol.append({
                            "var": vname, "index": str(v.index()),
                            "lb": lb, "value": x, "ub": ub,
                            "type": "BINARY_INTEGRALITY", "violation": min(dist0, dist1)
                        })
                        worst_var_violation = max(worst_var_violation, min(dist0, dist1))
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # C) Chequeos derivados específicos del modelo (para entender incidencias)
    # ----------------------------------------------------------------------
    drv_rows = []

    # 1) Asignación única (∑ y[j,p] = 1)
    if hasattr(model, "y"):
        for j in model.J:
            s = sum(float(value(model.y[j, p])) for p in model.P)
            if abs(s - 1.0) > 1e-4:
                drv_rows.append({
                    "check": "assign_unique", "entity": str(j),
                    "lhs": s, "rel": "==", "rhs": 1.0, "violation": abs(s - 1.0)
                })

    # 2) Ventanas y duración
    for j in model.J:
        ES = float(value(model.ES[j]));
        D = float(value(model.D[j]));
        LF = float(value(model.LF[j]))
        ts = float(value(model.t_start[j]));
        te = float(value(model.t_end[j]))
        if ts < ES - tol:
            drv_rows.append(
                {"check": "time_window_low", "entity": str(j), "lhs": ts, "rel": ">=", "rhs": ES, "violation": ES - ts})
        if te > LF + tol:
            drv_rows.append({"check": "time_window_high", "entity": str(j), "lhs": te, "rel": "<=", "rhs": LF,
                             "violation": te - LF})
        if abs((ts + D) - te) > tol:
            drv_rows.append({"check": "duration", "entity": str(j), "lhs": ts + D, "rel": "==", "rhs": te,
                             "violation": abs((ts + D) - te)})

    # 3) No solape por posición (ordenación por tiempo) — chequeo directo
    #    (Además de las z[j,k,p], verificamos físicamente que no se solapan)
    if hasattr(model, "y"):
        # reconstruimos asignaciones p* para cada job (argmax de y)
        def _argmax_position(j):
            best_p, best_v = None, -1.0
            for p in model.P:
                v = float(value(model.y[j, p]))
                if v > best_v:
                    best_v, best_p = v, p
            return best_p

        per_pos = {p: [] for p in model.P}
        for j in model.J:
            p = _argmax_position(j)
            ts = float(value(model.t_start[j]));
            te = float(value(model.t_end[j]))
            per_pos[p].append((ts, te, j))

        for p, blocks in per_pos.items():
            blocks.sort(key=lambda t: t[0])
            prev_end = -1e18
            for (ts, te, j) in blocks:
                # solape si comienzo antes del fin anterior
                if ts < prev_end - tol:
                    drv_rows.append({
                        "check": "no_overlap_by_pos", "entity": f"{j}@{p}",
                        "lhs": ts, "rel": ">=", "rhs": prev_end, "violation": prev_end - ts
                    })
                prev_end = max(prev_end, te)

    # 4) Bloqueos IN/OUT (separación en t_start y t_end)
    #    Usamos conjuntos E_in/E_out si existen
    epsIN = float(value(model.epsIN)) if hasattr(model, "epsIN") else 0.0
    epsOUT = float(value(model.epsOUT)) if hasattr(model, "epsOUT") else 0.0

    if hasattr(model, "E_in") and hasattr(model, "E_out") and hasattr(model, "y"):
        # Map job -> pos y tiempos
        def _p_of(j):
            best_p, best_v = None, -1.0
            for p in model.P:
                v = float(value(model.y[j, p]))
                if v > best_v:
                    best_v, best_p = v, p
            return best_p

        starts_by_pos = {p: [] for p in model.P}
        ends_by_pos = {p: [] for p in model.P}
        for j in model.J:
            p = _p_of(j)
            ts = float(value(model.t_start[j]));
            te = float(value(model.t_end[j]))
            starts_by_pos[p].append((ts, j));
            ends_by_pos[p].append((te, j))

        # IN: no pueden coincidir entradas en posiciones en conflicto
        for (a, b) in model.E_in:
            A = starts_by_pos.get(a, []);
            B = starts_by_pos.get(b, [])
            for tsA, jA in A:
                for tsB, jB in B:
                    if abs(tsA - tsB) < epsIN - tol:
                        drv_rows.append({
                            "check": "block_IN", "entity": f"{jA}@{a} vs {jB}@{b}",
                            "lhs": abs(tsA - tsB), "rel": ">=", "rhs": epsIN, "violation": epsIN - abs(tsA - tsB)
                        })

        # OUT: no pueden coincidir salidas en posiciones en conflicto
        for (a, b) in model.E_out:
            A = ends_by_pos.get(a, []);
            B = ends_by_pos.get(b, [])
            for teA, jA in A:
                for teB, jB in B:
                    if abs(teA - teB) < epsOUT - tol:
                        drv_rows.append({
                            "check": "block_OUT", "entity": f"{jA}@{a} vs {jB}@{b}",
                            "lhs": abs(teA - teB), "rel": ">=", "rhs": epsOUT, "violation": epsOUT - abs(teA - teB)
                        })

    # 5) Política cliente→posición (soft/hard)
    #    y[j,p] ≤ x[c,p]    y   x[c,p] ≤ sum_{j de c} y[j,p]
    #    y en {0,1}; si hard: Σ_p x[c,p] ≤ 1; si soft: vExtraPos[c] ≥ Σ_p x[c,p] - 1
    if hasattr(model, "x"):
        # c y p desde el modelo
        C = list(model.C.data())
        P = list(model.P.data())

        # y<=x
        for j in model.J:
            c = str(value(model.clientOf[j]))
            for p in model.P:
                yjp = float(value(model.y[j, p]))
                xcp = float(value(model.x[c, p])) if (c in C and p in P) else 0.0
                if yjp > xcp + 1e-6:
                    drv_rows.append({
                        "check": "policy_y_le_x", "entity": f"y[{j},{p}]<=x[{c},{p}]",
                        "lhs": yjp, "rel": "<=", "rhs": xcp, "violation": yjp - xcp
                    })

        # x<=sum_y (solo sentido de fortalecimiento)
        for c in model.C:
            for p in model.P:
                xcp = float(value(model.x[c, p]))
                sumy = sum(float(value(model.y[j, p])) for j in model.J if str(value(model.clientOf[j])) == str(c))
                if xcp > sumy + 1e-6:
                    drv_rows.append({
                        "check": "policy_x_le_sumy", "entity": f"x[{c},{p}]<=sum_y[{c},{p}]",
                        "lhs": xcp, "rel": "<=", "rhs": sumy, "violation": xcp - sumy
                    })

        # hard: Σ_p x[c,p] ≤ 1  (si tu modelo lo activa)
        if hasattr(model, "c_onepos_hard"):
            for c in model.C:
                sx = sum(float(value(model.x[c, p])) for p in model.P)
                if sx > 1.0 + 1e-6:
                    drv_rows.append({
                        "check": "policy_onepos_hard", "entity": f"sum_p x[{c},p]<=1",
                        "lhs": sx, "rel": "<=", "rhs": 1.0, "violation": sx - 1.0
                    })

        # soft: vExtraPos[c] ≥ Σ_p x[c,p] - 1
        if hasattr(model, "vExtraPos"):
            for c in model.C:
                sx = sum(float(value(model.x[c, p])) for p in model.P)
                vE = float(value(model.vExtraPos[c]))
                rhs = sx - 1.0
                if vE + tol < rhs:
                    drv_rows.append({
                        "check": "policy_extra_pos_soft", "entity": f"vExtraPos[{c}]>=sum_x-1",
                        "lhs": vE, "rel": ">=", "rhs": rhs, "violation": (rhs - vE)
                    })

    # ----------------------------------------------------------------------
    # Salida y resumen
    # ----------------------------------------------------------------------
    df_cons = pd.DataFrame(con_viol)
    df_vars = pd.DataFrame(var_viol)
    df_drv = pd.DataFrame(drv_rows)

    if not df_cons.empty:
        df_cons.sort_values("violation", ascending=False).to_csv(f"{csv_prefix}_constraints.csv", index=False)
    if not df_vars.empty:
        df_vars.sort_values("violation", ascending=False).to_csv(f"{csv_prefix}_variable_bounds.csv", index=False)
    if not df_drv.empty:
        df_drv.sort_values("violation", ascending=False).to_csv(f"{csv_prefix}_derived_checks.csv", index=False)

    print("\n--- Resumen ---")
    print(f"Restricciones revisadas: {total_cons:,}")
    print(f"• Violaciones de restricciones: {len(con_viol):,} | peor = {worst_violation:.3e}")
    print(f"• Violaciones de variables:     {len(var_viol):,} | peor = {worst_var_violation:.3e}")
    print(f"• Incidencias derivadas:        {len(drv_rows):,}")

    if not df_cons.empty:
        top = df_cons.sort_values("violation", ascending=False).head(10)
        p


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    XLSX = "input_data.xlsx"
    SHEET = "case_261"
    d = read_input(XLSX, SHEET, planning_start=PLANNING_START)
    model = build_model(d,
                        client_pos_policy=CLIENT_POS_POLICY,
                        w_client_pos=W_CLIENT_POS,
                        wms=W_MAKESPAN)
    solve_and_report(model, XLSX, SHEET, timelimit=1500, mipgap=GAP)
