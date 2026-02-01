#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Runner para el modelo aircraft-positioning.

- Itera sobre ficheros .xlsx (single-sheet "case") generados por tu scenario_maker.
- Cruza modos de política (soft12, soft, medium, hard) con presets del solver (base, fast, feasible).
- Lanza el solver con timelimit/gap y captura KPIs de rendimiento.
- Escribe un CSV con una fila por (escenario, modo, preset).

Requisitos en tu módulo de modelo (p.ej. model_func1_sh.py):
  - variable global POSITIONS (lista de strings) -> la inyectamos desde CLI (--n-positions o --positions)
  - función read_case_single_sheet(xlsx_path, sheet_name="case", planning_start=...)
  - función build_model(d, client_pos_policy, w_client_pos, wms)
"""

import argparse
import csv
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pyomo.environ as pyo
import importlib


# -----------------------
# Utils
# -----------------------

def parse_positions_arg(positions_csv: Optional[str], n_positions: Optional[int], prefix: str) -> List[str]:
    if positions_csv:
        return [s.strip() for s in positions_csv.split(",") if s.strip()]
    if n_positions:
        return [f"{prefix}{i}" for i in range(1, int(n_positions) + 1)]
    raise ValueError("You must provide either --positions or --n-positions.")


def find_scenarios(scenarios_dir: str, pattern: str) -> List[Path]:
    base = Path(scenarios_dir)
    if not base.exists():
        raise FileNotFoundError(f"Scenarios directory not found: {base}")
    return sorted(base.glob(pattern))


def solver_factory(name: str):
    solver = pyo.SolverFactory(name)
    if not solver.available(False):
        raise RuntimeError(f"Solver '{name}' is not available in this environment.")
    return solver


def safe_val(v) -> float:
    """Devuelve 0.0 si la variable no está inicializada (value=None) en vez de romper."""
    try:
        if hasattr(v, "value"):
            return float(v.value) if v.value is not None else 0.0
        return float(pyo.value(v))
    except Exception:
        return 0.0


def sum_indexed(var_like) -> float:
    """Suma valores de un IndexedVar sin evaluar los None (evita warnings/errores)."""
    total = 0.0
    try:
        for v in var_like.values():
            val = getattr(v, "value", None)
            if val is not None:
                total += float(val)
        return total
    except Exception:
        pass
    try:
        for idx in var_like:
            v = var_like[idx]
            val = getattr(v, "value", None)
            if val is not None:
                total += float(val)
    except Exception:
        pass
    return total


def _extract_solver_mipgap(results) -> Optional[float]:
    """Intenta extraer el gap del objeto results del solver."""
    # ruta solver
    try:
        g = results.solver.get("gap", None)
        if g is not None:
            return float(g)
    except Exception:
        pass
    # ruta problem (algunos plugins)
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("MIPGap", "mipgap", "gap"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def _extract_best_bound(results) -> Optional[float]:
    # ruta solver
    try:
        bb = results.solver.get("best_bound", None)
        if bb is not None:
            return float(bb)
    except Exception:
        pass
    # ruta problem
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("Lower bound", "lower_bound", "Best bound", "best_bound"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def objective_from_results(results) -> Optional[float]:
    """Intenta recuperar el valor del objetivo desde results (sin evaluar el objetivo simbólico)."""
    # ruta solver
    try:
        obj = results.solver.get("objective", None)
        if obj is not None:
            return float(obj)
    except Exception:
        pass
    # ruta problem
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("Upper bound", "upper_bound", "Objective", "objective"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def apply_preset_options(solver, solver_name: str, preset: str, timelimit: Optional[float], mipgap: Optional[float]):
    """
    Ajusta opciones del solver según preset. Implementado para Gurobi; otros solvers ignoran lo desconocido.
    Presets:
      - base: opciones por defecto + TimeLimit/MIPGap si se pasan
      - fast: prioriza velocidad (MIPFocus=1, heurísticas algo más activas)
      - feasible: prioriza factibilidad rápida (MIPFocus=3, heurísticas más altas)
    """
    sname = solver_name.lower()
    opts = solver.options

    # Timelimit / gap genéricos por solver
    try:
        if sname.startswith("gurobi"):
            if timelimit is not None and timelimit > 0:
                opts["TimeLimit"] = float(timelimit)
            if mipgap is not None:
                opts["MIPGap"] = float(mipgap)
        elif sname.startswith("cplex"):
            if timelimit is not None and timelimit > 0:
                opts["timelimit"] = float(timelimit)
            if mipgap is not None:
                opts["mip tolerances mipgap"] = float(mipgap)
        elif sname == "cbc":
            if timelimit is not None and timelimit > 0:
                opts["seconds"] = float(timelimit)
            if mipgap is not None:
                opts["ratioGap"] = float(mipgap)
        else:
            if timelimit is not None and timelimit > 0:
                opts["timelimit"] = float(timelimit)
            if mipgap is not None:
                opts["mipgap"] = float(mipgap)
    except Exception:
        pass

    # Presets específicos (Gurobi)
    if not sname.startswith("gurobi"):
        return

    preset = (preset or "base").lower().strip()
    if preset == "base":
        # nada especial
        return
    elif preset == "fast":
        # inclina a búsqueda rápida de incumbents
        opts["MIPFocus"] = 1
        opts["Heuristics"] = 0.2
        opts["Cuts"] = 1
        opts["Presolve"] = 2
        # opcional: nodeselect para explorar incumbents rápido
        # opts["NodeMethod"] = 1
    elif preset == "feasible":
        # prioriza factibilidad
        opts["MIPFocus"] = 3
        opts["Heuristics"] = 0.5
        opts["Cuts"] = 1
        opts["Presolve"] = 2
    # cualquier otro texto se ignora


# -----------------------
# Core
# -----------------------

def build_and_solve(model_mod,
                    xlsx_path: Path,
                    sheet_name: str,
                    mode: str,
                    w_client_pos: float,
                    wms: float,
                    solver_name: str,
                    timelimit: Optional[float],
                    mipgap: Optional[float],
                    preset: str) -> Dict[str, Any]:
    """Lee datos, construye el modelo, resuelve y devuelve KPIs."""
    # 1) Leer datos del escenario (NO pasar 'positions' al lector)
    d = model_mod.read_case_single_sheet(
        str(xlsx_path),
        sheet_name=sheet_name,
        planning_start=getattr(model_mod, "PLANNING_START", None)
    )

    # 2) Construir modelo
    m = model_mod.build_model(d, client_pos_policy=mode, w_client_pos=w_client_pos, wms=wms)

    # 3) Solver + presets + opciones
    solver = solver_factory(solver_name)
    apply_preset_options(solver, solver_name, preset, timelimit, mipgap)

    # Silenciar Pyomo para evitar ruido de evaluación
    logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    # 4) Resolver y medir tiempo
    t0 = time.perf_counter()
    results = solver.solve(m, tee=False)
    t1 = time.perf_counter()
    solve_time = t1 - t0

    # 5) Estado/gap/objetivo/bound
    term = str(getattr(results.solver, "termination_condition", ""))
    status = str(getattr(results.solver, "status", ""))

    mipgap_out = _extract_solver_mipgap(results)
    best_bound = _extract_best_bound(results)
    objective_val = objective_from_results(results)

    # 6) KPIs del modelo (robustos a variables no inicializadas)
    metrics = {
        "status": status,
        "termination": term,
        "solve_time_s": round(solve_time, 3),
        "mipgap": mipgap_out,
        "best_bound": best_bound,
        "objective": objective_val,
        "Tmax": safe_val(m.Tmax) if hasattr(m, "Tmax") else None,
        "client_delay_sum": sum_indexed(m.vClientDelay) if hasattr(m, "vClientDelay") else None,
        "viol_C_sum": sum_indexed(m.viol_C) if hasattr(m, "viol_C") else None,
        "client_switches": sum_indexed(m.client_change) if hasattr(m, "client_change") else None,
        "positions_slack_used": sum_indexed(m.u_pos) if hasattr(m, "u_pos") else None,
        "switches_slot_sum": sum_indexed(m.v01SwitchPlanes) if hasattr(m, "v01SwitchPlanes") else None,
        "presence_sum": sum_indexed(m.vPresence) if hasattr(m, "vPresence") else None,
        "idle_sum": sum_indexed(m.vIdle) if hasattr(m, "vIdle") else None,
        "n_jobs": len(list(m.J)) if hasattr(m, "J") else None,
        "n_planes": len(list(m.R)) if hasattr(m, "R") else None,
        "n_clients": len(list(m.C)) if hasattr(m, "C") else None,
        "n_positions": len(list(m.P)) if hasattr(m, "P") else None,
        "H": float(pyo.value(m.H)) if hasattr(m, "H") else None,
    }
    return metrics


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Batch runner for aircraft-positioning scenarios.")
    ap.add_argument("--model-module", type=str, default="model_func1_sh",
                    help="Nombre del módulo Python del modelo (sin .py).")

    # Aceptar ambos nombres (--scenarios y --scenarios-dir)
    ap.add_argument("--scenarios", type=str, default=None,
                    help="Carpeta que contiene los .xlsx (alias de --scenarios-dir).")
    ap.add_argument("--scenarios-dir", type=str, default=None,
                    help="Carpeta que contiene los .xlsx.")
    ap.add_argument("--pattern", type=str, default="*.xlsx",
                    help="Patrón glob para seleccionar escenarios.")
    ap.add_argument("--sheet", type=str, default="case",
                    help="Nombre de la hoja dentro del .xlsx.")

    ap.add_argument("--modes", type=str, nargs="+", default=["soft12", "soft", "medium", "hard"],
                    help="Modos de política a evaluar.")
    ap.add_argument("--solver-presets", type=str, nargs="+",
                    default=["base"],
                    choices=["base", "fast", "feasible"],
                    help="Presets del solver a cruzar con los modos.")
    ap.add_argument("--solver", type=str, default="gurobi",
                    help="Solver (gurobi, cplex, cbc, glpk, ...).")
    ap.add_argument("--timelimit", type=float, default=300.0,
                    help="Límite de tiempo en segundos.")
    ap.add_argument("--mipgap", type=float, default=0.02,
                    help="MIP gap objetivo.")

    ap.add_argument("--w-client-pos", type=float, default=1e6,
                    help="Peso para la parte blanda de la política cliente→posición.")
    ap.add_argument("--wms", type=float, default=1.0,
                    help="Peso del makespan.")

    ap.add_argument("--positions", type=str, default=None,
                    help="Lista de posiciones separadas por comas (e.g., position1,position2,position3).")
    ap.add_argument("--n-positions", type=int, default=None,
                    help="Si no se pasa --positions, genera este número con el prefijo.")
    ap.add_argument("--pos-prefix", type=str, default="position",
                    help="Prefijo para generar posiciones con --n-positions.")

    # Aceptar --results-dir además de --out
    ap.add_argument("--results-dir", type=str, default=None,
                    help="Carpeta donde guardar el CSV (se llamará results_batch.csv).")
    ap.add_argument("--out", type=str, default=None,
                    help="Ruta del CSV de salida. Si se pasa, tiene prioridad sobre --results-dir.")

    args = ap.parse_args()

    # Resolver escenarios dir (aceptar ambos flags)
    scenarios_dir = args.scenarios_dir or args.scenarios or "scenarios"

    # Importa el módulo del modelo
    model_mod = importlib.import_module(args.model_module)

    # Inyecta POSITIONS si nos lo piden
    if args.positions or args.n_positions:
        if args.positions:
            pos_list = [s.strip() for s in args.positions.split(",") if s.strip()]
        else:
            pos_list = [f"{args.pos_prefix}{i}" for i in range(1, int(args.n_positions) + 1)]
        setattr(model_mod, "POSITIONS", pos_list)

    # Asegura PLANNING_START por si el lector lo usa
    if not hasattr(model_mod, "PLANNING_START"):
        setattr(model_mod, "PLANNING_START", None)

    # Escenarios
    files = find_scenarios(scenarios_dir, args.pattern)
    if not files:
        print(f"No .xlsx files found in {scenarios_dir} with pattern {args.pattern}")
        sys.exit(1)

    # CSV de salida
    if args.out:
        out_path = Path(args.out)
    elif args.results_dir:
        out_path = Path(args.results_dir) / "results_batch.csv"
    else:
        out_path = Path("results_batch.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "mode", "preset",
        "status", "termination", "solve_time_s", "mipgap", "best_bound",
        "objective", "Tmax", "client_delay_sum", "viol_C_sum", "client_switches",
        "positions_slack_used", "switches_slot_sum", "presence_sum", "idle_sum",
        "n_jobs", "n_planes", "n_clients", "n_positions", "H"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fp in files:
            for mode in args.modes:
                for preset in args.solver_presets:
                    try:
                        metrics = build_and_solve(model_mod,
                                                  xlsx_path=fp,
                                                  sheet_name=args.sheet,
                                                  mode=mode,
                                                  w_client_pos=args.w_client_pos,
                                                  wms=args.wms,
                                                  solver_name=args.solver,
                                                  timelimit=args.timelimit,
                                                  mipgap=args.mipgap,
                                                  preset=preset)
                        row = {"scenario": fp.name, "mode": mode, "preset": preset}
                        row.update(metrics)
                        writer.writerow(row)
                        print(f"[OK] {fp.name} | {mode}:{preset} | {metrics['solve_time_s']}s | "
                              f"gap={metrics['mipgap']} | best={metrics['best_bound']} | obj={metrics['objective']}")
                    except Exception as e:
                        row = {
                            "scenario": fp.name, "mode": mode, "preset": preset,
                            "status": "ERROR", "termination": str(e)
                        }
                        writer.writerow(row)
                        print(f"[ERR] {fp.name} | {mode}:{preset} | {e}")

    print(f"\nSaved results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
