#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path

INFILE  = Path("results_suite_allmodes.xlsx")   # <-- si está en otra ruta, cámbiala
SHEET   = 0                                      # o nombre de hoja
OUTDIR  = Path("results_analysis")               # carpeta de salida

OUTDIR.mkdir(parents=True, exist_ok=True)

def _parse_scenario_name(name: str):
    """
    Extrae etiquetas del nombre de archivo de escenario.
    Convenciones que hemos venido usando:
      - few / many
      - loose / tight
      - seedXX
      - P<jobsPerPlane> (opcional)
      - pl<positions> (opcional)
    Devuelve dict con columnas categóricas útiles.
    """
    base = name.lower()
    # tipo few/many
    typ = "many" if "many" in base else ("few" if "few" in base else "unknown")
    # ventana loose/tight
    win = "tight" if "tight" in base else ("loose" if "loose" in base else "mix")
    # jobs-per-plane aproximado (si aparece P..\_)
    jobs_pp = None
    import re
    mP = re.search(r"_p(\d+)", base)
    if mP:
        jobs_pp = int(mP.group(1))
    # posiciones (si aparece pl..)
    mPL = re.search(r"_pl(\d+)", base)
    positions = int(mPL.group(1)) if mPL else None
    # seed
    mSeed = re.search(r"_seed(\d+)", base)
    seed = int(mSeed.group(1)) if mSeed else None

    return {
        "scenario_type": typ,
        "window_tightness": win,
        "jobs_per_plane_declared": jobs_pp,
        "positions_declared": positions,
        "seed": seed,
    }

def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _infer_feasible(status:str, term:str, objective):
    s = (status or "").lower()
    t = (term or "").lower()
    # Heurística: consideramos factible si el solver devuelve solution y no marca infeasible
    if "infeasible" in t:
        return False
    if "aborted" in (t + s):
        # pudo abortar con solución: consideramos factible si hay objetivo numérico
        return np.isfinite(_coerce_float(objective))
    # ok/optimal/feasible
    return True

def _gap_from_bound(obj, best_bound):
    """
    Gap relativo % a partir de obj y best_bound cuando el solver no lo reporta.
    Fórmula estándar MIP:
      gap = |obj - best_bound| / (1e-9 + max(1, |obj|))  (o usando max(|obj|, |best_bound|))
    Aquí usamos max(|obj|, |best_bound|, 1) para mayor estabilidad.
    """
    objf = _coerce_float(obj)
    bbf  = _coerce_float(best_bound)
    if not np.isfinite(objf) or not np.isfinite(bbf):
        return np.nan
    denom = max(abs(objf), abs(bbf), 1.0)
    return abs(objf - bbf) / denom

# ---- Cargar ----
if INFILE.suffix.lower() == ".csv":
    df = pd.read_csv(INFILE)
else:
    df = pd.read_excel(INFILE, sheet_name=SHEET)

# Normalizar columnas esperadas del runner (renombra si hace falta)
df.columns = [c.strip() for c in df.columns]

# Unificar nombres clave (ajusta si tu runner escribió otros)
rename_map = {
    "scenario": "scenario",
    "mode": "mode",
    "status": "status",
    "termination": "termination",
    "solve_time_s": "solve_time_s",
    "mipgap": "mipgap",
    "best_bound": "best_bound",
    "objective": "objective",
    "Tmax": "Tmax",
    "client_delay_sum": "client_delay_sum",
    "viol_C_sum": "viol_C_sum",
    "client_switches": "client_switches",
    "positions_slack_used": "positions_slack_used",
    "switches_slot_sum": "switches_slot_sum",
    "presence_sum": "presence_sum",
    "idle_sum": "idle_sum",
    "n_jobs": "n_jobs",
    "n_planes": "n_planes",
    "n_clients": "n_clients",
    "n_positions": "n_positions",
    "H": "H",
}
df = df.rename(columns=rename_map)

# Enriquecer con etiquetas de escenario
parsed = df["scenario"].apply(_parse_scenario_name).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

# Coaccionar tipos numéricos clave
for c in ["solve_time_s","mipgap","best_bound","objective","Tmax",
          "client_delay_sum","viol_C_sum","client_switches",
          "positions_slack_used","switches_slot_sum","presence_sum","idle_sum",
          "n_jobs","n_planes","n_clients","n_positions","H"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Inferir factibilidad y gap si falta
df["feasible"] = df.apply(lambda r: _infer_feasible(r.get("status",""), r.get("termination",""), r.get("objective",np.nan)), axis=1)
df["mipgap_filled"] = df["mipgap"]

# Relleno de gap con best_bound si solver no lo reportó
mask_gap_missing = ~np.isfinite(df["mipgap_filled"])
df.loc[mask_gap_missing, "mipgap_filled"] = df[mask_gap_missing].apply(
    lambda r: _gap_from_bound(r.get("objective", np.nan), r.get("best_bound", np.nan)),
    axis=1
)

# KPIs binarios
df["has_client_delay"] = (df["client_delay_sum"] > 1e-9)
df["violates_client_pos"] = (df["viol_C_sum"] > 1e-9)   # soft/medium/hard
df["has_client_switch"] = (df["client_switches"] > 1e-9)
df["uses_pos_slack"] = (df["positions_slack_used"] > 1e-9)

# ---------- Resúmenes ----------
def _agg_block(dfg):
    return pd.Series({
        "runs": len(dfg),
        "feasible_runs": int(dfg["feasible"].sum()),
        "feasible_rate": dfg["feasible"].mean(),
        "time_p50_s": dfg["solve_time_s"].median(),
        "time_p90_s": dfg["solve_time_s"].quantile(0.9),
        "gap_median": dfg["mipgap_filled"].median(skipna=True),
        "gap_p90": dfg["mipgap_filled"].quantile(0.9),
        "cases_with_delay": int(dfg["has_client_delay"].sum()),
        "cases_with_viol_pos": int(dfg["violates_client_pos"].sum()),
        "cases_with_switches": int(dfg["has_client_switch"].sum()),
        "cases_with_pos_slack": int(dfg["uses_pos_slack"].sum()),
        "avg_planes": dfg["n_planes"].mean(),
        "avg_jobs": dfg["n_jobs"].mean()
    })

# 1) Por modo
by_mode = df.groupby("mode", dropna=False).apply(_agg_block).reset_index()
by_mode.to_csv(OUTDIR / "summary_by_mode.csv", index=False)

# 2) Por tipo de escenario (few/many × loose/tight)
by_type = df.groupby(["scenario_type","window_tightness"], dropna=False).apply(_agg_block).reset_index()
by_type.to_csv(OUTDIR / "summary_by_type.csv", index=False)

# 3) Por modo × tipo
by_mode_type = df.groupby(["mode","scenario_type","window_tightness"], dropna=False).apply(_agg_block).reset_index()
by_mode_type.to_csv(OUTDIR / "summary_by_mode_type.csv", index=False)

# 4) Tabla “casos más al límite” (más planos y/o jobs y tiempos más altos o gap peor)
df["stress_score"] = (
    (df["n_planes"].fillna(0)) * 2.0
    + (df["n_jobs"].fillna(0)) * 1.0
    + (df["mipgap_filled"].fillna(0) * 100.0)
    + (df["solve_time_s"].fillna(0) / 30.0)
)
top_stress = df.sort_values(["feasible","stress_score","solve_time_s"], ascending=[False, False, False]).head(15)
top_stress.to_csv(OUTDIR / "top_stress_cases.csv", index=False)

# 5) Dump general enriquecido
df.to_csv(OUTDIR / "results_enriched.csv", index=False)

# 6) Resumen corto en TXT listo para pegar
def _fmt_pct(x):
    return f"{100*x:.1f}%" if pd.notna(x) else ""

lines = []
lines.append("# Resumen de KPIs\n")
for _, r in by_mode.iterrows():
    lines.append(f"- Modo {r['mode']}: {int(r['feasible_runs'])}/{int(r['runs'])} factibles ({_fmt_pct(r['feasible_rate'])}); "
                 f"tiempo p50={r['time_p50_s']:.2f}s, p90={r['time_p90_s']:.2f}s; "
                 f"GAP mediano={r['gap_median'] if pd.notna(r['gap_median']) else 'NA'}; "
                 f"viol. cliente→posición={int(r['cases_with_viol_pos'])}; "
                 f"cambios de cliente={int(r['cases_with_switches'])}; "
                 f"retrasos={int(r['cases_with_delay'])}.")
summary_txt = "\n".join(lines)
(OUTDIR / "summary_for_report.txt").write_text(summary_txt, encoding="utf-8")

print("Listo.\nArchivos generados en:", OUTDIR.resolve())
print("- summary_by_mode.csv")
print("- summary_by_type.csv")
print("- summary_by_mode_type.csv")
print("- top_stress_cases.csv")
print("- results_enriched.csv")
print("- summary_for_report.txt")
