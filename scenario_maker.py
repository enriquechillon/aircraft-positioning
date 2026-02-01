# scenario_generator.py
import os
import math
import random
import argparse
import numpy as np
import pandas as pd

# --- Presets de dificultad ---
PRESETS = {
    # pocos aviones, holguras grandes
    "few-loose":     dict(n_planes=3,  tasks_range=(3,5),  H=80,  slack=("loose"),   pos=5),
    # pocos aviones, holguras cortas
    "few-tight":     dict(n_planes=3,  tasks_range=(3,5),  H=45,  slack=("tight"),   pos=5),
    # media densidad, holgura media
    "many-medium":   dict(n_planes=8,  tasks_range=(4,6),  H=120, slack=("medium"),  pos=5),
    # alta densidad, holguras cortas
    "many-tight":    dict(n_planes=12, tasks_range=(5,7),  H=140, slack=("tight"),   pos=5),
    # alta densidad, holguras mixtas
    "heavy-mix":     dict(n_planes=1, tasks_range=(6,8),  H=160, slack=("mix"),     pos=5),
}

def _slack_window(level, dur):
    """
    Devuelve (pre_slack, post_slack) en días según la dificultad.
    """
    if level == "loose":
        return (dur*0.8 + 2.0, dur*0.8 + 2.0)
    if level == "medium":
        return (dur*0.35 + 0.5, dur*0.35 + 0.5)
    if level == "tight":
        return (max(0.2, dur*0.05), max(0.2, dur*0.05))
    # mix: aleatorio entre medium y tight
    pre = np.random.uniform(dur*0.05, dur*0.35) + np.random.uniform(0.2, 0.7)
    post = np.random.uniform(dur*0.05, dur*0.35) + np.random.uniform(0.2, 0.7)
    return (pre, post)

def _dur_sample(mean=3.0, var=1.0):
    d = max(0.5, np.random.normal(mean, math.sqrt(var)))
    return float(d)

def _fabricate_case(n_planes, tasks_range, H, slack, n_positions=5, seed=None):
    """
    Genera un DataFrame con columnas requeridas por el modelo de una sola hoja:
    job(str), plane(int), client(int), duration(float), es(float/datetime), lf(float/datetime)
    Opcional: task(int) para precedencias por avión.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    planes = list(range(1, n_planes+1))
    # Clientes: asigna 1..min( max(2, n_planes//2), n_planes )
    n_clients = max(2, min(6, n_planes//2 if n_planes >= 4 else 2))
    clients_pool = list(range(1, n_clients+1))

    rows = []
    for r in planes:
        n_tasks = random.randint(tasks_range[0], tasks_range[1])
        # Distribución secuencial (carril) para factibilidad local
        cursor = np.random.uniform(0.0, max(1.0, 0.1*H))
        for t in range(1, n_tasks+1):
            dur = _dur_sample(mean=3.0, var=1.2)
            pre, post = _slack_window(slack if slack != "mix" else random.choice(["tight","medium"]), dur)
            es = max(0.0, cursor - pre)
            lf = min(H, cursor + dur + post)
            if lf < es + dur:
                lf = es + dur + 0.1
            # cliente: fija por avión para coherencia (o cambia según prefieras)
            c = random.choice(clients_pool)
            job = f"{r}-{t}"
            rows.append({
                "job": job,
                "plane": r,
                "client": c,
                "duration": round(dur, 3),
                "task": t,      # opcional, tu lector la usa para PRED si está presente
                "es": round(es, 3),
                "lf": round(lf, 3)
            })
            cursor = lf + np.random.uniform(0.05, 0.8)  # separa un poco
        # pequeña compresión para no desbordar mucho H
        if cursor > H * 0.95:
            scale = (H * 0.9) / cursor
            for row in rows:
                if row["plane"] == r:
                    row["es"] *= scale
                    row["lf"] *= scale

    df = pd.DataFrame(rows)
    return df

def write_case_xlsx(df, out_path, sheet_name="case"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=sheet_name)

def main():
    ap = argparse.ArgumentParser(description="Generador de escenarios factibles (.xlsx, una sola hoja).")
    ap.add_argument("--outdir", default="scenarios", help="Carpeta de salida")
    ap.add_argument("--n", type=int, default=50, help="Número total de escenarios a generar")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sheet", default="case", help="Nombre de la hoja a escribir")
    ap.add_argument("--mix", action="store_true", help="Mezclar presets para la tirada")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    # Secuencia de presets de menor a mayor dificultad
    order = ["few-loose","few-tight","many-medium","many-tight","heavy-mix"]
    if args.mix:
        seq = [random.choice(order) for _ in range(args.n)]
    else:
        # rellenamos en “bloques” para cubrir todos
        blocks = (args.n + len(order) - 1) // len(order)
        seq = (order * blocks)[:args.n]

    for i, preset in enumerate(seq, start=1):
        cfg = PRESETS[preset].copy()
        df = _fabricate_case(cfg["n_planes"], cfg["tasks_range"], cfg["H"], cfg["slack"], n_positions=cfg["pos"], seed=args.seed+i)
        fname = f"scn_{preset}_{i:03d}.xlsx"
        path = os.path.join(args.outdir, fname)
        write_case_xlsx(df, path, sheet_name=args.sheet)
        print(f"[OK] {fname} ({preset}) → {path}")

if __name__ == "__main__":
    main()
