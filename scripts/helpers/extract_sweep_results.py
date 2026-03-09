"""Fetch sweep results from remote wandb and analyze error vs success rate.

Usage:
    python scripts/helpers/extract_sweep_results.py ta7fgh0c
    python scripts/helpers/extract_sweep_results.py ta7fgh0c --entity my_team
    python scripts/helpers/extract_sweep_results.py ta7fgh0c --save results.json
"""

import json


def fetch_data(entity, project, sweep_id):
    import wandb

    api = wandb.Api(timeout=60)
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    print(f"Sweep: {sweep.name}, state: {sweep.state}", flush=True)

    runs = sweep.runs
    print(f"Total runs: {len(runs)}", flush=True)

    data = []
    for i, run in enumerate(runs):
        try:
            nf = run.config.get("node_factor")
            ef = run.config.get("edge_factor")
            err = run.summary.get("error")
            sr = run.summary.get("success_rate", 0.0)
            if nf is not None and ef is not None and err is not None:
                data.append((nf, ef, err, sr if sr else 0.0))
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}...", flush=True)

    return data


def analyze(data):
    valid = [
        (nf, ef, err, sr) for nf, ef, err, sr in data if err != float("inf") and sr > 0
    ]
    print(f"Trials with data: {len(data)}")
    print(f"Valid trials (finite error, success > 0): {len(valid)}")

    if not valid:
        print("\nNo valid trials to analyze.")
        return

    print()
    by_error = sorted(valid, key=lambda r: r[2])
    print("=== TOP 15 BY LOWEST ERROR ===")
    _print_table(by_error[:15])

    print()
    by_sr = sorted(valid, key=lambda r: -r[3])
    print("=== TOP 15 BY HIGHEST SUCCESS RATE ===")
    _print_table(by_sr[:15])

    print()
    errors = [r[2] for r in valid]
    srs = [r[3] for r in valid]
    min_e, max_e = min(errors), max(errors)
    min_sr, max_sr = min(srs), max(srs)

    def combined(err, sr):
        ne = (err - min_e) / (max_e - min_e) if max_e > min_e else 0
        ns = (sr - min_sr) / (max_sr - min_sr) if max_sr > min_sr else 0
        return 0.5 * (1 - ne) + 0.5 * ns

    scored = sorted(valid, key=lambda r: -combined(r[2], r[3]))
    print("=== TOP 15 BY COMBINED SCORE (50/50) ===")
    print(f"{'NF':>5} {'EF':>5} {'Error':>8} {'Success%':>10} {'Score':>7}")
    print("-" * 40)
    for nf, ef, err, sr in scored[:15]:
        print(f"{nf:5.1f} {ef:5.1f} {err:8.3f} {sr*100:9.1f}% {combined(err, sr):7.3f}")

    print()
    pareto = []
    for r in valid:
        nf, ef, err, sr = r
        dominated = any(
            r2[2] <= err and r2[3] >= sr and (r2[2] < err or r2[3] > sr) for r2 in valid
        )
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda r: r[2])
    print(f"=== PARETO FRONT ({len(pareto)} points) ===")
    _print_table(pareto)


def _print_table(rows):
    print(f"{'NF':>5} {'EF':>5} {'Error':>8} {'Success%':>10}")
    print("-" * 32)
    for nf, ef, err, sr in rows:
        print(f"{nf:5.1f} {ef:5.1f} {err:8.3f} {sr*100:9.1f}%")


def main(
    sweep_id: str,
    entity: str = "yaxing_li",
    project: str = "hyperparam-tuning",
    save: str = None,
):
    """Analyze wandb sweep results.

    Args:
        sweep_id: Wandb sweep ID (e.g. ta7fgh0c).
        entity: Wandb entity.
        project: Wandb project name.
        save: Optional path to save raw data as JSON.
    """
    data = fetch_data(entity, project, sweep_id)
    analyze(data)

    if save:
        with open(save, "w") as f:
            json.dump(data, f)
        print(f"\nRaw data saved to {save}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
