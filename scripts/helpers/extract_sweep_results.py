import json


def _near_one_filter(data, max_distance=0.5):
    return [
        (nf, ef, err, sr)
        for nf, ef, err, sr in data
        if abs(nf - 1.0) <= max_distance and abs(ef - 1.0) <= max_distance
    ]


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


def analyze(data, near_one=None):
    valid = [
        (nf, ef, err, sr) for nf, ef, err, sr in data if err != float("inf") and sr > 0
    ]
    print(f"Trials with data: {len(data)}")
    print(f"Valid trials (finite error, success > 0): {len(valid)}")

    if near_one is not None:
        valid = _near_one_filter(valid, max_distance=near_one)
        print(f"Valid trials with NF, EF in [1±{near_one}]: {len(valid)}")

    if not valid:
        print("\nNo valid trials to analyze.")
        return

    # --- Summary: best error, best success rate, most balanced (near 1.0) ---
    print()
    print("=== SUMMARY (best error / best success rate / most balanced) ===")
    by_error = sorted(valid, key=lambda r: r[2])
    by_sr = sorted(valid, key=lambda r: -r[3])
    errors = [r[2] for r in valid]
    srs = [r[3] for r in valid]
    min_e, max_e = min(errors), max(errors)
    min_sr, max_sr = min(srs), max(srs)

    def combined(err, sr):
        ne = (err - min_e) / (max_e - min_e) if max_e > min_e else 0
        ns = (sr - min_sr) / (max_sr - min_sr) if max_sr > min_sr else 0
        return 0.5 * (1 - ne) + 0.5 * ns

    best_err_run = by_error[0]
    best_sr_run = by_sr[0]
    balanced = max(valid, key=lambda r: combined(r[2], r[3]))
    print(
        f"  Best error:        NF={best_err_run[0]:.1f} EF={best_err_run[1]:.1f}  \
            error={best_err_run[2]:.4f}  success_rate={best_err_run[3]:.2%}"
    )
    print(
        f"  Best success rate: NF={best_sr_run[0]:.1f} EF={best_sr_run[1]:.1f}  \
            error={best_sr_run[2]:.4f}  success_rate={best_sr_run[3]:.2%}"
    )
    print(
        f"  Most balanced:     NF={balanced[0]:.1f} EF={balanced[1]:.1f}  \
            error={balanced[2]:.4f}  success_rate={balanced[3]:.2%}  \
                (combined={combined(balanced[2], balanced[3]):.3f})"
    )
    print()

    print()
    by_error = sorted(valid, key=lambda r: r[2])
    print("=== TOP 15 BY LOWEST ERROR ===")
    _print_table(by_error[:15])

    print()
    by_sr = sorted(valid, key=lambda r: -r[3])
    print("=== TOP 15 BY HIGHEST SUCCESS RATE ===")
    _print_table(by_sr[:15])

    print()
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
    sweep_id: str = None,
    entity: str = "yaxing_li",
    project: str = "hyperparam-tuning",
    save: str = None,
    load: str = None,
    near_one: float = None,
):
    """Analyze wandb sweep results.

    Args:
        sweep_id: Wandb sweep ID (e.g. ta7fgh0c). Ignored if --load is set.
        entity: Wandb entity.
        project: Wandb project name.
        save: Optional path to save raw data as JSON.
        load: Optional path to load raw data from JSON (skips wandb fetch).
        near_one: If set, only consider runs where NF and EF are in [1-near_one, 1+near_one].
    """
    if load:
        with open(load) as f:
            data = [tuple(x) for x in json.load(f)]
        print(f"Loaded {len(data)} runs from {load}")
    elif sweep_id:
        data = fetch_data(entity, project, sweep_id)
    else:
        raise SystemExit("Provide either sweep_id or --load path to JSON.")
    analyze(data, near_one=near_one)

    if save:
        with open(save, "w") as f:
            json.dump(data, f)
        print(f"\nRaw data saved to {save}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
