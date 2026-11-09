from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("IND", "EVO_P", "EVO_F", "RS_PARAM", "GR_P", "RGR_P", "RD")
FIXED = {"GR_P", "RGR_P", "RD"}
LABELS = dict(zip(METHODS, ("IND", "EVO-P", "EVO-F", "RS-PARAM", "GR+P", "RGR+P", "Reverse-delete")))
EXTERNAL_METHODS = ("GR+P", "RGR+P", "Reverse-delete", "IND", "EVO_P", "EVO_F", "RS_PARAM")
CONDITIONS = (("test_id", "random"), ("test_id", "hub_expensive"),
              ("test_id", "hub_cheap"), ("test_size", "random"),
              ("test_topology", "random"))
SEED, DRAWS = 4212, 5000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def ratio(numerator, denominator):
    denominator = Fraction(denominator)
    if denominator <= 0:
        raise ValueError("cost denominator must be positive")
    return Fraction(numerator) / denominator


def certified_optimum(oracle):
    """An incumbent is an optimum only when both certification fields agree."""
    if oracle.get("certified_optimal") is not True or oracle.get("status") != "OPTIMAL":
        return None
    optimum = oracle["incumbent_cost"]
    if Fraction(optimum) <= 0:
        raise ValueError("certified optimum must be positive")
    return optimum


def counts(ratios):
    return [sum(x < 1 for x in ratios), sum(x == 1 for x in ratios), sum(x > 1 for x in ratios)]


def summarize_synthetic(rows, oracles):
    """Retain failures at operational cost; collapse only fixed quality replicas."""
    grouped, seen, fixed_values = defaultdict(list), set(), {}
    for source in rows:
        row = dict(source)
        key = row["method"], row["run_id"], row["instance_id"]
        if key in seen:
            raise ValueError("duplicate quality observation")
        seen.add(key)
        if row["method"] in FIXED:
            fixed_key = row["method"], row["instance_id"]
            value = row["operational_cost"], row["reference_cost"], row["success"]
            if fixed_key in fixed_values:
                if fixed_values[fixed_key] != value:
                    raise ValueError("inconsistent fixed baseline replication")
                continue
            fixed_values[fixed_key] = value
        optimum = certified_optimum(oracles[row["instance_id"]])
        if optimum is not None and min(row["operational_cost"], row["reference_cost"]) < optimum:
            raise ValueError("cost below certified optimum")
        row["optimum"] = optimum
        grouped[row["method"], row["split"], row["variant"]].append(row)
    summaries = []
    for (method, split, variant), group in sorted(grouped.items()):
        ratios = [ratio(r["operational_cost"], r["reference_cost"]) for r in group]
        certified = [r for r in group if r["optimum"] is not None]
        gaps = [100 * (ratio(r["operational_cost"], r["optimum"]) - 1) for r in certified]
        hits = sum(r["success"] is True and r["operational_cost"] == r["optimum"] for r in certified)
        head = [r for r in certified if r["reference_cost"] > r["optimum"]]
        summaries.append(dict(method=method, split=split, variant=variant,
            observations=len(group), base_graphs=len({r["base_graph_id"] for r in group}),
            runs=len({r["run_id"] for r in group}) if method not in FIXED else None,
            mean_operational_ratio=mean(ratios), median_operational_ratio=median(ratios),
            mean_excess_percent=100 * (mean(ratios) - 1),
            median_excess_percent=100 * (median(ratios) - 1),
            wins_ties_losses=counts(ratios), failures=sum(r["success"] is not True for r in group),
            failure_rate=Fraction(sum(r["success"] is not True for r in group), len(group)),
            certified_pairs=len(certified), mean_gap_percent=mean(gaps),
            optimal_hits=hits, optimal_hit_rate=Fraction(hits, len(certified)) if certified else None,
            headroom=dict(observations=len(head), base_graphs=len({r["base_graph_id"] for r in head}),
                wins_ties_losses=counts([ratio(r["operational_cost"], r["reference_cost"]) for r in head]),
                optimal_hits=sum(r["success"] is True and r["operational_cost"] == r["optimum"] for r in head),
                mean_gap_closed=mean([ratio(r["reference_cost"] - r["operational_cost"],
                    r["reference_cost"] - r["optimum"]) for r in head]))))
    return summaries


def summarize_runtime(rows):
    """Median of repeated timings per program-instance pair, then across pairs.

    Synthetic fixed-baseline run replicas remain timing observations, matching
    the paper's recorded timing convention rather than its quality denominator.
    """
    grouped, seen = defaultdict(lambda: defaultdict(list)), set()
    for r in rows:
        pair = r["run_id"], r["instance_id"]
        group = r["method"], r["split"], r["variant"]
        key = (*group, *pair, r["timing_repeat"])
        if key in seen:
            raise ValueError("duplicate timing repeat")
        if type(r["runtime_ns"]) is not int or r["runtime_ns"] < 0:
            raise ValueError("runtime_ns must be a nonnegative integer")
        seen.add(key)
        grouped[group][pair].append(r["runtime_ns"])
    return [dict(method=method, split=split, variant=variant,
        median_runtime_ns=median(median(v) for v in pairs.values()),
        median_runtime_ms=median(median(v) for v in pairs.values()) / 1e6,
        program_instance_pairs=len(pairs), timing_measurements=sum(map(len, pairs.values())),
        timing_definition="median of per-program-instance repeat medians")
        for (method, split, variant), pairs in sorted(grouped.items())]


def aggregate_external(rows, oracles):
    """Use saved first-call membership rows; uncertified incumbents are excluded."""
    if not rows:
        raise ValueError("external aggregation requires observations")
    ratios = [ratio(r["operational_cost"], r["reference_cost"]) for r in rows]
    certified = [(r, certified_optimum(oracles[r["instance_id"]])) for r in rows]
    certified = [(r, optimum) for r, optimum in certified if optimum is not None]
    gaps = [100 * (ratio(r["operational_cost"], optimum) - 1) for r, optimum in certified]
    if any(g < 0 for g in gaps):
        raise ValueError("heuristic improves a certified optimum")
    return dict(pairs=len(rows), mean_operational_ratio=mean(ratios),
        mean_excess_percent=100 * (mean(ratios) - 1),
        median_excess_percent=100 * (median(ratios) - 1), wins_ties_losses=counts(ratios),
        feasibility_percent=100 * Fraction(sum(r["feasible"] is True for r in rows), len(rows)),
        failures=sum(r["success"] is not True for r in rows),
        median_runtime_ms=median(r["runtime_ns"] for r in rows) / 1e6,
        mean_runtime_ms=float(mean([r["runtime_ns"] for r in rows])) / 1e6,
        certified_pairs=len(certified), certified_mean_gap_percent=mean(gaps),
        certified_median_gap_percent=median(gaps) if gaps else None,
        certified_hit_percent=100 * Fraction(sum(r["success"] is True and r["raw_cost"] == optimum
            for r, optimum in certified), len(certified)) if certified else None)


def paired_crossed_bootstrap(differences, *, seed=SEED, draws=DRAWS):
    """Original convention: paired run labels and paired graphs in each draw."""
    values = _bootstrap_array(differences)
    _bootstrap_parameters(seed, draws)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        runs = rng.integers(values.shape[0], size=values.shape[0])
        graphs = rng.integers(values.shape[1], size=values.shape[1])
        samples[i] = values[np.ix_(runs, graphs)].mean()
    return tuple(float(x) for x in np.quantile(samples, [.025, .975]))


def independent_run_bootstrap(a, b, *, seed=SEED, draws=DRAWS):
    """Sensitivity: independent method run blocks, shared graph blocks per draw."""
    a, b = _bootstrap_array(a), _bootstrap_array(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("run-by-graph arrays must share graph columns")
    _bootstrap_parameters(seed, draws)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        runs_a = rng.integers(a.shape[0], size=a.shape[0])
        runs_b = rng.integers(b.shape[0], size=b.shape[0])
        graphs = rng.integers(a.shape[1], size=a.shape[1])
        samples[i] = a[np.ix_(runs_a, graphs)].mean() - b[np.ix_(runs_b, graphs)].mean()
    return tuple(float(x) for x in np.quantile(samples, [.025, .975]))


def _bootstrap_array(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("require finite nonempty run-by-graph arrays")
    return values


def _bootstrap_parameters(seed, draws):
    if type(seed) is not int or type(draws) is not int or draws <= 0:
        raise ValueError("require integer seed and positive integer draws")


def recompute_contrasts(rows, saved_contrasts, saved_sensitivity):
    primary = [r for r in rows if (r["split"], r["variant"]) == CONDITIONS[0]]
    graphs = sorted({r["base_graph_id"] for r in primary})
    methods = {m for c in saved_contrasts["contrasts"] for m in c["methods"]}
    lookup = {}
    for r in primary:
        if r["method"] not in methods:
            continue
        key = r["method"], r["run_id"], r["base_graph_id"]
        if key in lookup:
            raise ValueError("duplicate primary bootstrap observation")
        lookup[key] = r
    arrays = {}
    runs = sorted({r["run_id"] for r in primary if r["method"] in methods})
    for method in methods:
        # The archived sensitivity converted each cost ratio to float first.
        # Retain that arithmetic order as well as its sampling convention.
        arrays[method] = np.array([[100 * (lookup[method, run, graph]["operational_cost"] /
            lookup[method, run, graph]["reference_cost"] - 1) for graph in graphs] for run in runs])
    contrasts = []
    for saved in saved_contrasts["contrasts"]:
        a, b = saved["methods"]
        # Reproduce the original exact cost difference before converting to float.
        differences = []
        for run in runs:
            values = []
            for graph in graphs:
                ra, rb = lookup[a, run, graph], lookup[b, run, graph]
                if ra["reference_cost"] != rb["reference_cost"]:
                    raise ValueError("paired rows disagree on reference cost")
                values.append(float(100 * ratio(ra["operational_cost"] - rb["operational_cost"], ra["reference_cost"])))
            differences.append(values)
        matrix = np.asarray(differences)
        estimate = float(matrix.mean())
        original_interval = paired_crossed_bootstrap(matrix)
        independent_interval = independent_run_bootstrap(arrays[a], arrays[b])
        assert_close(estimate, saved["mean_percentage_points"], "contrast point estimate")
        assert_close(original_interval, saved["interval_95"], "original bootstrap interval")
        archived = next(c for c in saved_sensitivity["contrasts"] if c["methods"] == saved["methods"])
        assert_close(independent_interval, archived["independent_run_paired_graph_interval_95"], "independent-run interval")
        contrasts.append(dict(methods=[a, b], hypothesis_scope=saved["scope"],
            mean_percentage_points=estimate, per_run_means=matrix.mean(axis=1).tolist(),
            archived_run_index_paired_interval_95=original_interval,
            independent_run_paired_graph_interval_95=independent_interval))
    return dict(seed=SEED, draws=DRAWS, runs_per_method=len(runs), primary_graphs=len(graphs),
        conditional_on="fixed training corpus", contrasts=contrasts)


def summarize_ablations(payload):
    grouped = defaultdict(list)
    for r in payload["rows"]:
        grouped[r["method"], r["intervention"], r["split"], r["variant"]].append(r)
    return [dict(method=m, intervention=i, split=s, variant=v, observations=len(rows),
        mean_excess_percent=mean([100 * (ratio(r["cost"], r["reference_cost"]) - 1) for r in rows]),
        mean_change_pp=mean([100 * ratio(r["cost"] - r["original_cost"], r["reference_cost"]) for r in rows]))
        for (m, i, s, v), rows in sorted(grouped.items())]


def summarize_control(payload, oracles):
    summaries = []
    for method in ("GR+P", "RGR+P", "EVO-F", "CP-SAT50"):
        rows = [r for r in payload["rows"] if r["method"] == method]
        if not rows:
            raise ValueError("missing control method: " + method)
        pairs = defaultdict(list)
        for r in rows:
            pairs[r["run_id"], r["instance_id"]].append(r["end_to_end_s"] * 1000)
        certified = [(r, certified_optimum(oracles[r["instance_id"]])) for r in rows]
        certified = [(r, optimum) for r, optimum in certified if optimum is not None]
        gaps = [100 * (ratio(r["cost"], optimum) - 1) for r, optimum in certified]
        if any(g < 0 for g in gaps):
            raise ValueError("control cost below certified optimum")
        summaries.append(dict(method=method, measured_calls=len(rows), certified_calls=len(certified),
            mean_gap_percent=mean(gaps), optimal_hit_percent=100 * Fraction(sum(r["cost"] == optimum
                for r, optimum in certified), len(certified)) if certified else None,
            median_ms=median(median(v) for v in pairs.values()),
            overrun_count=sum(r["deadline_exceeded"] is True for r in rows)))
    return summaries


def assert_close(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(label + " differs from saved result")
    elif isinstance(actual, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError(label + " differs from saved result")
        for a, b in zip(actual, expected):
            assert_close(a, b, label)
    elif not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} differs from saved result: {actual!r} versus {expected!r}")


def write_csv(path, rows):
    """Readable decimal columns, with companion exact rational columns."""
    output = []
    for source in rows:
        row = {}
        for key, value in source.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, Fraction):
                row[key], row[key + "_exact"] = float(value), str(value)
            elif isinstance(value, (tuple, list)):
                row[key] = "/".join(map(str, value))
            else:
                row[key] = value
        output.append(row)
    fields = list(dict.fromkeys(k for row in output for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def json_value(value):
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(type(value).__name__)


def write_tables(tables, synthetic, runtimes, external, control, ablations):
    tables.mkdir(parents=True, exist_ok=True)
    quality = {(r["method"], r["split"], r["variant"]): r for r in synthetic}
    times = {(r["method"], r["split"], r["variant"]): r for r in runtimes}

    def write(name, lines):
        (tables / name).write_text("% Generated from saved JSON by scripts/reproduce_results.py.\n" +
            "\n".join(lines) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")

    lines = []
    for m in METHODS:
        r, timing = quality[m, *CONDITIONS[0]], times[m, *CONDITIONS[0]]
        wtl = "/".join(map(str, r["wins_ties_losses"]))
        lines.append(f"{LABELS[m]} & {float(r['mean_excess_percent']):+.3f} & {float(r['mean_gap_percent']):.3f} & "
            f"{100 * float(r['optimal_hit_rate']):.1f} & {wtl} & {timing['median_runtime_ms']:.2f} " + r"\\")
    write("primary_rows.tex", lines)
    write("shift_focus_rows.tex", [LABELS[m] + " & " + " & ".join(
        f"{float(quality[m, *c]['mean_excess_percent']):+.3f}" for c in CONDITIONS[1:]) + r" \\" for m in METHODS])
    by_external = {(r["family"], r["method"]): r for r in external}
    lines = []
    for m in EXTERNAL_METHODS:
        a, b = by_external["T1", m], by_external["T2", m]
        label = LABELS.get(m, m)
        lines.append(f"{label} & {float(a['mean_excess_percent']):+.2f} & {float(b['mean_excess_percent']):+.2f} & "
            f"{a['median_runtime_ms'] / 1000:.3f}/{b['median_runtime_ms'] / 1000:.3f} " + r"\\")
    write("external_focus_rows.tex", lines)
    write("control_rows.tex", [f"{r['method']} & {float(r['mean_gap_percent']):.2f} & "
        f"{float(r['optimal_hit_percent']):.1f} & {r['median_ms']:.2f} & "
        f"{r['overrun_count']}/{r['measured_calls']} " + r"\\" for r in control])
    by_ablation = {(r["method"], r["intervention"], r["split"], r["variant"]): r for r in ablations}
    lines = []
    for intervention, label in (("original", "Locked"), ("rarity_zero", "$r=0$"),
                                ("overlap_zero", "$o=0$"), ("no_pruning", "No pruning")):
        values = [by_ablation["EVO_F", intervention, split, "random"]["mean_excess_percent"]
                  for split in ("train", "test_id", "test_topology")]
        lines.append(label + " & " + " & ".join(f"{float(v):+.3f}" for v in values) + r" \\")
    write("ablation_rows.tex", lines)


def plot_bootstrap(contrasts, path):
    """The paper's original reporting plot, using recomputed intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    for label, key, offset, marker, fill in (
        ("Original: run-index paired", "archived_run_index_paired_interval_95", .13, "o", "white"),
        ("Sensitivity: independent runs", "independent_run_paired_graph_interval_95", -.13, "s", "black"),
    ):
        points = np.array([c["mean_percentage_points"] for c in contrasts])
        bounds = np.array([c[key] for c in contrasts])
        y = np.arange(len(contrasts))[::-1] + offset
        ax.errorbar(points, y, xerr=np.vstack((points - bounds[:, 0], bounds[:, 1] - points)),
            fmt=marker, color="black", markerfacecolor=fill, markersize=4,
            linewidth=1.1, capsize=2, label=label)
    ax.axvline(0, color=".55", linestyle="--", linewidth=.8)
    labels = [" − ".join(LABELS.get(m, m) for m in c["methods"]) for c in contrasts]
    ax.set_yticks(np.arange(len(contrasts))[::-1], labels)
    ax.set_xlabel("Excess difference (percentage points)", fontsize=8)
    ax.set_ylim(-.5, 3.05)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def reproduce(root=ROOT, output_dir=None):
    root = Path(root)
    destination = Path(output_dir) if output_dir is not None else root
    results = destination / "results"
    rows = read_json(root / "results/evaluation.json")
    timing = read_json(root / "results/timing.json")
    oracles = read_json(root / "results/oracles.json")
    external_rows = read_json(root / "results/external/quality.json")
    external_oracles = read_json(root / "results/external/oracles.json")
    saved_external = read_json(root / "results/external/summary.json")
    synthetic = summarize_synthetic(rows, oracles)
    runtimes = summarize_runtime(timing)
    external = []
    for family in ("T1", "T2"):
        for method in EXTERNAL_METHODS:
            group = [r for r in external_rows if r["family"] == family and r["method"] == method]
            summary = aggregate_external(group, external_oracles)
            for key, value in saved_external["families"][family][method].items():
                assert_close(summary[key], value, f"external {family}/{method}/{key}")
            external.append(dict(family=family, method=method, **summary))
    bootstrap = recompute_contrasts(rows, read_json(root / "results/contrasts.json"),
        read_json(root / "results/bootstrap_sensitivity.json"))
    ablations = summarize_ablations(read_json(root / "results/feature_ablations.json"))
    control = summarize_control(read_json(root / "results/control.json"), oracles)
    results.mkdir(parents=True, exist_ok=True)
    primary = [r for r in synthetic if (r["split"], r["variant"]) == CONDITIONS[0]]
    write_csv(results / "primary_results.csv", primary)
    write_csv(results / "synthetic_shifts.csv", synthetic)
    write_csv(results / "external_t1_t2.csv", external)
    write_csv(results / "runtime_results.csv", runtimes)
    write_csv(results / "ablations.csv", ablations)
    write_csv(results / "control_results.csv", control)
    write_csv(results / "headroom.csv", [dict(method=r["method"], split=r["split"],
        variant=r["variant"], **r["headroom"]) for r in synthetic])
    contrast_csv = []
    for c in bootstrap["contrasts"]:
        old, new = c["archived_run_index_paired_interval_95"], c["independent_run_paired_graph_interval_95"]
        contrast_csv.append(dict(method_a=c["methods"][0], method_b=c["methods"][1],
            scope=c["hypothesis_scope"], mean_percentage_points=c["mean_percentage_points"],
            paired_low=old[0], paired_high=old[1], independent_low=new[0], independent_high=new[1],
            seed=SEED, draws=DRAWS))
    write_csv(results / "primary_contrasts.csv", contrast_csv)
    payload = dict(synthetic=synthetic, runtime=runtimes, external=external,
        bootstrap=bootstrap, ablations=ablations, control=control,
        timing_provenance="saved measurements; no new timings", network_calls=0)
    (results / "recomputed_metrics.json").write_text(json.dumps(payload, indent=2,
        default=json_value) + "\n", encoding="utf-8")
    write_tables(destination / "report/tables", synthetic, runtimes, external, control, ablations)
    plot_bootstrap(bootstrap["contrasts"], destination / "report/figures/primary_bootstrap_sensitivity.pdf")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
        help="write results/ and report/ under this directory (default: repository root)")
    args = parser.parse_args()
    payload = reproduce(output_dir=args.output_dir)
    print(f"Recomputed {len(payload['synthetic'])} synthetic groups, {len(payload['external'])} external groups, "
          f"{len(payload['bootstrap']['contrasts'])} contrasts, five paper tables and the bootstrap figure.")
    print("Saved original times retained; no new experiments or network calls.")


if __name__ == "__main__":
    main()
