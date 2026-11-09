"""Summarize frozen-winner behavior from saved observations; no solver/search runs.

Run from the repository root:
    python scripts/analyze_selected.py

Only results/diagnostics/selected_heuristics.json is written. Construction order
is the saved selected_before_pruning sequence on the original vertex labels;
set comparisons ignore sequence order. All rational arithmetic is exact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction as F
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverevo.contracts import Instance
from coverevo.dsl import parse_program, score_program
from coverevo.features import build_context, compute_features
from coverevo.solver import check_solution, prune_solution
from reproduce_results import recompute_contrasts, summarize_ablations

CONDITIONS = (("train", "random"), ("test_id", "random"), ("test_topology", "random"))


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean_excess(rows):
    require(bool(rows), "empty summary group")
    return sum((100 * (F(r["cost"]) / F(r["reference_cost"]) - 1) for r in rows), F()) / len(rows)


def verify_solution(instance, row):
    require(row["success"] is True, "mechanism analysis requires a successful saved call")
    before, after = tuple(row["selected_before_pruning"]), tuple(row["selected"])
    require(check_solution(instance, before) and check_solution(instance, after),
            "saved construction or final set is infeasible")
    require(tuple(prune_solution(instance, before)) == after, "saved pruning differs from fixed rule")
    require(sum(instance.costs[v] for v in before) == row["cost_before_pruning"],
            "pre-pruning objective mismatch")
    cost = sum(instance.costs[v] for v in after)
    require(cost == row["raw_cost"] == row["operational_cost"], "final objective mismatch")


def verify_trace(instance, result, method, program):
    """Check each saved choice, including all eligible candidates and tie-breaking."""
    verify_solution(instance, result)
    context = build_context(instance)
    uncovered, selected = set(range(instance.n)), []
    checked = 0
    for event in result["trace"]:
        if event["phase"] == "pruning":
            require(not uncovered, "pruning before coverage is complete")
            require(event["order"] == sorted(selected, key=lambda v: (-instance.costs[v], v)),
                    "pruning order mismatch")
            require(event["selected_after"] == result["selected"], "pruning event mismatch")
            require(set(event["removed"]) == set(selected) - set(result["selected"]),
                    "removed vertices mismatch")
            continue
        require(event["phase"] == "construct", "unknown trace phase")
        require(event["step"] == len(selected) and event["selected_before"] == selected,
                "construction prefix mismatch")
        require(event["uncovered_before"] == sorted(uncovered), "uncovered vertices mismatch")
        expected_nodes = {v for v in range(instance.n) if v not in selected
                          and context.closed_neighborhoods[v].intersection(uncovered)}
        candidates = event["candidates"]
        require(len(candidates) == len(expected_nodes)
                and {c["node"] for c in candidates} == expected_nodes, "candidate set mismatch")
        for candidate in candidates:
            v = candidate["node"]
            features = compute_features(context, frozenset(uncovered), v)
            require(asdict(features) == {k: F(x) for k, x in candidate["features"].items()},
                    "saved feature values differ from original graph")
            gain_count = len(context.closed_neighborhoods[v].intersection(uncovered))
            require(gain_count == candidate["gain_count"], "saved gain mismatch")
            require(candidate["already_covered"] == (v not in uncovered), "covered-state mismatch")
            expected = (F(gain_count, instance.costs[v]) if method == "GR_P"
                        else features.gain * features.rarity / features.cost if method == "RGR_P"
                        else score_program(program, features))
            require(expected == F(candidate["score"]), "saved score mismatch")
            # Verify the multiplicative rarity identity on every saved candidate.
            inverse_sum = sum((F(1, len(context.closed_neighborhoods[u]))
                               for u in context.closed_neighborhoods[v].intersection(uncovered)), F())
            weighted_coverage = context.mean_cost / len(uncovered) * inverse_sum / instance.costs[v]
            require(features.gain * features.rarity / features.cost == weighted_coverage,
                    "rarity-weighted coverage identity failed")
            checked += 1
        chosen = max(candidates, key=lambda c: (F(c["score"]),
                     -instance.costs[c["node"]], -c["node"]))["node"]
        require(chosen == event["chosen"], "saved choice violates deterministic tie-breaking")
        selected.append(chosen)
        uncovered.difference_update(context.closed_neighborhoods[chosen])
    require(not uncovered and selected == result["selected_before_pruning"],
            "trace does not reproduce the recorded construction sequence")
    return checked


def analyze():
    winners = read("results/winners.json")
    evo = sorted((w for w in winners if w["method"] == "EVO_F"), key=lambda w: w["run_id"])
    require([w["run_id"] for w in evo] == list(range(5)), "missing EVO-F frozen winner")
    programs = {}
    for w in winners:
        program = parse_program(w["expression"], w["provenance"])
        require(program.program_hash == w["program_hash"], "winner expression/identity mismatch")
        programs[w["program_hash"]] = program
    seed_hash = parse_program("div(gain, cost)", "handwritten_seed").program_hash
    nonseed = [w["run_id"] for w in evo if w["program_hash"] != seed_hash]
    instances = {x["instance_id"]: Instance.from_mapping(x)
                 for split in ("train", "test") for x in read(f"data/synthetic/{split}.json")}
    evaluation = read("results/evaluation.json")
    lookup = {(r["method"], r["run_id"], r["instance_id"]): r for r in evaluation}
    require(len(lookup) == len(evaluation), "duplicate saved evaluation")
    primary = [r for r in evaluation if r["split"] == "test_id" and r["variant"] == "random"]
    primary_ids = {r["instance_id"] for r in primary}
    require(len(primary_ids) == 60, "primary graph grid changed")
    baseline = {}
    for r in primary:
        if r["method"] == "GR_P":
            previous = baseline.setdefault(r["instance_id"], r)
            fields = ("selected_before_pruning", "selected", "raw_cost", "operational_cost")
            require(all(previous[k] == r[k] for k in fields), "greedy replicas disagree")
    require(set(baseline) == primary_ids, "incomplete primary baseline")
    per_winner, total = [], Counter()
    for winner in evo:
        rows = [r for r in primary if r["method"] == "EVO_F" and r["run_id"] == winner["run_id"]]
        require(len(rows) == 60 and {r["instance_id"] for r in rows} == primary_ids,
                "incomplete winner evaluation grid")
        counts = Counter()
        for row in rows:
            b = baseline[row["instance_id"]]
            require(row["program_hash"] == winner["program_hash"], "wrong frozen program")
            require(row["reference_cost"] == b["operational_cost"], "reference denominator mismatch")
            verify_solution(instances[row["instance_id"]], row)
            verify_solution(instances[row["instance_id"]], b)
            order = row["selected_before_pruning"] != b["selected_before_pruning"]
            before = set(row["selected_before_pruning"]) != set(b["selected_before_pruning"])
            after = set(row["selected"]) != set(b["selected"])
            cost = row["operational_cost"] != b["operational_cost"]
            require((not before or order) and (not after or before) and (not cost or after),
                    "comparison nesting violated")
            counts.update(observations=1, construction_order_different=int(order),
                pre_pruning_set_different=int(before), final_set_different=int(after),
                final_cost_different=int(cost), pre_different_final_same=int(before and not after),
                final_different_cost_equal=int(after and not cost))
        per_winner.append(dict(run_id=winner["run_id"], expression=winner["expression"],
                               is_seed=winner["program_hash"] == seed_hash, **counts))
        total.update(counts)

    ablations = read("results/feature_ablations.json")
    seen = set()
    for row in ablations["rows"]:
        if row["method"] != "EVO_F":
            continue
        key = row["run_id"], row["instance_id"], row["intervention"]
        require(key not in seen, "duplicate intervention observation")
        seen.add(key)
        winner = evo[row["run_id"]]
        require(row["program_hash"] == winner["program_hash"], "intervention source winner mismatch")
        if row["intervention"] == "original":
            if row["split"] == "train":
                run = read(f"results/discovery/runs/EVO_F-{row['run_id']}.json")
                records = run["evaluations"][winner["program_hash"]]["per_instance"]
                source = next(r for r in records if r["instance_id"] == row["instance_id"])
            else:
                source = lookup["EVO_F", row["run_id"], row["instance_id"]]
            require(row["cost"] == row["original_cost"] == source["operational_cost"],
                    "original ablation differs from saved discovery/evaluation")
            require(F(row["reference_cost"]) == F(source["reference_cost"]), "ablation denominator mismatch")
    ablation_summary = []
    for runs, label in ((nonseed, "nonseed"), (list(range(5)), "all_five")):
        for split, variant in CONDITIONS:
            for intervention in ("original", "rarity_zero"):
                rows = [r for r in ablations["rows"] if r["method"] == "EVO_F"
                        and r["run_id"] in runs and (r["split"], r["variant"]) == (split, variant)
                        and r["intervention"] == intervention]
                expected_n = (30 if split == "train" else 60) * len(runs)
                require(len(rows) == expected_n, "incomplete intervention grid")
                value = mean_excess(rows)
                ablation_summary.append(dict(group=label, runs=runs, split=split,
                    intervention=intervention, observations=len(rows),
                    mean_excess_percent_exact=str(value), mean_excess_percent=float(value)))
    per_run_ablation = []
    for run in nonseed:
        for split, variant in CONDITIONS:
            group = [r for r in ablations["rows"] if r["method"] == "EVO_F"
                     and r["run_id"] == run and (r["split"], r["variant"]) == (split, variant)]
            per_run_ablation.append(dict(run_id=run, split=split, **{
                i: float(mean_excess([r for r in group if r["intervention"] == i]))
                for i in ("original", "rarity_zero")}))
    # Confirm the unchanged five-winner intervention table from the original records.
    summary = summarize_ablations(ablations)
    for intervention, label in (("original", "Locked"), ("rarity_zero", "$r=0$"),
                               ("overlap_zero", "$o=0$"), ("no_pruning", "No pruning")):
        values = [next(s["mean_excess_percent"] for s in summary if s["method"] == "EVO_F"
                      and s["intervention"] == intervention and (s["split"], s["variant"]) == c)
                  for c in CONDITIONS]
        line = label + " & " + " & ".join(f"{float(v):+.3f}" for v in values)
        require(line in (ROOT / "report/tables/ablation_rows.tex").read_text(),
                "paper intervention table differs from saved observations")

    oracles = read("results/oracles.json")
    trace_summaries, checked = [], 0
    for case in read("results/diagnostics/decision_traces.json")["cases"]:
        instance = Instance.from_mapping(case["instance"])
        require(instance.to_mapping() == instances[instance.instance_id].to_mapping(), "trace graph mismatch")
        program = programs[case["program"]["program_hash"]]
        require(case["program"]["expression"] == evo[case["run_id"]]["expression"], "trace winner mismatch")
        oracle = oracles[instance.instance_id]
        require(oracle["status"] == "OPTIMAL" and oracle["certified_optimal"] is True
                and oracle["incumbent_cost"] == case["optimum"], "trace lacks a certified optimum")
        for method, result in case["results"].items():
            checked += verify_trace(instance, result, method, program)
            saved = lookup[method, case["run_id"], instance.instance_id]
            require(all(result[k] == saved[k] for k in
                        ("selected_before_pruning", "selected", "raw_cost", "operational_cost")),
                    "trace solution differs from original evaluation")
        a, b = (case["results"][method] for method in ("EVO_F", "GR_P"))
        first = next(i for i, (x, y) in enumerate(zip(a["selected_before_pruning"],
                                                    b["selected_before_pruning"])) if x != y)
        event = a["trace"][first]
        chosen = [a["selected_before_pruning"][first], b["selected_before_pruning"][first]]
        pair = [next(c for c in event["candidates"] if c["node"] == v) for v in chosen]
        trace_summaries.append(dict(label=case["label"], instance_id=instance.instance_id,
            run_id=case["run_id"], first_different_step_one_based=first + 1,
            chosen_evo_greedy=chosen, candidate_features=[c["features"] for c in pair],
            candidate_scores=[c["score"] for c in pair],
            greedy_priorities=[str(F(c["gain_count"], instance.costs[c["node"]])) for c in pair],
            evo_without_rarity_scores=[str(F(c["features"]["gain"]) / F(c["features"]["cost"])
                - (F(1, 5) * F(c["features"]["overlap"]) if case["run_id"] == 4 else 0)) for c in pair],
            final_costs={m: r["raw_cost"] for m, r in case["results"].items()}, optimum=case["optimum"]))
    relabel = read("results/relabeling.json")
    example = next(c for c in trace_summaries if c["label"] == "primary_win")
    w = evo[example["run_id"]]
    relabeled = [r for r in relabel["rows"] if r["instance_id"] == example["instance_id"]]
    require(len(relabeled) == 10 and len({r["permutation_index"] for r in relabeled}) == 10,
            "incomplete relabeling diagnostic")
    relabel_counts = Counter("win" if r["costs"][w["program_hash"]] < r["reference_cost"]
                            else "tie" if r["costs"][w["program_hash"]] == r["reference_cost"] else "loss"
                            for r in relabeled)
    external = []
    for family in ("T1", "T2"):
        rows = [r for r in read("results/external/quality.json")
                if r["method"] == "RGR+P" and r["family"] == family]
        require(len(rows) == len({r["instance_id"] for r in rows}) == 24, "external grid mismatch")
        external.append(dict(family=family, wins=sum(r["operational_cost"] < r["reference_cost"] for r in rows),
            ties=sum(r["operational_cost"] == r["reference_cost"] for r in rows),
            losses=sum(r["operational_cost"] > r["reference_cost"] for r in rows)))
    bootstrap = recompute_contrasts(evaluation, read("results/contrasts.json"),
                                   read("results/bootstrap_sensitivity.json"))
    return dict(scope="descriptive analysis of saved frozen programs; no new experiments",
        reproduction_command="python scripts/analyze_selected.py", nonseed_runs=nonseed,
        behavioral_totals=dict(total), per_winner_behavior=per_winner,
        rarity_ablations=ablation_summary, per_winner_rarity_ablations=per_run_ablation,
        illustrative_traces=trace_summaries, saved_trace_candidate_checks=checked,
        rarity_weighted_identity_checks=checked, primary_example_relabeling=dict(relabel_counts),
        external_rarity_wins_ties_losses=external, unchanged_bootstrap=bootstrap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/diagnostics/selected_heuristics.json")
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("behavioral_totals", "nonseed_runs",
        "rarity_ablations", "saved_trace_candidate_checks", "primary_example_relabeling",
        "external_rarity_wins_ties_losses")}, indent=2))
    print("All original bootstrap estimates/intervals and the five-winner intervention table agree.")


if __name__ == "__main__":
    main()
