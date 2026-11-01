"""Regenerate experiments/selector_results.md from selector_log.csv and selector_instance_metrics.csv."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build(log_csv="experiments/selector_log.csv", inst_csv="experiments/selector_instance_metrics.csv") -> str:
    log = pd.read_csv(log_csv)
    res = log[(log.row_type == "result") & (log.status == "completed")]
    # one row per config: the latest completed run
    res = res.assign(cfg=res.experiment_id.str.split("@").str[0]).drop_duplicates("cfg", keep="last")
    agg = pd.read_csv(inst_csv).set_index("config_name")
    order = ["S0_pilot_filter", "S1_conservative", "S2_conservative_border_rule", "S3_S1_plus_review_flag",
             "S4_S1_plus_relaxed_fallback", "E1_S1_alt_shape", "E2_S1_alt_conf", "E4_S1_rel_area", "E5_S1_alt_gate_fallback"]
    res = res.set_index("cfg").loc[[c for c in order if c in set(res.cfg)]]
    L = ["# Selector results on dev (88 images, split v2)", "",
         "Generated from `experiments/selector_log.csv` and `experiments/selector_instance_metrics.csv` by "
         "`python -m src.experiments.results_table`. Labels are AI-assisted image-level counts and AI-assisted manual candidate "
         "verdicts, not pixel ground truth. No IoU is reported.", "",
         "| config | exact | over | under | mean abs err | median | max | precision (lenient) | recall (lenient) | precision (strict) | recall (strict) | F1 (lenient) | bad-boundary TPs | FP | missed | runtime s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg, r in res.iterrows():
        m = json.loads(r.dev_metrics)["all_dev"]; a = agg.loc[cfg]
        L.append(f"| {cfg} | {m['count_exact_rate']} | {m['over_rate']} | {m['under_rate']} | {m['mean_abs_count_err']} | {m['median_abs_count_err']} | "
                 f"{m['max_abs_count_err']} | {a.precision_lenient} | {a.recall_lenient} | {a.precision_strict} | {a.recall_strict} | {a.f1_lenient} | "
                 f"{a.egg_bad_boundary} | {a.false_positives} | {a.missed_eggs} | {r.runtime_s} |")
    tags = json.loads(res.iloc[0].dev_metrics)["by_tag"]
    L += ["", "## Exact-count rate by difficulty tag (n images)", "", "| tag | " + " | ".join(res.index) + " |", "|---|" + "---|" * len(res)]
    for t in sorted(tags):
        L.append(f"| {t} ({tags[t]['n_images']}) | " + " | ".join(str(json.loads(r.dev_metrics)["by_tag"][t]["count_exact_rate"]) for _, r in res.iterrows()) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = build()
    Path("experiments/selector_results.md").write_text(out)
    print(out)
