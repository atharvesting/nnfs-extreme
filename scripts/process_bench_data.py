import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse

"""
Config with (average of all seeds):
- max accuracy achieved
- max final accuracy
- min final accuracy
- average accuracy
- earliest accuracy threshold arrival
- latest accuracy threshold arrival (if applicable)
- max run time
- max training time
- max testing time
- min run time
- min training time
- min testing time
"""

def load_data(filename: str) -> pd.DataFrame:
    with open(filename, "r") as file:
        if filename.endswith(".jsonl"):
            results = [
                json.loads(line)  # as opposed to load() which accepts file objects
                for line in file
                if line.strip()  # if line is non-empty after stripping
            ]
        else:
            results = json.load(file)

    completed = [
        row for row in results
        if row.get("status", "completed") == "completed"
    ]

    print(
        f"Completed: {len(completed)}, "
        f"excluded: {len(results) - len(completed)}"
    )

    if not completed:
        raise ValueError("No completed runs to analyze.")

    for row in completed:
        if "run results" in row:
            row["run"] = row.pop("run results")

    return pd.json_normalize(completed)

def summarize_configurations(data: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "mean_peak_accuracy": ("run.max_accuracy", "mean"),
        "mean_final_accuracy": ("run.final_accuracy", "mean"),
        "mean_time_to_accuracy_threshold": ("run.time_to_accuracy_threshold", "mean"),
        "mean_epochs_to_accuracy_threshold": ("run.epochs_to_accuracy_threshold", "mean"),
        "mean_total_time": ("run.total_time", "mean"),
        "mean_training_time": ("run.total_training_time", "mean"),
        "mean_testing_time": ("run.total_testing_time", "mean"),
        "run_count": ("config.random_seed", "size"),
        "seed_count": ("config.random_seed", "nunique"),
    }

    required = {
        "config.random_seed",
        "config.topology",
        *(column for column, _ in aggregations.values()),
    }
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("Cannot summarize an empty results table.")

    group_columns = [
        column for column in data.columns
        if column.startswith("config.")
            and column != "config.random_seed"
    ]

    working = data.copy()
    working["config.topology"] = working["config.topology"].map(tuple)

    return (
        working.groupby(group_columns, as_index=False, dropna=False)
        .agg(**aggregations)
        .sort_values("mean_final_accuracy", ascending=False)
        .reset_index(drop=True)
    )

def accuracy_extremes(summary: pd.DataFrame) -> pd.DataFrame:

    if summary.empty:
        raise ValueError("Empty summary table cannot be analyzed.")

    mean_final_acc_series = summary["mean_final_accuracy"]

    mask = (mean_final_acc_series == mean_final_acc_series.max()) \
           | (mean_final_acc_series == mean_final_acc_series.min())

    extremes = summary.loc[mask]

    return extremes

result_filename = "data/output/results_2026-10-03_19-40-41_021_UTC_merged.jsonl"
data = load_data(result_filename)
# data.to_csv("data/output/full_benchmark_raw.csv")
summary = summarize_configurations(data)
# extremes = accuracy_extremes(summary)
# print(summary)
# print(extremes)
summary.to_csv("summary.csv")
# extremes.to_csv("extremes.csv")
