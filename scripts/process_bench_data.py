import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse

"""
Config with (average of all seeds):
- max accuracy achieved
- min accuarcy achieved
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
parser = argparse.ArgumentParser()
parser.add_argument("results", nargs="?", default="results_1.json")
result_filename = parser.parse_args().results

with open(result_filename, "r") as res:
    results = [json.loads(line) for line in res if line.strip()] if result_filename.endswith(".jsonl") else json.load(res)

failed = [row for row in results if row.get("status", "completed") != "completed"]
print(f"Completed: {len(results) - len(failed)}, failed: {len(failed)}")
results = [row for row in results if row.get("status", "completed") == "completed"]
if not results:
    raise SystemExit("No completed runs to analyze.")
for row in results:
    if "run results" in row:
        row["run"] = row.pop("run results")

data = pd.json_normalize(results)
print(data[["config.random_seed", "run.max_accuracy"]])
best_accuracy = data["run.max_accuracy"].max()
mask = data["run.max_accuracy"] == best_accuracy

print(best_accuracy)
print(mask)
print(data.loc[mask, ["config.random_seed", "run.max_accuracy"]])

config_columns = [
    column for column in data.columns
    if column.startswith("config.")
]

print(data.loc[mask, config_columns].to_string(index=False))

lowest_final = data["run.final_accuracy"].min()
mask = data["run.final_accuracy"] == lowest_final

print(data.loc[mask, config_columns])

group_columns = [
    column for column in config_columns
    if column != "config.random_seed"
]

data["config.topology"] = data["config.topology"].map(tuple)

groups = data.groupby(group_columns)
final_accuracies = groups["run.final_accuracy"]
means = final_accuracies.mean()

print("Means", means)

summary = means.reset_index()
print("Summary", summary.to_string(index=False))

progression_attributes = ["run.training_time_progression", "run.testing_time_progression", "run.accuracy_progression"]


def get_stats():
    global data
    max_accuracy = data[data["run.max_accuracy"] == data["run.max_accuracy"].max()]
    min_accuracy = data[data["run.min_accuracy"] == data["run.min_accuracy"].max()]

def progression_comparison():
    global data

    for run in data:
        plt.plot()

