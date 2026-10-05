import pandas as pd
from data_profiling import ProfileReport
from pathlib import Path

fpath_data = Path("data/output")

fpath_summary = fpath_data / "summary.csv"
fpath_full_bench_data = fpath_data / "full_benchmark_raw.csv"

independent_variables = ["config.activation","config.eta", "config.initializer", "config.mini_batch_size", "config.topology"]

dependent_variables_raw = ["run.avg_testing_time", "run.avg_training_time", "run.epochs_to_accuracy_threshold",
                           "run.final_accuracy","run.max_accuracy", "run.time_to_accuracy_threshold", "run.total_training_time",
                           "run.total_testing_time", "run.total_time"]
dependent_variables_summary = ["mean_peak_accuracy", "mean_final_accuracy", "mean_total_time", "mean_training_time", "mean_testing_time",
                               "mean_time_to_accuracy_threshold", "mean_epochs_to_accuracy_threshold"]

all_vars_raw = independent_variables + dependent_variables_raw
all_vars_summary = independent_variables + dependent_variables_summary

df_summary = pd.read_csv(fpath_summary)

profile_summary = ProfileReport(
    df_summary[all_vars_summary],
    title="Performance Analysis of Different DNN Configurations",
    explorative=True,
    interactions={"targets": all_vars_summary},
    correlations={
        "auto": {"calculate": True},
        "phi_k": {"calculate" : True}
    }
)

profile_summary.to_file("profile_summary.html")
