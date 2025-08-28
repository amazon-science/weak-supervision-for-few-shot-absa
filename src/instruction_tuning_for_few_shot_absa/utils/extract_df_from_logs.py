import pandas as pd
import glob
import os
import numpy as np
import tensorboard
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def extract_log(path, task, version):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    runlog = pd.DataFrame(columns=["task", "version", "metric", "step", "value"])
    for tag in event_acc.Tags()["scalars"]:
        for e in event_acc.Scalars(tag):
            runlog.loc[len(runlog.index)] = [task, version, tag, e.step, e.value]
    return runlog


event_paths = glob.glob(
    f"<ABSAWithWeaklySupervisedPreTraining_Path>/logs/fs/fewshot_20/*/*/*/events*"
)
print(event_paths)
print(event_paths[0].split("logs")[-1].split("/")[4])
# print(extract_log(event_paths[0], event_paths[0].split('logs')[-1].split('/')[2]))
df = pd.concat(
    [
        extract_log(
            x, x.split("logs")[-1].split("/")[4], x.split("logs")[-1].split("/")[5]
        )
        for x in event_paths
    ]
)
print(df)

task1_mtl = (
    df[(df["metric"] == "task1_f1") & (df["task"] == "mtl")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task2_mtl = (
    df[(df["metric"] == "task2_f1") & (df["task"] == "mtl")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task3_mtl = (
    df[(df["metric"] == "task3_f1") & (df["task"] == "mtl")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task4_mtl = (
    df[(df["metric"] == "task4_f1") & (df["task"] == "mtl")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task5_mtl = (
    df[(df["metric"] == "task5_f1") & (df["task"] == "mtl")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)

task1 = (
    df[(df["metric"] == "f1") & (df["task"] == "task1")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task2 = (
    df[(df["metric"] == "f1") & (df["task"] == "task2")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task3 = (
    df[(df["metric"] == "f1") & (df["task"] == "task3")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task4 = (
    df[(df["metric"] == "f1") & (df["task"] == "task4")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)
task5 = (
    df[(df["metric"] == "f1") & (df["task"] == "task5")]
    .groupby(["metric", "version"])["value"]
    .max()
    .values
)


task1_df = pd.DataFrame(
    np.hstack([task1_mtl[:, None], task1[:, None]]), columns=["mtl", "single"]
).melt(value_vars=["mtl", "single"], var_name="type", value_name="f1")
task2_df = pd.DataFrame(
    np.hstack([task2_mtl[:, None], task2[:, None]]), columns=["mtl", "single"]
).melt(value_vars=["mtl", "single"], var_name="type", value_name="f1")
task3_df = pd.DataFrame(
    np.hstack([task3_mtl[:, None], task3[:, None]]), columns=["mtl", "single"]
).melt(value_vars=["mtl", "single"], var_name="type", value_name="f1")
task4_df = pd.DataFrame(
    np.hstack([task4_mtl[:, None], task4[:, None]]), columns=["mtl", "single"]
).melt(value_vars=["mtl", "single"], var_name="type", value_name="f1")
task5_df = pd.DataFrame(
    np.hstack([task5_mtl[:, None], task5[:, None]]), columns=["mtl", "single"]
).melt(value_vars=["mtl", "single"], var_name="type", value_name="f1")


sns.lineplot(data=task1_df, x="type", y="f1")
plt.title("Task 1")
plt.savefig("fs_20_task1.png")
plt.clf()
plt.cla()
sns.lineplot(data=task2_df, x="type", y="f1")
plt.title("Task 2")
plt.savefig("fs_20_task2.png")
plt.clf()
plt.cla()
sns.lineplot(data=task3_df, x="type", y="f1")
plt.title("Task 3")
plt.savefig("fs_20_task3.png")
plt.clf()
plt.cla()
sns.lineplot(data=task4_df, x="type", y="f1")
plt.title("Task 4")
plt.savefig("fs_20_task4.png")
plt.clf()
plt.cla()
sns.lineplot(data=task5_df, x="type", y="f1")
plt.title("Task 5")
plt.savefig("fs_20_task5.png")
plt.clf()
plt.cla()


exit()


df[(df["metric"] == "task1_f1") & (df["task"] == "mtl")].groupby(
    ["metric", "version"]
).get_group(("task1_f1", "version_0"))["value"].values
df[(df["metric"] == "task1_f1") & (df["task"] == "mtl")].groupby(
    ["metric", "version"]
).get_group(("task1_f1", "version_1"))["value"].values
df[(df["metric"] == "task1_f1") & (df["task"] == "mtl")].groupby(
    ["metric", "version"]
).get_group(("task1_f1", "version_2"))["value"].values
df[(df["metric"] == "task1_f1") & (df["task"] == "mtl")].groupby(
    ["metric", "version"]
).get_group(("task1_f1", "version_3"))["value"].values
df[(df["metric"] == "task1_f1") & (df["task"] == "mtl")].groupby(
    ["metric", "version"]
).get_group(("task1_f1", "version_4"))["value"].values
