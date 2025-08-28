"""
Extracts the results that were saved in the pickle format
Assumes the following format:
[all_scores, all_inputs, all_labels, all_preds]
"""

import glob
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

"""
Load the pickle results
Return only the scores
"""


def load_results(path):
    with open(path, "rb") as fin:
        data = pickle.load(fin)
    return data[0]


"""
Search recursively for all "results.pickle" in that directory (and its subdirectories)
"""


def get_paths_from_subfolder(subfolder) -> List[str]:
    files = glob.glob(f"{subfolder}/**/results_dev.pickle", recursive=True)
    return files


"""
Extracts the f1 score from the scores
"""


def extract_f1_score(scores: List[Dict[str, Dict[str, float]]], task):
    return [s[task]["f1"] for s in scores]


"""
Average all the scores
"""


def average_scores(scores: List[Dict[str, Dict[str, float]]], task):
    result = {
        "precision_mean": np.mean([s[task]["precision"] for s in scores]),
        "recall_mean": np.mean([s[task]["recall"] for s in scores]),
        "f1_mean": np.mean([s[task]["f1"] for s in scores]),
        "precision_std": np.std([s[task]["precision"] for s in scores]),
        "recall_std": np.std([s[task]["recall"] for s in scores]),
        "f1_std": np.std([s[task]["f1"] for s in scores]),
    }

    return result


"""
Extracts a pandas DataFrame from the paths
"""


def extract_all(paths: List[Tuple[str, str]]):
    results = defaultdict(list)
    for loc, config in paths:
        p1 = [
            (f"{loc}/rest15/task1/", "task1"),
            (f"{loc}/rest15/task2/", "task2"),
            (f"{loc}/rest15/task3/", "task3"),
            (f"{loc}/rest15/task4/", "task4"),
            (f"{loc}/rest15/task5/", "task5"),
        ]

        p2 = [
            (f"{loc}/rest15/mtl/", "task1"),
            (f"{loc}/rest15/mtl/", "task2"),
            (f"{loc}/rest15/mtl/", "task3"),
            (f"{loc}/rest15/mtl/", "task4"),
            (f"{loc}/rest15/mtl/", "task5"),
        ]

        for path, task_name in p1:
            results["f1"].extend(
                extract_f1_score(
                    [load_results(x) for x in get_paths_from_subfolder(path)],
                    task=task_name,
                )
            )
            results["task"].extend([task_name for x in get_paths_from_subfolder(path)])
            results["type"].extend(["single" for x in get_paths_from_subfolder(path)])
            results["config"].extend([config for x in get_paths_from_subfolder(path)])

        for path, task_name in p2:
            results["f1"].extend(
                extract_f1_score(
                    [load_results(x) for x in get_paths_from_subfolder(path)],
                    task=task_name,
                )
            )
            results["task"].extend([task_name for x in get_paths_from_subfolder(path)])
            results["type"].extend(["mtl" for x in get_paths_from_subfolder(path)])
            results["config"].extend([config for x in get_paths_from_subfolder(path)])

    df = pd.DataFrame(results)
    return df


paths = [
    # ("logs/append_the_task_description/", "Append the Task Description", "zz_append_task_description.png", "Append Task Description"),
    # ("logs/original/", "Original", "zz_original.png", "Original"),
    # ("logs/category_change/", "Category Change + No Tuple Word", "zz_category_change.png", "Category Change + No Tuple Word"),
    # ("logs/prompts_without_tuple_words/", "Prompts without Tuple Words", "zz_prompts_without_tuple_words.png", "No Tuple Words"),
    (
        "logs/original_with_lrscheduler",
        "Original with LR",
        "zz_original_with_lr.png",
        "Original with LR",
    ),
]

df = extract_all([(x[0], x[-1]) for x in paths])
exit()
# print(df)
# q = df[df['config'] == 'Original']
# t1 = ((q[(q['task']=='task1') & (q['type']=='mtl')]['f1'].values - q[(q['task']=='task1') & (q['type']=='single')]['f1'].values) * 100).tolist()
# t2 = ((q[(q['task']=='task2') & (q['type']=='mtl')]['f1'].values - q[(q['task']=='task2') & (q['type']=='single')]['f1'].values) * 100).tolist()
# t3 = ((q[(q['task']=='task3') & (q['type']=='mtl')]['f1'].values - q[(q['task']=='task3') & (q['type']=='single')]['f1'].values) * 100).tolist()
# t4 = ((q[(q['task']=='task4') & (q['type']=='mtl')]['f1'].values - q[(q['task']=='task4') & (q['type']=='single')]['f1'].values) * 100).tolist()
# t5 = ((q[(q['task']=='task5') & (q['type']=='mtl')]['f1'].values - q[(q['task']=='task5') & (q['type']=='single')]['f1'].values) * 100).tolist()
# df1 = pd.DataFrame({'task1': t1, 'task2': t2, 'task3': t3, 'task4': t4, 'task5': t5}).melt(var_name='task', value_name='f1')
# # df1 = pd.DataFrame({'task1': [np.mean(t1)], 'task2': [np.mean(t2)], 'task3': [np.mean(t3)], 'task4': [np.mean(t4)], 'task5': [np.mean(t5)]}, columns=['task1', 'task2', 'task3', 'task4', 'task5']).melt(var_name='task', value_name='f1')
# sns.barplot(data=df1, x='task', y='f1', estimator = np.mean, ci=None)
# plt.title("Improvements brought by MTL for each task")
# plt.savefig('zz_delta.png')
# print(t1)
# print(t2)
# print(t3)
# print(t4)
# print(t5)
# exit()
# exit()
# df[(df['task']=='task1') & (df['type']=='single')]['f1'].mean(), df[(df['task']=='task1') & (df['type']=='single')]['f1'].std()
# df[(df['task']=='task2') & (df['type']=='single')]['f1'].mean(), df[(df['task']=='task2') & (df['type']=='single')]['f1'].std()
# df[(df['task']=='task3') & (df['type']=='single')]['f1'].mean(), df[(df['task']=='task3') & (df['type']=='single')]['f1'].std()
# df[(df['task']=='task4') & (df['type']=='single')]['f1'].mean(), df[(df['task']=='task4') & (df['type']=='single')]['f1'].std()
# df[(df['task']=='task5') & (df['type']=='single')]['f1'].mean(), df[(df['task']=='task5') & (df['type']=='single')]['f1'].std()
# print('--------------')
# df[(df['task']=='task1') & (df['type']=='mtl')]['f1'].mean(), df[(df['task']=='task1') & (df['type']=='mtl')]['f1'].std()
# df[(df['task']=='task2') & (df['type']=='mtl')]['f1'].mean(), df[(df['task']=='task2') & (df['type']=='mtl')]['f1'].std()
# df[(df['task']=='task3') & (df['type']=='mtl')]['f1'].mean(), df[(df['task']=='task3') & (df['type']=='mtl')]['f1'].std()
# df[(df['task']=='task4') & (df['type']=='mtl')]['f1'].mean(), df[(df['task']=='task4') & (df['type']=='mtl')]['f1'].std()
# df[(df['task']=='task5') & (df['type']=='mtl')]['f1'].mean(), df[(df['task']=='task5') & (df['type']=='mtl')]['f1'].std()
for _, title, save_path, config in paths:
    sns.barplot(data=df[df["config"] == config], x="task", y="f1", hue="type", ci="sd")
    plt.ylim((0.0, 1.0))
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()

# df = df[df['type'] == 'mtl']
# sns.barplot(data=df, x='task', y='f1', hue='config', ci='sd')
# plt.ylim((0.0, 1.0))
# plt.title("MTL Config Comparison")
# plt.savefig('zz_mtl_comparison.png')
# plt.clf()
