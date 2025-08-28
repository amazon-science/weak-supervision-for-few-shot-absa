"""
The role of this script is to take the predictions of the model (already produced, this script does not produce them)
and to convert them to per-token predictions. Then, compute metrics to allow comparison with previous state-of-the-art
"""

from collections import defaultdict
import pickle
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pandas as pd


def calculate_ate_sent_score():
    for path2, dataset in [
        ("data_as_jsonl/zero_shot_absa_data/rest_total_test.jsonl", "rest_total_test"),
        ("data_as_jsonl/zero_shot_absa_data/lap14_test.jsonl", "lap14_test"),
    ]:
        f1_scores = []
        for pft_seed in [1, 10, 100]:
            for seed in [123, 262, 401, 540, 679]:
                path1 = f"logs/zero_shot_exp/eval_results_t5base/{dataset}_ate_sent_{seed}_{pft_seed}.pickle"
                with open(path1, "rb") as fin:
                    data = pickle.load(fin)

                sent_map = {
                    "great": "T-POS",
                    "ok": "T-NEU",
                    "bad": "T-NEG",
                    "": "O",
                }
                pred = []
                gold = []
                with open(path2, "r") as fin:
                    for original_line, line in zip(
                        fin.readlines(), data[3]["ate_sent"]
                    ):
                        quads = defaultdict(list)
                        loaded = json.loads(original_line)
                        sentence_split = loaded["sentence"].split(" ")
                        for x in line:
                            for y in x[0].split(" "):
                                quads[y].append(sent_map.get(x[1], "O"))

                        for token in sentence_split:
                            if token in quads and len(quads[token]) > 0:
                                pred.append(quads[token][0])
                                quads[token] = quads[token][1:]
                            else:
                                pred.append("O")

                with open(path2, "r") as fin:
                    for original_line, line in zip(
                        fin.readlines(), data[2]["ate_sent"]
                    ):
                        quads = defaultdict(list)
                        loaded = json.loads(original_line)
                        sentence_split = loaded["sentence"].split(" ")
                        for x in line:
                            for y in x[0].split(" "):
                                quads[y].append(sent_map.get(x[1], "O"))

                        for token in sentence_split:
                            if token in quads and len(quads[token]) > 0:
                                gold.append(quads[token][0])
                                quads[token] = quads[token][1:]
                            else:
                                gold.append("O")

                # print('------------')
                # print('ate_sent')
                # print(path1.split('/')[-1], ',', path2.split('/')[-1])
                # print(f1_score(gold, pred, average='macro') * 100)
                # print(f1_score(gold, pred, average=None) * 100)
                # print(accuracy_score(gold, pred) * 100)
                # print(confusion_matrix(gold, pred))
                # print('------------')
                f1_scores.append(f1_score(gold, pred, average="macro") * 100)
                # # print("\n\n\n\n\n\n")
        print("-----------------")
        print(path2)
        print(pd.Series(f1_scores).mean(), "", pd.Series(f1_scores).std())
        print("-----------------")


def calculate_ate_score():
    for path2, dataset in [
        ("data_as_jsonl/zero_shot_absa_data/rest_total_test.jsonl", "rest_total_test"),
        ("data_as_jsonl/zero_shot_absa_data/lap14_test.jsonl", "lap14_test"),
    ]:
        accuracy_scores = []
        for pft_seed in [1, 10, 100]:
            for seed in [123, 262, 401, 540, 679]:
                path1 = f"logs/zero_shot_exp/eval_results_t5base/{dataset}_ate_{seed}_{pft_seed}.pickle"
                with open(path1, "rb") as fin:
                    data = pickle.load(fin)

                pred = []
                gold = []
                with open(path2, "r") as fin:
                    for original_line, line in zip(fin.readlines(), data[3]["ate"]):
                        # Could have used a defaultdict(int), but we chose to keep the code as similar as possible with the previous function
                        quads = defaultdict(list)
                        loaded = json.loads(original_line)
                        sentence_split = loaded["sentence"].split(" ")
                        for x in line:
                            for y in x.split(" "):
                                quads[y].append(x)

                        for token in sentence_split:
                            if token in quads and len(quads[token]) > 0:
                                pred.append("B")
                                quads[token] = quads[token][1:]
                            else:
                                pred.append("O")

                with open(path2, "r") as fin:
                    for original_line, line in zip(fin.readlines(), data[2]["ate"]):
                        # Could have used a defaultdict(int), but we chose to keep the code as similar as possible with the previous function
                        quads = defaultdict(list)
                        loaded = json.loads(original_line)
                        sentence_split = loaded["sentence"].split(" ")
                        for x in line:
                            for y in x.split(" "):
                                quads[y].append(x)

                        for token in sentence_split:
                            if token in quads and len(quads[token]) > 0:
                                gold.append("B")
                                quads[token] = quads[token][1:]
                            else:
                                gold.append("O")

                # print('------------')
                # print('ate')
                # print(path1.split('/')[-1], ',', path2.split('/')[-1])
                # print(f1_score(gold, pred, average='macro') * 100)
                # print(f1_score(gold, pred, average=None) * 100)
                # print(accuracy_score(gold, pred) * 100)
                # print(confusion_matrix(gold, pred))
                # exit()
                # print('------------')
                accuracy_scores.append(accuracy_score(gold, pred) * 100)
        print("-----------------")
        print(path2)
        print(pd.Series(accuracy_scores).mean(), "", pd.Series(accuracy_scores).std())
        print("-----------------")


if __name__ == "__main__":
    calculate_ate_score()
    calculate_ate_sent_score()
