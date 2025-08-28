"""
This script is for reading the results from s3
"""

import s3fs
import tarfile
import pickle


def extract_pickle_results_from(path):
    tar = tarfile.open(path, "r:gz")
    member = [x for x in tar.getmembers() if "results_dev.pickle" in x.path][0]
    return pickle.load(tar.extractfile(member))[0]


def download_file_locally(remote, local_name):
    s3.get(remote, local_name)


s3 = s3fs.S3FileSystem(anon=False)

checkpoint_to_number_of_steps = [
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 1)) * 1)),
        "pft_batch_size": 16 * 1,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 2)) * 1)),
        "pft_batch_size": 16 * 2,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 4)) * 1)),
        "pft_batch_size": 16 * 4,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 8)) * 1)),
        "pft_batch_size": 16 * 8,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 16)) * 1)),
        "pft_batch_size": 16 * 16,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 32)) * 1)),
        "pft_batch_size": 16 * 32,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 64)) * 1)),
        "pft_batch_size": 16 * 64,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 128)) * 1)),
        "pft_batch_size": 16 * 128,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 256)) * 1)),
        "pft_batch_size": 16 * 256,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 512)) * 1)),
        "pft_batch_size": 16 * 512,
        "pft_epochs": 1,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 1)) * 2)),
        "pft_batch_size": 16 * 1,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 2)) * 2)),
        "pft_batch_size": 16 * 2,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 4)) * 2)),
        "pft_batch_size": 16 * 4,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 8)) * 2)),
        "pft_batch_size": 16 * 8,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 16)) * 2)),
        "pft_batch_size": 16 * 16,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 32)) * 2)),
        "pft_batch_size": 16 * 32,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 64)) * 2)),
        "pft_batch_size": 16 * 64,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 128)) * 2)),
        "pft_batch_size": 16 * 128,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 256)) * 2)),
        "pft_batch_size": 16 * 256,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 512)) * 2)),
        "pft_batch_size": 16 * 512,
        "pft_epochs": 2,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 1)) * 3)),
        "pft_batch_size": 16 * 1,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 2)) * 3)),
        "pft_batch_size": 16 * 2,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 4)) * 3)),
        "pft_batch_size": 16 * 4,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 8)) * 3)),
        "pft_batch_size": 16 * 8,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 16)) * 3)),
        "pft_batch_size": 16 * 16,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 32)) * 3)),
        "pft_batch_size": 16 * 32,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 64)) * 3)),
        "pft_batch_size": 16 * 64,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 128)) * 3)),
        "pft_batch_size": 16 * 128,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 256)) * 3)),
        "pft_batch_size": 16 * 256,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 512)) * 3)),
        "pft_batch_size": 16 * 512,
        "pft_epochs": 3,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 1)) * 4)),
        "pft_batch_size": 16 * 1,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 2)) * 4)),
        "pft_batch_size": 16 * 2,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 4)) * 4)),
        "pft_batch_size": 16 * 4,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 8)) * 4)),
        "pft_batch_size": 16 * 8,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 16)) * 4)),
        "pft_batch_size": 16 * 16,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 32)) * 4)),
        "pft_batch_size": 16 * 32,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 64)) * 4)),
        "pft_batch_size": 16 * 64,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 128)) * 4)),
        "pft_batch_size": 16 * 128,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 256)) * 4)),
        "pft_batch_size": 16 * 256,
        "pft_epochs": 4,
    },
    {
        "pft_total_number_of_steps": int(((149790 / (16 * 512)) * 4)),
        "pft_batch_size": 16 * 512,
        "pft_epochs": 4,
    },
]


all_results = []
base_path = "<s3_path>"
for checkpoint in range(0, 40):
    checkpoint_name = f"checkpoint_{checkpoint}"
    for dataset in ["rest15", "rest16"]:
        for task in ["task1", "task2", "task3", "task4", "task5"]:
            for seed in [1, 2, 3, 4, 5]:
                files = s3.glob(
                    f"{base_path}/{checkpoint_name}/{dataset}/{task}/seed_{seed}/**/**/output/model.tar.gz"
                )
                if len(files) > 0:
                    download_file_locally(files[0], "file.tar.gz")
                    result = extract_pickle_results_from("file.tar.gz")
                    config = {
                        "checkpoint": checkpoint_name,
                        "dataset": dataset,
                        "task": task,
                        "seed": seed,
                        "precision": result[task]["precision"],
                        "recall": result[task]["recall"],
                        "f1": result[task]["f1"],
                        **checkpoint_to_number_of_steps[checkpoint],
                    }
                    all_results.append(config)
                else:
                    config = {
                        "checkpoint": checkpoint_name,
                        "dataset": dataset,
                        "task": task,
                        "seed": seed,
                    }
                    print(
                        f"Nothing for ",
                        f"{base_path}/{checkpoint_name}/{dataset}/{task}/seed_{seed}/**/**/output/model.tar.gz",
                    )


original_results = []
base_path = "<s3_path>"
for dataset in ["rest15", "rest16"]:
    for task in ["task1", "task2", "task3", "task4", "task5"]:
        for seed in [1, 2, 3, 4, 5]:
            files = s3.glob(
                f"{base_path}/{dataset}/{task}/seed_{seed}/**/**/output/model.tar.gz"
            )
            if len(files) > 0:
                download_file_locally(files[0], "file.tar.gz")
                result = extract_pickle_results_from("file.tar.gz")
                config = {
                    "checkpoint": "original",
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                    "precision": result[task]["precision"],
                    "recall": result[task]["recall"],
                    "f1": result[task]["f1"],
                    "pft_total_number_of_steps": 0,
                    "pft_batch_size": 0,
                    "pft_epochs": 0,
                }
                original_results.append(config)
            else:
                config = {
                    "checkpoint": "original",
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                }
                print(
                    f"Nothing for ",
                    f"{base_path}/{dataset}/{task}/seed_{seed}/**/**/output/model.tar.gz",
                )


with open("checkpoints_results.pickle", "wb+") as fout:
    pickle.dump(all_results, fout)

with open("original_results.pickle", "wb+") as fout:
    pickle.dump(original_results, fout)
