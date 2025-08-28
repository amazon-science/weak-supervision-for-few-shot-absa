"""
Converts the data from the triplet format used by https://github.com/xuuuluuu/Position-Aware-Tagging-for-ASTE/ to
the format used in this code (quad)
Appends "LAPTOP" as the category
"""

sentiment_map = {
    "POS": "positive",
    "NEG": "negative",
    "NEU": "neutral",
}


def convert_line_to_quad(line: str) -> str:
    sentence, tuples = line.split("####")
    words = sentence.split(" ")
    labels = eval(tuples)
    new_labels = []
    for aspect_term_indices, opinion_term_indices, sentiment in labels:
        aspect_term = " ".join([words[x] for x in aspect_term_indices])
        opinion_term = " ".join([words[x] for x in opinion_term_indices])
        new_labels.append(
            [aspect_term, "restaurant", sentiment_map[sentiment], opinion_term]
        )
    return "####".join([sentence, str(new_labels)])


for source_path, output_path in [
    ("data2/original/15res/train_triplets.txt", "data2/converted/15res/train.txt"),
    ("data2/original/15res/dev_triplets.txt", "data2/converted/15res/dev.txt"),
    ("data2/original/15res/test_triplets.txt", "data2/converted/15res/test.txt"),
    ("data2/original/16res/train_triplets.txt", "data2/converted/16res/train.txt"),
    ("data2/original/16res/dev_triplets.txt", "data2/converted/16res/dev.txt"),
    ("data2/original/16res/test_triplets.txt", "data2/converted/16res/test.txt"),
]:
    with open(source_path) as fin:
        lines = []
        for line in fin:
            lines.append(convert_line_to_quad(line))
        with open(output_path, "w+") as fout:
            for line in lines:
                _ = fout.write(f"{line}\n")
exit()

with open("data2/original/15res/train_triplets.txt") as fin:
    lines = []
    for line in fin:
        lines.append(convert_line_to_quad(line))
    with open("data2/converted/15res/train.txt", "w+") as fout:
        for line in lines:
            _ = fout.write(f"{line}\n")

with open("data/laptop_original/dev.txt") as fin:
    lines = []
    for line in fin:
        lines.append(convert_line_to_quad(line))
    with open("data/lap14/dev.txt", "w+") as fout:
        for line in lines:
            _ = fout.write(f"{line}\n")

with open("data/laptop_original/test.txt") as fin:
    lines = []
    for line in fin:
        lines.append(convert_line_to_quad(line))
    with open("data/lap14/test.txt", "w+") as fout:
        for line in lines:
            _ = fout.write(f"{line}\n")
