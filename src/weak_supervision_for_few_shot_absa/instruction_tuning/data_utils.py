from collections import defaultdict
from string import Template
import random

from torch.utils.data import Dataset

sentword2opinion = {"positive": "great", "negative": "bad", "neutral": "ok"}

categories2text = {
    "RESTAURANT#GENERAL": "restaurant in general is",
    "SERVICE#GENERAL": "service in general is",
    "FOOD#QUALITY": "food quality is",
    "FOOD#STYLE_OPTIONS": "food style options are",
    "DRINKS#STYLE_OPTIONS": "drinks style options are",
    "DRINKS#PRICES": "drinks prices are",
    "RESTAURANT#PRICES": "restaurant prices are",
    "AMBIENCE#GENERAL": "ambience in general is",
    "RESTAURANT#MISCELLANEOUS": "restaurant miscellaneous is",
    "FOOD#PRICES": "food prices are",
    "LOCATION#GENERAL": "location in general is",
    "DRINKS#QUALITY": "drinks quality is",
    "FOOD#GENERAL": "food in general is",
    "laptop": "laptop",
    "LAPTOP": "laptop",
}


absa_quad_text2category = {
    "location general": "LOCATION#GENERAL",
    "food prices": "FOOD#PRICES",
    "food quality": "FOOD#QUALITY",
    "food general": "FOOD#GENERAL",
    "ambience general": "AMBIENCE#GENERAL",
    "service general": "SERVICE#GENERAL",
    "restaurant prices": "RESTAURANT#PRICES",
    "drinks prices": "DRINKS#PRICES",
    "restaurant miscellaneous": "RESTAURANT#MISCELLANEOUS",
    "drinks quality": "DRINKS#QUALITY",
    "drinks style_options": "DRINKS#STYLE_OPTIONS",
    "restaurant general": "RESTAURANT#GENERAL",
    "food style_options": "FOOD#STYLE_OPTIONS",
    "laptop": "laptop",
    "LAPTOP": "laptop",
}

main_templates = {
    1: {
        # "input1": "What are the terms in the text \"$TEXT\" that have opinion expressed towards them?",
        # "input2": "In the following text : \"$TEXT\", what are the terms that have opinion expressed towards them?",
        # "input3": "In the following text : \"$TEXT\", what are the real world entities that are worthy of a subjective opinion?","
        "inputs": ["input1", "input2"],
        "input1": "Given the text: $TEXT, what are the aspect terms in it ?",
        "input2": "What are the aspect terms in the text: $TEXT ?",
        "phrases": ["aspect terms"],
        "output": "$AT",
        "outputs": ["output"],
    },
    2: {
        "input1": "Given the text: $TEXT, what are the aspect terms and their sentiments ?",
        "input2": "What are the aspect terms and their sentiments in the text: $TEXT ?",
        "input3": "Given the text: $TEXT, what are the aspect term, sentiment pairs ?",
        "input4": "What are the aspect term, sentiment pairs in the text: $TEXT ?",
        "inputs": ["input1", "input2", "input3", "input4"],
        "phrases": [
            "aspect terms and their sentiments",
            "aspect term, sentiment pairs",
        ],
        "output": "$AT is $S",
        "outputs": ["output"],
    },
    3: {
        "input1": "Given the text: $TEXT, what are the aspect term, sentiment and category triplets ?",
        "input2": "What are the aspect term, sentiment and category triplets in the text: $TEXT ?",
        "input3": "Given the text: $TEXT, what are the aspect term, category and sentiment triplets ?",
        "input4": "What are the aspect term, category and sentiment triplets in the text: $TEXT ?",
        "inputs": ["input1", "input2", "input3", "input4"],
        "phrases": [
            "aspect term, sentiment and category triplets",
            "aspect term, category and sentiment triplets",
        ],
        "output1": "$AT is $S means $AC $S",
        "output2": "$AC $S because $AT is $S",
        "outputs": ["output1"],
    },
    4: {
        "input1": "Given the text: $TEXT, what are the aspect term, opinion term and sentiment triplets ?",
        "input2": "What are the aspect term, opinion term and sentiment triplets in the text: $TEXT ?",
        "input3": "Given the text: $TEXT, what are the opinion term, aspect term and sentiment triplets ?",
        "input4": "What are the opinion term, aspect term and sentiment triplets in the text: $TEXT ?",
        "inputs": ["input1", "input2", "input3", "input4"],
        "phrases": [
            "aspect term, opinion term and sentiment triplets",
            "opinion term, aspect term and sentiment triplets",
        ],
        "output1": "$AT is $OT means $DOMAIN is $S",
        "output2": "$DOMAIN is $S because $AT is $OT",
        "outputs": ["output1"],
    },
    5: {
        # the templates are such that aspect term, opinion term always comes before sentiment and category
        "input1": "Given the text: $TEXT, what are the aspect term, opinion term, sentiment and category quadruples ?",
        "input2": "What are the aspect term, opinion term, sentiment and category quadruples in the text: $TEXT ?",
        "input3": "Given the text: $TEXT, what are the aspect term, opinion term, category and sentiment quadruples ?",
        "input4": "What are the aspect term, opinion term, category and sentiment quadruples in the text: $TEXT ?",
        "input5": "Given the text: $TEXT, what are the opinion term, aspect term, sentiment and category quadruples ?",
        "input6": "What are the opinion term, aspect term, sentiment and category quadruples in the text: $TEXT ?",
        "input7": "Given the text: $TEXT, what are the opinion term, aspect term, category and sentiment quadruples ?",
        "input8": "What are the opinion term, aspect term, category and sentiment quadruples in the text: $TEXT ?",
        "inputs": [
            "input1",
            "input2",
            "input3",
            "input4",
            "input5",
            "input6",
            "input7",
            "input8",
        ],
        "phrases": [
            "aspect term, opinion term, sentiment and category quadruples",
            "aspect term, opinion term, category and sentiment quadruples",
            "opinion term, aspect term, sentiment and category quadruples",
            "opinion term, aspect term, category and sentiment quadruples",
        ],
        "output1": "$AT is $OT means $AC $S",
        "output2": "$AC $S because $AT is $OT",
        "outputs": ["output1"],
    },
}


additional_templates = {
    1: {
        "input1": "For the aspect term: $AT in the text: $TEXT, what is the sentiment ?",
        "input2": "What is the sentiment towards the aspect term $AT in the text: $TEXT ?",
        "inputs": ["input1", "input2"],
        "output": "$AT is $S",
        "outputs": ["output"],
    },
    2: {
        "input1": "For the aspect term: $AT and opinion term $OT in the text: $TEXT, what is the sentiment ?",
        "input2": "what is the sentiment towards the aspect term $AT in the text: $TEXT where the opinion term is $OT ?",
        "input3": "what is the sentiment for the aspect term: $AT and opinion term $OT in the text: $TEXT ?",
        "inputs": ["input1", "input2", "input3"],
        "output": "$AT is $S",
        "outputs": ["output"],
    },
    3: {
        "input1": "For the aspect term: $AT and opinion term $OT in the text: $TEXT, what is the category ?",
        "input2": "What is the category for the aspect term: $AT and opinion term $OT in the text: $TEXT ?",
        "inputs": ["input1", "input2"],
        "output": "$AC $S",
        "outputs": ["output"],
    },
}


def read_acos_from_file(data_path):
    pass


def read_absa_quad_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[str]], List[List[Tuple]], Dict
    """
    all_sents, all_labels = [], []
    unique_labels = defaultdict(int)
    with open(data_path, "r", encoding="UTF-8") as fp:
        words = []
        for line in fp:
            line = line.strip()
            if line != "":
                words, tuples = line.split("####")
                all_sents.append(words.split())
                tmp_labels = eval(tuples)
                new_labels = []
                for label in tmp_labels:
                    at, ac, sp, ot = label
                    if at == "NULL":
                        at = "none"
                    if ot == "NULL":
                        ot = "none"
                    if "#" not in ac:
                        ac = absa_quad_text2category[ac]
                    unique_labels[ac] += 1
                    new_labels.append((at.lower(), ac, sp, ot.lower()))
                all_labels.append(new_labels)
    return all_sents, all_labels, unique_labels


def get_input_output_pairs_main_task(tokens, annotations, task):
    # for each template, generate input, output pairs
    text = " ".join(tokens)
    inputs = []
    outputs = []
    for template_id, template in main_templates.items():
        if task != "mtl" and task != f"task{template_id}":
            continue
        input_key = random.choice(template["inputs"])
        output_key = random.choice(template["outputs"])
        input = Template(template[input_key]).substitute(TEXT=text)

        output = []
        for annotation in annotations:
            at, ac, sp, ot = annotation
            # WHAT DO WE DO IF ot is 'none'
            if at == "none":  # for implicit aspect term
                at = "it"
            ac = categories2text[ac]
            sp_op = sentword2opinion[sp]
            if template_id == 1:
                output.append(Template(template[output_key]).substitute(AT=at))
            elif template_id == 2:
                output.append(Template(template[output_key]).substitute(AT=at, S=sp_op))
            elif template_id == 3:
                output.append(
                    Template(template[output_key]).substitute(AT=at, S=sp_op, AC=ac)
                )
            elif template_id == 4:
                output.append(
                    Template(template[output_key]).substitute(
                        AT=at, OT=ot, S=sp_op, DOMAIN="restaurant"
                    )
                )
            elif template_id == 5:
                output.append(
                    Template(template[output_key]).substitute(
                        AT=at, S=sp_op, AC=ac, OT=ot
                    )
                )
        output = " [SSEP] ".join(output)
        inputs.append(input)
        outputs.append(output)
    return inputs, outputs


def get_input_output_pairs_additional_tasks(tokens, annotations):
    # for each template, generate input, output pairs
    text = " ".join(tokens)
    inputs = []
    outputs = []
    for template_id, template in additional_templates.items():
        for annotation in annotations:
            at, ac, sp, ot = annotation
            # WHAT DO WE DO IF ot is 'none'
            if at == "none":  # for implicit aspect term
                continue
            ac = categories2text[ac]
            sp_op = sentword2opinion[sp]
            input_key = random.choice(template["inputs"])
            output_key = random.choice(template["outputs"])
            if template_id == 1:
                input = Template(template[input_key]).substitute(AT=at, TEXT=text)
                output = Template(template[output_key]).substitute(AT=at, S=sp_op)
            elif template_id == 2:
                input = Template(template[input_key]).substitute(
                    AT=at, OT=ot, TEXT=text
                )
                output = Template(template[output_key]).substitute(AT=at, S=sp_op)
            elif template_id == 3:
                input = Template(template[input_key]).substitute(
                    AT=at, OT=ot, TEXT=text
                )
                output = Template(template[output_key]).substitute(AC=ac, S=sp_op)
            inputs.append(input)
            outputs.append(output)
    return inputs, outputs


def get_transformed_io(data_path, data_type, task):
    """
    The main function to transform input & target according to the task
    """
    # sents, labels = read_line_examples_from_file(data_path)
    all_tokens, all_annotations, _ = read_absa_quad_from_file(data_path)

    # the input is just the raw sentence
    # inputs = [s.copy() for s in sents]
    all_inputs = []
    all_targets = []

    for tokens, annotations in zip(all_tokens, all_annotations):
        main_inputs, main_targets = get_input_output_pairs_main_task(
            tokens, annotations, task
        )
        all_inputs += main_inputs
        all_targets += main_targets
        # if data_type == "train":
        # experiment with additional tasks during training
        # addtl_inputs, addtl_outputs = get_input_output_pairs_additional_tasks(tokens, annotations)
        # all_inputs += addtl_inputs
        # all_targets += addtl_outputs

    return all_inputs, all_targets


class ABSADataset(Dataset):
    def __init__(
        self, base_path, tokenizer, data_dir, data_type, max_len=128, task="mtl"
    ):
        # './data/rest16/train.txt'
        self.base_path = base_path
        self.data_path = f"{base_path}/data/{data_dir}/{data_type}.txt"
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.data_type = data_type
        self.task = task

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze
        target_mask = self.targets[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_type, self.task)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = inputs[i]
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
