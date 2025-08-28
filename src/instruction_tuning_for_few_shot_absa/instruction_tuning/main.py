# -*- coding: utf-8 -*-

import argparse
from ast import literal_eval
import logging
import os
import pickle
import random
import json
import glob
from pathlib import Path
from shutil import copyfile
from typing import Dict, List
from string import Template

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)

from src.weak_supervision_for_few_shot_absa.instruction_tuning.eval_utils import (
    compute_scores,
)
from src.weak_supervision_for_few_shot_absa.instruction_tuning.templates import (
    main_templates,
    noisy_templates,
    categories2text,
    sentword2opinion,
    template_map,
)
from src.weak_supervision_for_few_shot_absa.utils.pl_utils import (
    SavingPointsCheckpoint,
)

logger = logging.getLogger(__name__)


class T5FineTuner2(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, hparams):
        super(T5FineTuner2, self).__init__()
        self.hyperparameters = hparams
        self.model_name_to_params = {
            "t5-base": (T5ForConditionalGeneration, T5Tokenizer),
            "t5-large": (T5ForConditionalGeneration, T5Tokenizer),
            "google/flan-t5-base": (T5ForConditionalGeneration, T5Tokenizer),
            "google/flan-t5-large": (T5ForConditionalGeneration, T5Tokenizer),
            "facebook/bart-large": (BartForConditionalGeneration, BartTokenizer),
            "facebook/bart-base": (BartForConditionalGeneration, BartTokenizer),
        }

        # A small indirection to support t5 and bart as well, while retaining the ability
        # to load models from local file; Ideally, loading from local files would be
        # supported with all models, but there is no `AutoModelForConditionalGeneration` for
        # the transformers library used here
        if self.hyperparameters["model_name_or_path"] in self.model_name_to_params:
            (model_constructor, tokenizer_constructor) = self.model_name_to_params[
                self.hyperparameters["model_name_or_path"]
            ]
        elif os.path.exists(self.hyperparameters["model_name_or_path"]):
            (model_constructor, tokenizer_constructor) = (
                T5ForConditionalGeneration,
                T5Tokenizer,
            )
        else:
            model_name = self.hyperparameters["model_name_or_path"]
            raise ValueError(
                f"The model name that was supplied ({model_name}) is neither inside the supported models ({list(self.model_name_to_params.keys())}), nor it is a valid path in the system. Note: For a path in the system, we only support T5"
            )

        self.model = model_constructor.from_pretrained(
            self.hyperparameters["model_name_or_path"]
        )

        if self.hyperparameters.get("use_weight_distance_factor", False):
            orig_model = model_constructor.from_pretrained(
                self.hyperparameters["model_name_or_path"]
            )
            self.original_parameters = torch.cat(
                [x.view(-1) for x in orig_model.parameters()]
            ).detach()

        self.tokenizer = tokenizer_constructor.from_pretrained(
            self.hyperparameters["model_name_or_path"]
        )
        self.save_hyperparameters("hparams")

        # Simple dictionary to allow switching different types of templates
        # Useful because we might want during our pre-finetuning step to use different
        # templates than the ones we will use during are finetuning step on the task

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # print(self.original_parameters.to(self.device))
        # print(current_params)
        # print(reg_loss)
        self.log("train_loss_generation", loss)
        if self.hyperparameters.get("use_weight_distance_factor", False):
            current_params = torch.cat([x.view(-1) for x in self.model.parameters()])
            reg_loss = torch.linalg.vector_norm(
                current_params - self.original_parameters.to(self.device), 2
            )
            total_loss = (
                loss
                + self.hyperparameters.get("weight_distance_factor", 0.01) * reg_loss
            )
            self.log("train_loss_generation", loss)
            self.log("train_loss_reg", reg_loss)
            self.log("train_loss", total_loss, on_step=True, on_epoch=True)
            return total_loss
        else:
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return loss

    # Do the generation for the given batch
    def generate_for_batch(self, batch):
        self.model.eval()
        outs = self.model.generate(
            input_ids=batch["source_ids"].to(self.device),
            attention_mask=batch["source_mask"].to(self.device),
            max_length=128,
        )  # num_beams=8, early_stopping=True)

        dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        batch_inputs = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["source_ids"]
        ]

        return {
            "outputs": dec,
            "targets": target,
            "inputs": batch_inputs,
        }

    def validation_step(self, batch, batch_idx):
        # Do the generation with f1 score if flag is set. Otherwise set to {}
        # to allow for a uniform return statement
        if self.hyperparameters["evaluate_on_generation"]:
            generated_output = self.generate_for_batch(batch)
        else:
            generated_output = {}

        loss = self._step(batch)

        return {"val_loss": loss, **generated_output}

    def validation_epoch_end(self, val_outputs):
        avg_loss = torch.stack([o["val_loss"] for o in val_outputs]).mean().item()

        self.log("val_loss", avg_loss, prog_bar=True)

        if self.hyperparameters["evaluate_on_generation"]:
            # Calculate the F1 score on the tasks
            inputs, outputs, targets = [], [], []
            for o in val_outputs:
                outputs.extend(o["outputs"])
                targets.extend(o["targets"])
                inputs.extend(o["inputs"])

            # all_scores, _, _, _ = compute_scores(inputs, outputs, targets, template_map[self.hyperparameters['template_name']], verbose = self.hyperparameters['verbose_output'], lower = self.hyperparameters['lower_at_ot'])
            all_scores, all_inputs, all_labels, all_preds = compute_scores(
                inputs,
                outputs,
                targets,
                template_map[self.hyperparameters["template_name"]],
                verbose=self.hyperparameters["verbose_output"],
                lower=self.hyperparameters["lower_at_ot"],
                default_task=(
                    self.hyperparameters["task"]
                    if self.hyperparameters["use_default_task"]
                    else None
                ),
            )
            # print("\n\n-----------")
            # print(all_scores)
            # print(all_inputs)
            # print(all_labels)
            # print(all_preds)
            # print("-----------\n\n")

            if self.hyperparameters["task"] == "mtl":
                f1_score = (
                    all_scores[self.hyperparameters["task_mtl_monitor"]]["f1"] * 100
                )
            else:
                f1_score = all_scores[self.hyperparameters["task"]]["f1"] * 100

            # print(all_scores)
            self.log("f1", f1_score, prog_bar=True)

            for task_name in all_scores:
                self.log(f"{task_name}_p", all_scores[task_name]["precision"] * 100)
                self.log(f"{task_name}_r", all_scores[task_name]["recall"] * 100)
                self.log(f"{task_name}_f1", all_scores[task_name]["f1"] * 100)

            return {"val_loss": avg_loss, "f1": f1_score}
        else:
            return {"val_loss": avg_loss}

    def prepare_optimizers_1(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyperparameters["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adafactor(
            optimizer_grouped_parameters, relative_step=True, warmup_init=True, lr=None
        )
        return optimizer

    def prepare_optimizers_2(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyperparameters["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hyperparameters["learning_rate"],
            eps=self.hyperparameters["adam_epsilon"],
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameters["warmup_steps"],
            num_training_steps=self.hyperparameters["num_training_steps"],
        )
        print("-----------")
        print(
            self.hyperparameters["warmup_steps"],
            self.hyperparameters["num_training_steps"],
        )
        print("-----------")
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                    "reduce_on_plataeu": False,
                }
            ],
        )

    def configure_optimizers(self):
        # Allow for multiple optimizer settings via CL argument
        if self.hyperparameters.get("optimizer_type", 1) == 1:
            return self.prepare_optimizers_1()
        elif self.hyperparameters.get("optimizer_type", 1) == 2:
            return self.prepare_optimizers_2()
        else:
            value = self.hyperparameters.get("optimizer_type", 1)
            raise ValueError(
                f"Make sure you change the code to support a new type of optimization procedure (Used {value}; but only 1 and 2 are supported)"
            )

    @staticmethod
    def custom_collate_fn(tokenizer, batch, max_length=160):
        tokenized_input = tokenizer.batch_encode_plus(
            [x["inputs"] for x in batch],
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_target = tokenizer.batch_encode_plus(
            [x["outputs"] for x in batch],
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "source_ids": tokenized_input["input_ids"],
            "source_mask": tokenized_input["attention_mask"],
            "target_ids": tokenized_target["input_ids"],
            "target_mask": tokenized_target["attention_mask"],
        }

    def train_dataloader(self):
        train_dataset = get_dataset(
            self.hyperparameters["train_dataset_path"],
            self.hyperparameters["task"],
            self.hyperparameters["template_name"],
            dropout=self.hyperparameters["generated_sequence_dropout_probability"],
            to_lower=self.hyperparameters["lower_at_ot"],
            default_domain=self.hyperparameters["default_domain"],
            skip_tasks_for_mtl=self.hyperparameters["skip_tasks_for_mtl"],
        )  # .select(range(50000))
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.hyperparameters["train_batch_size"],
            drop_last=self.hyperparameters["drop_last_batch"],
            shuffle=True,
            num_workers=8,
            prefetch_factor=self.hyperparameters["prefetch_factor"],
            collate_fn=lambda x: self.custom_collate_fn(
                self.tokenizer, x, max_length=self.hyperparameters["max_seq_length"]
            ),
        )
        return dataloader

    def val_dataloader(self):
        if self.hyperparameters["val_dataset_path"]:
            val_dataset = get_dataset(
                self.hyperparameters["val_dataset_path"],
                self.hyperparameters["task"],
                self.hyperparameters["template_name"],
                dropout=0.0,
                to_lower=self.hyperparameters["lower_at_ot"],
                default_domain=self.hyperparameters["default_domain"],
                skip_tasks_for_mtl=self.hyperparameters["skip_tasks_for_mtl"],
            )  # .select(range(5000)) # No dropout for val

            dataloader = DataLoader(
                val_dataset,
                batch_size=self.hyperparameters["eval_batch_size"],
                num_workers=8,
                prefetch_factor=self.hyperparameters["prefetch_factor"],
                collate_fn=lambda x: self.custom_collate_fn(
                    self.tokenizer, x, max_length=self.hyperparameters["max_seq_length"]
                ),
            )
            return dataloader
        else:
            return None


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=False,
        help="A configuration file of type json. It will override the command-line arguments (this is to allow defaults in command-line arguments)",
    )
    parser.add_argument(
        "--sagemaker_config_file",
        default=None,
        type=str,
        required=False,
        help="A configuration file of type json with a very specific format. This is as a workaround sagemaker transforming every parameter to string",
    )
    parser.add_argument(
        "--task",
        default="mtl",
        type=str,
        required=False,
        choices=[
            "task1",
            "task2",
            "task3",
            "task4",
            "task5",
            "ate",
            "ote",
            "ate_ote",
            "ate_sent",
            "ate_ote_sent",
            "mtl",
        ],
        help="The name of the task, selected from: [task1, task2, task3, task4, task5, mtl]",
    )
    parser.add_argument(
        "--skip_tasks_for_mtl",
        choices=[
            "task1",
            "task2",
            "task3",
            "task4",
            "task5",
            "ate",
            "ote",
            "ate_ote",
            "ate_ote_sent",
        ],
        nargs="*",
        default=[],
        help="For laptop domain, not all tasks (i.e. `task1`, `task2`, `task3`, `task4`, `task5`; `task3` and `task5` are not available) are available. As such, we cannot do MTL using all tasks. Therefore, we allow for certain tasks to be skipped. This has effect only when `task` is set to `mtl`",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="t5-base",
        type=str,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_direct_eval",
        action="store_true",
        help="Whether to run eval on the dev/test set.",
    )
    parser.add_argument(
        "--do_inference",
        action="store_true",
        help="Whether to run inference with trained checkpoints",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="checkpoint path (required if --do_inference is passed)",
    )
    # other parameters
    parser.add_argument("--max_seq_length", default=160, type=int)
    parser.add_argument(
        "--n_gpu", nargs="+", type=int, default=[0], help="List of gpus to use"
    )

    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument(
        "--num_train_epochs",
        default=None,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="the number of batches to prefetch in torch.utils.data.DataLoader",
    )
    parser.add_argument(
        "--log_save_name",
        default="",
        type=str,
        required=False,
        help="where to save the checkpoints; defaults to pytorch-lightning default location",
    )
    parser.add_argument(
        "--log_save_dir",
        default="logs",
        type=str,
        required=False,
        help="the model's output directory; defaults to 'logs'",
    )
    parser.add_argument(
        "--verbose_output",
        action="store_true",
        help="whether to print additional info or not; additional info consists of: (i) performance; (ii) examples",
    )
    parser.add_argument(
        "--not_save_checkpoint",
        action="store_true",
        help="if set it will delete the checkpoint at the end; useful for few-shot evaluation, for example, where we do not really need all the models",
    )
    parser.add_argument(
        "--drop_last_batch",
        action="store_true",
        help="whether to keep the last batch or not; it will be passed to `drop_last` parameter in the Dataloader",
    )
    parser.add_argument(
        "--description",
        default=None,
        type=str,
        required=False,
        help="A short description of the model trained. Does not have any effect on the training procedure, but can help with additional information of how/why/when/etc the model was trained",
    )
    parser.add_argument(
        "--reload_dataloaders_every_epoch",
        action="store_true",
        help="if set, the dataloaders will be reloaded every epoch",
    )
    parser.add_argument(
        "--half_precision", action="store_true", help="if set, train in half precision"
    )
    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="if set, progress bar will be enabled",
    )

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument(
        "--generated_sequence_dropout_probability",
        default=0.0,
        type=float,
        help="How many words to drop from the sequence to be generated. Should only be used during the pre-finetuning phase",
    )
    parser.add_argument(
        "--template_name",
        default="main_templates",
        choices=["main_templates", "noisy_templates", "no_templates"],
        type=str,
        help="Which templates to use. Might be useful for experimenting with multiple, different, types of templates",
    )
    parser.add_argument(
        "--train_dataset_path",
        required=False,
        type=str,
        help="Path to a `jsonl` type of file to serve as train",
    )
    parser.add_argument(
        "--val_dataset_path",
        default=None,
        type=str,
        help="Path to a `jsonl` type of file to serve as val",
    )
    parser.add_argument(
        "--evaluation_dataset_path_dev",
        type=str,
        help="Path to a `jsonl` type of file to do direct_evaluation or do_inference on. There are two args (_dev and _test) because sometimes we do not save the model. As such, we might generate the results on both datasets, then do our selection using dev, and report the test performance using the generated file.",
    )
    parser.add_argument(
        "--evaluation_dataset_path_test",
        type=str,
        help="Path to a `jsonl` type of file to do direct_evaluation or do_inference on. There are two args (_dev and _test) because sometimes we do not save the model. As such, we might generate the results on both datasets, then do our selection using dev, and report the test performance using the generated file.",
    )

    parser.add_argument(
        "--evaluate_on_generation",
        action="store_true",
        help="If this flag is set, we will perform the generation type of evaluation. In other words, we will use the model to generate the answer and score how many tuples were correctly predicted. Otherwise we use the loss",
    )
    parser.add_argument(
        "--learningrate_monitor",
        action="store_true",
        help="If this flag is set, we will monitor the learning rate (some optimizers work without a lr, for example Adafactor)",
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="The maximum number of steps to train for (if both epochs and this is specified, stop on whichever is earliest)",
    )
    parser.add_argument(
        "--save_top",
        default=1,
        type=int,
        help="How many checkpoints to save (default=1)",
    )
    parser.add_argument(
        "--use_weight_distance_factor",
        action="store_true",
        help="If this flag is set, add a new regularization factor",
    )
    parser.add_argument(
        "--weight_distance_factor",
        default=0.01,
        type=float,
        help="The regularization factor",
    )
    parser.add_argument(
        "--optimizer_type",
        default=1,
        choices=[1, 2],
        type=int,
        help="What type of optimizer to use. 1 is for AdaFactor, 2 is for AdamW with linear scheduling",
    )
    parser.add_argument(
        "--gradient_clip_val",
        default=1.0,
        type=float,
        help="The value for the gradient clip. Gradient clipping can be disabled by giving 0.0",
    )
    parser.add_argument(
        "--lower_at_ot",
        action="store_true",
        help="If this flag is set, we will lower the aspect terms and the opinion terms. Note that this applies lowercase the datasets, and the predictions as well",
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help="How often to run the validation loop",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="If this flag is set, we will do early stopping",
    )
    parser.add_argument(
        "--early_stop_patience",
        default=3,
        type=int,
        help="How many validation checks to wait before stopping. Has effect only if early_stopping is set",
    )
    parser.add_argument(
        "--use_best_checkpoint",
        action="store_true",
        help="If this flag is set, we will use the best checkpoint on dev to do the evaluation (NOTE: you need to set save_top to something >= 1)",
    )
    parser.add_argument(
        "--task_mtl_monitor",
        type=str,
        help="What metric to monitor; Has effect only if --evaluate_on_generation is set and --task is set to `mtl`",
    )
    parser.add_argument(
        "--default_domain",
        type=str,
        default="restaurant",
        help="What is the default domain (defaults to `restaurant`). Some templates need a domain",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        default=10,
        type=int,
        help="How many validation batches to do before starting training (to check that everything is working)",
    )
    parser.add_argument(
        "--use_default_task",
        action="store_true",
        help="If this flag is set, we will use the the value passed with `--task` as the task for evaluation. Helpful for allowing the code to work without any instructions as well",
    )

    args = parser.parse_args()

    args = vars(args)
    print(args)
    result = {**args}
    config_data = {}
    if args["config_file"]:
        with open(args["config_file"]) as fin:
            config_data = json.load(fin)
            # result = {**config_data}

    if args["sagemaker_config_file"]:
        if args["config_file"]:
            raise ValueError(
                "The argument `--sagemaker_config_file` is provided, implying that the code is run on sagemaker, with its special path requirements. But `--config_file` is also provided. Maybe a mistake?"
            )

        with open(args["sagemaker_config_file"]) as fin:
            sagemaker_config = json.load(fin)
            # The format will be {'sagemaker_config': str(my_parameters)} where my_parameters can be something like
            # {'config_file': None, 'sagemaker_config_file': None, 'model_name_or_path': 't5-base', 'do_train': True, 'do_direct_eval': True, 'max_seq_length': 128, 'n_gpu': [0, 1, 2, 3], 'weight_decay': 0.0001, 'adam_epsilon': 1e-08}
            # Hacky solution to handle
            config_data = literal_eval(sagemaker_config["sagemaker_config"])

    # result = {**config_data}
    # This allows us to have defaults and use them, as long as they were not overwritten in the config
    print(config_data)
    for k, v in config_data.items():
        result[k] = v

    return result


def evaluate(
    data_loader,
    model,
    tokenizer,
    n_gpu,
    templates=main_templates,
    verbose=False,
    lower=False,
    default_task=None,
):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f"cuda:{n_gpu[0]}")
    # device = torch.device('cpu')
    model.model.to(device)

    model.model.eval()

    inputs, outputs, targets = [], [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            max_length=128,
        )  # num_beams=8, early_stopping=True)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        batch_inputs = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["source_ids"]
        ]

        outputs.extend(dec)
        targets.extend(target)
        inputs.extend(batch_inputs)

    return compute_scores(
        inputs,
        outputs,
        targets,
        templates=templates,
        verbose=verbose,
        lower=lower,
        default_task=default_task,
    )


def get_dataset(
    dataset_path,
    task,
    template_name,
    dropout,
    to_lower,
    default_domain,
    skip_tasks_for_mtl,
):
    dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]
    # if do_filter:
    # dataset = dataset.filter(lambda x: len([q for q in x["quads"] if q[0] != ""]) > 1 and len([q for q in x["quads"] if q[3] != ""]) > 1)
    return dataset.map(
        lambda x: prepare_line(
            x,
            task,
            template_map[template_name],
            categories2text,
            sentword2opinion,
            dropout,
            to_lower,
            default_domain,
            skip_tasks_for_mtl,
        ),
        batched=True,
        load_from_cache_file=False,  # Very important for the mapping; allows for different words to be dropped, because otherwise the same data will be used
    )


"""
Construct some of the pytorch_lightning callbacks
- ModelCheckpoint (on f1, custom filename, saving the last)
- LearningRateMonitor
- EarlyStopping (sometimes commented because, empirically, training for longer proved to be more beneficial)
"""


def get_callbacks(args):
    output = {}
    # Align the `monitor` and `mode` to how we are doing the evaluation
    # If we evaluate on generation, we care about `f1` and `max` score
    # If we evaluate on `val_loss`, we care about the `val_los` score and `min`
    mc = ModelCheckpoint(
        # prefix   = "ckt",
        monitor="f1" if args["evaluate_on_generation"] else "val_loss",
        mode="max" if args["evaluate_on_generation"] else "min",
        filename=(
            "{epoch}-{val_loss:.3f}-{f1:.3f}"
            if args["evaluate_on_generation"]
            else "{epoch}-{val_loss:.3f}"
        ),
        # filename   = '{epoch}-{global_step}',
        save_last=True,
        save_top_k=args["save_top_k"],
        save_weights_only=True,  # worth considering if further training is not required
    )

    output["model_checkpoint"] = mc
    # output['model_checkpoint'] = SavingPointsCheckpoint()

    if args["learningrate_monitor"]:
        lrm = LearningRateMonitor(logging_interval="step")
        output["learningrate_monitor"] = lrm

    if args["early_stopping"]:
        es = EarlyStopping(
            monitor="f1" if args["evaluate_on_generation"] else "val_loss",
            patience=args["early_stop_patience"],
            mode="max" if args["evaluate_on_generation"] else "min",
        )
        output["early_stopping"] = es

    return output


# Important: Calls random inside
# This function takes the file in the jsonl format (one json per line, each json containins `sentence` and `quads`, corresponding
# to the ABSA quad format) and adds the instruction to the input (`input` key) and the expected output (`output` key)
def prepare_line(
    lines: Dict,
    task: str,
    main_templates: Dict,
    categories2text: Dict,
    sentword2opinion: Dict,
    drop_term_probability=0.0,
    to_lower: bool = False,
    default_domain="restaurant",
    skip_tasks_for_mtl=[],
):
    all_sentences = []
    all_quads = []
    inputs = []
    outputs = []
    if not isinstance(lines["sentence"], list) or not isinstance(lines["quads"], list):
        raise ValueError(
            "This function has to be called with batched=True. The reason why is that this function handles both single task and multi-task preparation, and in the case of multi-task, a single instance can give rise to multiple examples"
        )
    for sentence, quads in zip(lines["sentence"], lines["quads"]):
        # If the task is different than our task continue (unless our task is 'mtl')
        for template_id, templates in main_templates.items():
            if task != "mtl" and task != template_id:
                continue
            # If our template is in the set of tasks to be skipped AND the task is `mtl`, continue
            # This check is necessary to allow training for domains which do not have all the neccessary annotations
            # For example, Laptop domain has problematic annotations for category. As such, for Laptop domain task3 and task5 are not applicable
            if task == "mtl" and template_id in skip_tasks_for_mtl:
                continue
            input_key = random.choice(templates["inputs"])
            output_key = random.choice(templates["outputs"])
            input_instruction = Template(templates[input_key]).substitute(TEXT=sentence)
            output = []
            for annotation in quads:
                # Drop some of them, but ONLY if we have something
                if random.random() < drop_term_probability:
                    continue
                at, ac, sp, ot = annotation
                if to_lower:
                    at = at.lower()
                    ot = ot.lower()
                # FIXME WHAT DO WE DO IF ot is 'none'
                if (
                    at.lower() == "none" or at.lower() == "null"
                ):  # for implicit aspect term
                    at = "it"
                if template_id == "task1":
                    output.append(Template(templates[output_key]).substitute(AT=at))
                elif template_id == "task2":
                    sp_op = sentword2opinion[sp]
                    output.append(
                        Template(templates[output_key]).substitute(AT=at, S=sp_op)
                    )
                elif template_id == "task3":
                    # if '#' not in ac:
                    # ac = absa_quad_text2category[ac]
                    ac = categories2text[ac]
                    sp_op = sentword2opinion[sp]
                    output.append(
                        Template(templates[output_key]).substitute(
                            AT=at, S=sp_op, AC=ac
                        )
                    )
                elif template_id == "task4":
                    sp_op = sentword2opinion[sp]
                    output.append(
                        Template(templates[output_key]).substitute(
                            AT=at, OT=ot, S=sp_op, DOMAIN=default_domain
                        )
                    )
                elif template_id == "task5":
                    # if '#' not in ac:
                    # ac = absa_quad_text2category[ac]
                    ac = categories2text[ac]
                    sp_op = sentword2opinion[sp]
                    output.append(
                        Template(templates[output_key]).substitute(
                            AT=at, S=sp_op, AC=ac, OT=ot
                        )
                    )
                elif template_id == "ate":
                    if at != "":  # We skip the sentences without any aspect terms
                        output.append(Template(templates[output_key]).substitute(AT=at))
                elif template_id == "ote":
                    if ot != "":  # We skip the sentences without amy opinion terms
                        output.append(Template(templates[output_key]).substitute(OT=ot))
                elif template_id == "ate_ote":
                    output.append(
                        Template(templates[output_key]).substitute(AT=at, OT=ot)
                    )
                elif template_id == "ate_sent":
                    sp_op = sentword2opinion[sp]
                    output.append(
                        Template(templates[output_key]).substitute(AT=at, S=sp_op)
                    )
                elif template_id == "ate_ote_sent":
                    sp_op = sentword2opinion[sp]
                    output.append(
                        Template(templates[output_key]).substitute(
                            AT=at, OT=ot, S=sp_op
                        )
                    )
                else:
                    raise ValueError(
                        f"{template_id} is not supported here even though it is in the list of templates. Was this intentional?"
                    )
            # We only add if there is something in output. It might be that everything has been dropped, case in which we do not append anything
            if len(output) != 0:
                output = " [SSEP] ".join(output)
                inputs.append(input_instruction)
                outputs.append(output)
                all_sentences.append(sentence)
                all_quads.append(quads)

    return {
        "sentence": all_sentences,
        "quads": all_quads,
        "inputs": inputs,
        "outputs": outputs,
    }


# python main.py --base_path . --task task1 --dataset rest16 --model_name_or_path t5-base --n_gpu 0 --do_train --do_direct_eval --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-4 --num_train_epochs 20 --seed 123 --log_save_name yelp_expanded/task1
def main():
    # Prints to make "debugging" on sagemaker easier
    print("----------")
    for f in glob.glob("/opt/ml/input/**", recursive=True):
        print(f)
    print("----------")

    # initialization
    args = init_args()
    # Some checks to prevent combinations that are not really supposed to work (fail fast)
    if args["use_best_checkpoint"] == True and args["save_top"] < 1:
        raise ValueError(
            "Cannot use the best checkpoint if we do not save the best checkpoint"
        )
    if args["template_name"] == "no_templates" and args["use_default_task"] == False:
        raise ValueError(
            "You need to explicitly set to use the default task when you train without any instructions"
        )
    if args["template_name"] == "no_templates" and args["task"] == "mtl":
        raise ValueError(
            "MTL is not supported without any instructions as the model cannot know what to generate"
        )
    if args["template_name"] != "no_templates" and args["use_default_task"] == True:
        raise ValueError(
            "Using default task instead of automatically inferring it is discouraged"
        )
    if (
        args["task"] == "mtl"
        and args["early_stopping"] == True
        and args["task_mtl_monitor"] is None
    ):
        raise ValueError(
            "We need to specify what task from the `mtl` to monitor for early stop"
        )
    if (
        args["task"] == "mtl"
        and len(
            set(template_map[args["template_name"]]).difference(
                set(args["skip_tasks_for_mtl"])
            )
        )
        == 0
    ):
        raise ValueError("We cannot do `MTL` if we are skipping all the templates")
    if args["task"] != "mtl" and len(args["skip_tasks_for_mtl"]) > 0:
        raise ValueError(
            "There are some tasks set to be skipped, but the training task is not set to `mtl`. Is everything ok?"
        )
    if (
        args["task"] != "mtl"
        and args["task"] not in template_map[args["template_name"]]
    ):
        raise ValueError(
            "The task set for training is not in the template map. Is everything ok?"
        )

    # Some checks to prevent combinations that are not really supposed to work (fail fast)
    pl.seed_everything(args["seed"])
    print(args)
    print(
        "\n",
        "=" * 30,
        f"NEW EXP: {args['task']} on {args['train_dataset_path']} (eval on {args['val_dataset_path']})",
        "=" * 30,
        "\n",
    )

    # sanity check
    # show one sample to check the code and the expected output
    model_name_to_params = {
        "t5-base": (T5ForConditionalGeneration, T5Tokenizer),
        "t5-large": (T5ForConditionalGeneration, T5Tokenizer),
        "facebook/bart-large": (BartForConditionalGeneration, BartTokenizer),
        "facebook/bart-base": (BartForConditionalGeneration, BartTokenizer),
    }
    if args["model_name_or_path"] in model_name_to_params:
        (model_constructor, tokenizer_constructor) = model_name_to_params[
            args["model_name_or_path"]
        ]
    elif os.path.exists(args["model_name_or_path"]):
        (model_constructor, tokenizer_constructor) = (
            T5ForConditionalGeneration,
            T5Tokenizer,
        )
    else:
        model_name = args["model_name_or_path"]
        raise ValueError(
            f"The model name that was supplied ({model_name}) is neither inside the supported models ({list(model_name_to_params.keys())}), nor it is a valid path in the system. Note: For a path in the system, we only support T5"
        )

    tokenizer = tokenizer_constructor.from_pretrained(args["model_name_or_path"])
    print(f"Here is an example (from the dev set):")

    # Maybe print some examples from val
    # Printing is performed only with `verbose_output`, but the sampling is preserved, nevertheless
    # The reasoning is to always keep the same random state
    # Otherwise, results could be with a slight inconsistency when `verbose_output` is set and when it is not
    if args["val_dataset_path"]:
        val_dataset = get_dataset(
            args["val_dataset_path"],
            args["task"],
            args["template_name"],
            dropout=0.0,
            to_lower=args["lower_at_ot"],
            default_domain=args["default_domain"],
            skip_tasks_for_mtl=args["skip_tasks_for_mtl"],
        )  # No dropout for validation

        for i in range(10):
            sample_id = random.randint(0, len(val_dataset) - 1)
            data_sample = T5FineTuner2.custom_collate_fn(
                tokenizer, [val_dataset[sample_id]], max_length=args["max_seq_length"]
            )  # a random data sample

            i = tokenizer.decode(data_sample["source_ids"][0], skip_special_tokens=True)
            o = tokenizer.decode(data_sample["target_ids"][0], skip_special_tokens=True)
            if args["verbose_output"]:
                print("-" * 30)
                print("Input :", i)
                print("Output:", o)

    # training process
    if args["do_train"]:
        print("\n****** Conduct Training ******")
        train_dataset = get_dataset(
            args["train_dataset_path"],
            args["task"],
            args["template_name"],
            dropout=args["generated_sequence_dropout_probability"],
            to_lower=args["lower_at_ot"],
            default_domain=args["default_domain"],
            skip_tasks_for_mtl=args["skip_tasks_for_mtl"],
        )
        args["num_training_steps"] = (
            len(train_dataset)
            // (args["train_batch_size"] * args["gradient_accumulation_steps"])
        ) * args["num_train_epochs"]

        # Initialize the T5 model
        # If a checkpoint path is supplied, load from it, but override the model's hyperparameters
        # with the ones supplied as command-line arguments
        # Otherwise, simply create a model according to the command-line arguments
        if args["checkpoint_path"]:
            # We actually want to use the hyperparameters passed as CL arguments (but be careful if there are any dimensions passed)
            model = T5FineTuner2.load_from_checkpoint(
                args["checkpoint_path"], hparams=args
            )
            # Have to manually set the arguments in case of loading from checkpoint
            # model.hyperparameters = args
            # model.save_hyperparameters(model.hyperparameters)
        else:
            model = T5FineTuner2(args)

        callbacks = get_callbacks(
            {
                "evaluate_on_generation": args["evaluate_on_generation"],
                "early_stopping": args["early_stopping"],
                "learningrate_monitor": args["learningrate_monitor"],
                "num_train_epochs": args["num_train_epochs"],
                "save_top_k": args["save_top"],
                "early_stop_patience": args["early_stop_patience"],
            }
        )

        logger = TensorBoardLogger(args["log_save_dir"], name=args["log_save_name"])
        from pytorch_lightning.callbacks import TQDMProgressBar

        # prepare for trainer
        train_params = dict(
            accumulate_grad_batches=args["gradient_accumulation_steps"],
            gpus=args["n_gpu"],
            gradient_clip_val=args["gradient_clip_val"],  # 1.0,
            # gradient_clip_algorithm           = 'norm',
            max_epochs=args["num_train_epochs"],
            max_steps=args.get("max_steps", None),
            callbacks=[*list(callbacks.values())],
            reload_dataloaders_every_n_epochs=(
                1 if args["reload_dataloaders_every_epoch"] else 0
            ),
            logger=logger,
            num_sanity_val_steps=args["num_sanity_val_steps"],
            val_check_interval=args["val_check_interval"],
            precision=16 if args["half_precision"] else 32,
            enable_progress_bar=args["enable_progress_bar"],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # Save the best as best.ckpt in the directory used for checkpoints
        model_checkpoint = callbacks["model_checkpoint"]
        # copyfile(model_checkpoint.best_model_path, f'{model_checkpoint.dirpath}/best.ckpt')

        print("Finish training and saving the model!")

        # evaluation
        if args["do_direct_eval"]:
            if args["evaluation_dataset_path_dev"]:
                pl.seed_everything(args["seed"])
                if args["use_best_checkpoint"]:
                    print(
                        f"\n****** Conduct Evaluating using the best model {trainer.checkpoint_callback.best_model_path} ******"
                    )
                    model = T5FineTuner2.load_from_checkpoint(
                        trainer.checkpoint_callback.best_model_path
                    )
                else:
                    print(
                        f"\n****** Conduct Evaluating using the last model {trainer.checkpoint_callback.last_model_path} ******"
                    )
                    model = T5FineTuner2.load_from_checkpoint(
                        trainer.checkpoint_callback.last_model_path
                    )
                print(
                    f"Model signature: {torch.cat([x.view(-1) for x in model.parameters()]).sum()}"
                )

                # sents, _ = read_absa_quad_from_file(f'data/{args['dataset']}/test.txt')

                print()
                test_dataset = get_dataset(
                    args["evaluation_dataset_path_dev"],
                    task=args["task"],
                    template_name=args["template_name"],
                    dropout=0.0,
                    to_lower=args["lower_at_ot"],
                    default_domain=args["default_domain"],
                    skip_tasks_for_mtl=args["skip_tasks_for_mtl"],
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=32,
                    num_workers=4,
                    prefetch_factor=args.get("prefetch_factor", 2),
                    collate_fn=lambda x: T5FineTuner2.custom_collate_fn(
                        tokenizer, x, max_length=args["max_seq_length"]
                    ),
                )
                # print(test_loader.device)

                # compute the performance scores
                with torch.no_grad():
                    all_scores, all_inputs, all_labels, all_preds = evaluate(
                        test_loader,
                        model,
                        tokenizer,
                        args["n_gpu"],
                        templates=template_map[args["template_name"]],
                        verbose=True,
                        lower=args["lower_at_ot"],
                        default_task=args["task"] if args["use_default_task"] else None,
                    )

                with open(
                    os.path.join(
                        callbacks["model_checkpoint"].dirpath, "results_dev.pickle"
                    ),
                    "wb",
                ) as fhw:
                    pickle.dump(
                        [all_scores, all_inputs, all_labels, all_preds, args], fhw
                    )

            if args["evaluation_dataset_path_test"]:
                # Do again, but for test
                pl.seed_everything(args["seed"])
                if args["use_best_checkpoint"]:
                    print(
                        f"\n****** Conduct Evaluating using the best model {trainer.checkpoint_callback.best_model_path} ******"
                    )
                    model = T5FineTuner2.load_from_checkpoint(
                        trainer.checkpoint_callback.best_model_path
                    )
                else:
                    print(
                        f"\n****** Conduct Evaluating using the last model {trainer.checkpoint_callback.last_model_path} ******"
                    )
                    model = T5FineTuner2.load_from_checkpoint(
                        trainer.checkpoint_callback.last_model_path
                    )

                # sents, _ = read_absa_quad_from_file(f'data/{args['dataset']}/test.txt')

                print()
                test_dataset = get_dataset(
                    args["evaluation_dataset_path_test"],
                    task=args["task"],
                    template_name=args["template_name"],
                    dropout=0.0,
                    to_lower=args["lower_at_ot"],
                    default_domain=args["default_domain"],
                    skip_tasks_for_mtl=args["skip_tasks_for_mtl"],
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=32,
                    num_workers=4,
                    prefetch_factor=args.get("prefetch_factor", 2),
                    collate_fn=lambda x: T5FineTuner2.custom_collate_fn(
                        tokenizer, x, max_length=args["max_seq_length"]
                    ),
                )
                # print(test_loader.device)

                # compute the performance scores
                with torch.no_grad():
                    all_scores, all_inputs, all_labels, all_preds = evaluate(
                        test_loader,
                        model,
                        tokenizer,
                        args["n_gpu"],
                        templates=template_map[args["template_name"]],
                        verbose=True,
                        lower=args["lower_at_ot"],
                        default_task=args["task"] if args["use_default_task"] else None,
                    )

                with open(
                    os.path.join(
                        callbacks["model_checkpoint"].dirpath, "results_test.pickle"
                    ),
                    "wb",
                ) as fhw:
                    pickle.dump(
                        [all_scores, all_inputs, all_labels, all_preds, args], fhw
                    )

        # Delete the checkpoints if the flag is set
        if args["not_save_checkpoint"]:
            os.remove(model_checkpoint.last_model_path)
            for model_path in model_checkpoint.best_k_models.keys():
                os.remove(model_path)

            # os.remove(f'{model_checkpoint.dirpath}/best.ckpt')

    if args["do_inference"]:
        pl.seed_everything(args["seed"])
        print("\n****** Conduct inference on trained checkpoint ******")

        # initialize the T5 model from previous checkpoint
        print(f"Load trained model from {args['checkpoint_path']}")
        if (
            args["checkpoint_path"] is None
            or args["checkpoint_path"] == ""
            or not os.path.exists(args["checkpoint_path"])
        ):
            raise ValueError(
                "The config is set to `do_inference`, but no valid model path was supplied. Is everything ok?"
            )
        model = T5FineTuner2.load_from_checkpoint(args["checkpoint_path"])
        model.eval()

        print()
        if args["evaluation_dataset_path_dev"]:
            pl.seed_everything(args["seed"])
            test_dataset = get_dataset(
                args["evaluation_dataset_path_dev"],
                task=args["task"],
                template_name=args["template_name"],
                dropout=0.0,
                to_lower=args["lower_at_ot"],
                default_domain=args["default_domain"],
                skip_tasks_for_mtl=args["skip_tasks_for_mtl"],
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                num_workers=8,
                prefetch_factor=args.get("prefetch_factor", 2),
                collate_fn=lambda x: T5FineTuner2.custom_collate_fn(
                    tokenizer, x, max_length=args["max_seq_length"]
                ),
            )
            # print(test_loader.device)

            # compute the performance scores
            with torch.no_grad():
                all_scores, all_inputs, all_labels, all_preds = evaluate(
                    test_loader,
                    model,
                    tokenizer,
                    args["n_gpu"],
                    templates=template_map[args["template_name"]],
                    verbose=args["verbose_output"],
                    lower=args["lower_at_ot"],
                    default_task=args["task"] if args["use_default_task"] else None,
                )
            # print(all_scores)
            # for x, y, z in zip(all_inputs['ate_sent'], all_labels['ate_sent'], all_preds['ate_sent']):
            #     print("-----")
            #     print(x)
            #     print(y)
            #     print(z)
            #     print("-----")
            # # Dump the results to the directory from where the checkpoint was loaded from
            # sd = args['seed']
            # ckpt_sd = args['checkpoint_path'].split('/')[-1].split('.')[0][4:]
            # dtst = args['evaluation_dataset_path_dev'].split('/')[-1].split('.')[0]
            # task = args['task']
            # with open(f"logs/zero_shot_exp/{dtst}_{task}_{sd}_{ckpt_sd}.pickle", 'wb') as fhw:
            #     pickle.dump([all_scores, all_inputs, all_labels, all_preds, args], fhw)

        if args["evaluation_dataset_path_test"]:
            pl.seed_everything(args["seed"])
            test_dataset = get_dataset(
                args["evaluation_dataset_path_test"],
                task=args["task"],
                template_name=args["template_name"],
                dropout=0.0,
                to_lower=args["lower_at_ot"],
                default_domain=args["default_domain"],
                skip_tasks_for_mtl=args["skip_tasks_for_mtl"],
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                num_workers=8,
                prefetch_factor=args.get("prefetch_factor", 2),
                collate_fn=lambda x: T5FineTuner2.custom_collate_fn(
                    tokenizer, x, max_length=args["max_seq_length"]
                ),
            )
            # print(test_loader.device)

            # compute the performance scores
            with torch.no_grad():
                all_scores, all_inputs, all_labels, all_preds = evaluate(
                    test_loader,
                    model,
                    tokenizer,
                    args["n_gpu"],
                    templates=template_map[args["template_name"]],
                    verbose=args["verbose_output"],
                    lower=args["lower_at_ot"],
                    default_task=args["task"] if args["use_default_task"] else None,
                )

            # # Dump the results to the directory from where the checkpoint was loaded from
            # with open(os.path.join(Path(args['checkpoint_path']).parent, "results_directeval_test.pickle"), 'wb') as fhw:
            #     pickle.dump([all_scores, all_inputs, all_labels, all_preds, args], fhw)

        # # write to file
        # log_file_path = f"results_log/{args['dataset']}.txt"
        # local_time = time.asctime(time.localtime(time.time()))

        # exp_settings = f"Datset={args['dataset']}; Train bs={args['train_batch_size']}, num_epochs = {args['num_train_epochs']}"
        # exp_results = f"F1 = {scores['f1']:.4f}"

        # log_str = f'============================================================\n'
        # log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

        # if not os.path.exists('./results_log'):
        #     os.mkdir('./results_log')

        # with open(log_file_path, "a+") as f:
        #     f.write(log_str)


if __name__ == "__main__":
    main()
