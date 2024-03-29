# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import collections
import copy
import json
import logging
import math
import os
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from models import BERT_MTL, BERT_MTL_ATTN, BERT_SCL, BERT_STL, BERT_MTL_MAX


logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--emotion_model",
        type=str,
        help="Path to pretrained model on emotion representatons.",
        # required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["multi_feature", "base"],
        required=True,
        help="the name of the model to use. some models may use different model args than others.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in [
            "csv",
            "json",
        ], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in [
            "csv",
            "json",
        ], "`validation_file` should be a csv or a json file."
    if args.test_file is not None:
        extension = args.test_file.split(".")[-1]
        assert extension in [
            "csv",
            "json",
        ], "`test_file` should be a csv or a json file."

    if args.model_type == "multi_feature" and args.emotion_model is None:
        raise ValueError("Need an emotion pre-trained model")

    return args


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir)
        if args.with_tracking
        else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # For seed results
    results = {}

    # list of seeds
    seeds = [42, 52, 62, 72, 100]
    for seed in seeds:
        # set training seed
        set_seed(seed)

        accelerator.wait_for_everyone()

        # Loading the dataset from local csv  file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = (
            args.train_file if args.train_file is not None else args.validation_file
        ).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
        label_to_id = {v: i for i, v in enumerate(label_list)}

        # Load tokenizer and model
        config = AutoConfig.from_pretrained(args.bert_model, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(
            args.bert_model, use_fast=not args.use_slow_tokenizer
        )
        if args.model_type == "base":
            model = BERT_STL(
                args.bert_model,
                num_labels,
            )
        elif args.model_type == "multi_feature":
            model = BERT_MTL(
                args.bert_model,
                args.emotion_model,
                num_labels,
            )

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if label_list:
            model.enc_model.config.label2id = {l: i for i, l in enumerate(label_list)}
            model.enc_model.config.id2label = {
                id: label for label, id in config.label2id.items()
            }

        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (examples["text"],)
            result = tokenizer(
                *texts, padding=padding, max_length=args.max_length, truncation=True
            )

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        with accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        test_dataset = processed_datasets["test"]

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorWithPadding(
                tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
            )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Prepare everything with our `accelerator`.
        (
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lr_scheduler,
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

        # Figure out how many steps we should save the Accelerator states
        if hasattr(args.checkpointing_steps, "isdigit"):
            checkpointing_steps = args.checkpointing_steps
            if args.checkpointing_steps.isdigit():
                checkpointing_steps = int(args.checkpointing_steps)
        else:
            checkpointing_steps = None

        # Get the metric function
        metric = evaluate.load("f1")

        # Train!
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Seed = {seed}")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if (
                args.resume_from_checkpoint is not None
                or args.resume_from_checkpoint != ""
            ):
                accelerator.print(
                    f"Resumed from checkpoint: {args.resume_from_checkpoint}"
                )
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        best_model = None
        best_eval_loss = float("inf")
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                outputs = model(**batch)
                loss = outputs[0]
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            samples_seen = 0
            eval_loss = 0.0
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs[1].argmax(dim=-1)
                predictions, references = accelerator.gather(
                    (predictions, batch["labels"])
                )
                eval_loss += outputs[0].item()
            eval_loss = eval_loss / len(eval_dataloader)
            logger.info(f"epoch {epoch}: eval loss: {eval_loss}")

            if eval_loss < best_eval_loss:
                logger.info("Saving best model")
                best_eval_loss = eval_loss
                best_model = copy.deepcopy(model)

        # evaluate best model on test data
        best_model.eval()
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs[1].argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(test_dataloader) - 1:
                    predictions = predictions[
                        : len(test_dataloader.dataset) - samples_seen
                    ]
                    references = references[
                        : len(test_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        # logger.info(f"epoch {epoch}: eval loss: {eval_loss}")
        test_metric = metric.compute(average="macro")

        # Print metric
        logger.info("--------------------------------")
        logger.info(f"Eval metric for seed {seed}: {test_metric}")
        logger.info("--------------------------------")

        results[seed] = test_metric

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"eval_f1": test_metric["f1"]}, f)

    # Print fold results
    print(f"RESULTS FOR {len(seeds)} SEEDS")
    print("--------------------------------")
    counter = collections.Counter()
    for key, result in results.items():
        print(f"Seed {key}: {result} %")
        counter.update(result)
    sum = dict(counter)
    print(f"Average: {({key: value / len(results) for key, value in sum.items()})} %")


if __name__ == "__main__":
    main()
