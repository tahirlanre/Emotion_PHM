import argparse
import logging
from datetime import datetime
import math
import os
import random
from pathlib import Path
import shutil
import copy

import datasets
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

import numpy as np

import torch
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    SchedulerType,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)

from utils.utils import init_logger, set_seed, compute_metrics

import wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformer model on text classification task"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a csv file contatining the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a csv file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a csv file containing the test data.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=50,
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
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
        default=1e-5,
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
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Number of updates steps before two checkpoint saves",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Number of updates steps before logging metrics",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_logger()

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    wandb.login()
    wandb.init(project="phm-classification")

    run_name = (
        wandb.run.name
        if wandb.run.name
        else datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    )
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, run_name)
    else:
        output_dir = os.path.join("runs", run_name)
        os.makedirs(output_dir, exist_ok=True)

    random_seeds = [
        69556,
        79719,
        30010,
        46921,
        25577,
        52538,
        56440,
        41228,
        66558,
        48642,
    ]
    best_result, best_val = {}, []

    for seed in random_seeds:
        set_seed(seed)

        raw_dataset = load_dataset("csv", data_files=args.train_file, split="train")
        train_devtest = raw_dataset.train_test_split(
            shuffle=True, seed=42, test_size=0.1
        )
        dev_test = train_devtest["test"].train_test_split(
            shuffle=True, seed=42, test_size=0.5
        )

        dataset = DatasetDict(
            {
                "train": train_devtest["train"],
                "validation": dev_test["train"],
                "test": dev_test["test"],
            }
        )

        label_list = dataset["train"].unique("label")
        label_list.sort()
        num_labels = len(label_list)

        config = AutoConfig.from_pretrained(
            args.model_name_or_path, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        special_tokens_dict = {"additional_special_tokens": ["<url>", "<user>"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        label_to_id = {v: i for i, v in enumerate(label_list)}
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
        model.to(device)

        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            texts = (examples["text"],)
            result = tokenizer(
                *texts, padding=padding, max_length=args.max_seq_length, truncation=True
            )

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        test_dataset = processed_datasets["test"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorWithPadding(tokenizer)

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        total_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )

        logger.info(f"***** Running training with seed = {seed} *****")
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
        logger.info(f"  Total optimization steps = {max_train_steps}")
        progress_bar = tqdm(range(max_train_steps))

        global_step = 0
        best_val_loss = float("inf")
        best_model = None
        train_loss = 0.0
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch.to(device)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                train_loss += loss.item()
                loss.backward()
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    global_step += 1

                # log interval loss
                if global_step % args.log_interval == 0:
                    cur_loss = train_loss / args.log_interval
                    wandb.log({"train_loss": cur_loss})
                    logger.info(f"| epoch {epoch:3d} | loss {cur_loss:5.2f}")
                    train_loss = 0.0

                # if global_step >= args.max_train_steps:
                #     break

            model.eval()
            eval_loss = 0.0
            y_pred = None
            y_true = None

            for step, batch in enumerate(eval_dataloader):
                batch.to(device)
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
                if y_pred is None:
                    y_pred = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
                    y_true = batch["labels"].detach().cpu().numpy()
                else:
                    y_pred = np.append(
                        y_pred,
                        outputs.logits.argmax(dim=-1).detach().cpu().numpy(),
                        axis=0,
                    )
                    y_true = np.append(y_true, batch["labels"].detach().cpu().numpy())

            eval_loss = eval_loss / len(eval_dataloader)
            results = {
                "loss": eval_loss,
            }
            logger.info(f"| epoch {epoch:3d} | eval loss {eval_loss:5.2f}")
            wandb.log({"eval_loss": eval_loss})

            result = compute_metrics(y_true, y_pred)
            results.update(result)

            if eval_loss < best_val_loss:
                best_model = copy.deepcopy(model)
                best_val_loss = eval_loss

        y_pred = None
        y_true = None
        best_model.eval()
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                batch.to(device)
                outputs = best_model(**batch)
                if y_pred is None:
                    y_pred = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
                    y_true = batch["labels"].detach().cpu().numpy()
                else:
                    y_pred = np.append(
                        y_pred,
                        outputs.logits.argmax(dim=-1).detach().cpu().numpy(),
                        axis=0,
                    )
                    y_true = np.append(y_true, batch["labels"].detach().cpu().numpy())

        results = compute_metrics(y_true, y_pred)

        output_test_file = os.path.join(output_dir, f"test_results_{seed}.txt")
        output_prediction_file = os.path.join(
            output_dir, f"test_predictions_{seed}.txt"
        )

        with open(output_test_file, "w") as f_w:
            logger.info("***** Eval results on test dataset *****")
            for key in sorted(results.keys()):
                logger.info("  {} = {}".format(key, str(results[key])))
                f_w.write("  {} = {}\n".format(key, str(results[key])))

        with open(output_prediction_file, "w") as f_w:
            f_w.write("index\tprediction\n")
            for index, item in enumerate(y_pred):
                item = label_list[item]
                f_w.write(f"{index}\t{item}\n")

        best_result[seed] = results

    output_avg_test_file = os.path.join(output_dir, f"test_results.txt")
    logger.info(f"*****  Average eval results on test dataset *****")
    with open(output_avg_test_file, "w") as f_w:
        for key in sorted(results.keys()):
            avg_value = np.mean([d[key] for d in best_result.values()])
            logger.info("  {} = {}".format(key, str(avg_value)))
            f_w.write("  {} = {}\n".format(key, str(avg_value)))


if __name__ == "__main__":
    main()
