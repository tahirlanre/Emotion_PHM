import argparse
from collections import defaultdict
import json
import logging
from datetime import datetime
import math
import os
from pickletools import optimize
import random
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import datasets
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    SchedulerType,
)
import transformers.optimization as tfoptim
import ax

from models.model import BertClassificationModel, BiLSTMAttn, BiLSTM, MLP, EmoBERTMLP
from utils.utils import init_logger, set_seed, compute_metrics
from const import SEEDS
from data import PHMDataset

logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformer model on text classification task"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory contatining the training, validation and test data.",
    )
    # parser.add_argument(
    #     "--train_file",
    #     type=str,
    #     default=None,
    #     help="A csv or a csv file contatining the training data.",
    # )
    # parser.add_argument(
    #     "--validation_file",
    #     type=str,
    #     default=None,
    #     help="A csv or a csv file containing the validation data.",
    # )
    # parser.add_argument(
    #     "--test_file",
    #     type=str,
    #     default=None,
    #     help="A csv or a csv file containing the test data.",
    # )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--emotion_model_name_or_path",
        type=str,
        help="Path to pretrained model on emotion representatons.",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "bilstm_attn", "bilstm", "mlp", "emobert"],
        required=True,
        help="the name of the model to use. some models may use different model args than others.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    # parser.add_argument(
    #     "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    # )
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
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
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
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="size of the model's hidden layer. ignored for BERT.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="number of lstm layers to use."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout to apply to the model during training",
    )
    parser.add_argument(
        "--num_restarts",
        type=int,
        default=1,
        help="the number of random restarts to average. we have 10 random seeds predefined in const.py; more "
        "restarts than this will cause an error unless you add more seeds.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="number of hyperparameter trials to attempt. ignored if not optimizing.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="include this flag to use the ax library to tune parameters.",
    )
    args = parser.parse_args()

    return args


def read_data(data_dir):
    df_train = pd.read_csv(Path(data_dir) / ("train.csv"))
    df_dev = pd.read_csv(Path(data_dir) / ("dev.csv"))
    df_test = pd.read_csv(Path(data_dir) / ("test.csv"))

    zipped_train = list(zip(df_train["text"].tolist(), df_train["label"].tolist()))
    zipped_dev = list(zip(df_dev["text"].tolist(), df_dev["label"].tolist()))
    zipped_test = list(zip(df_test["text"].tolist(), df_test["label"].tolist()))

    random.shuffle(zipped_train)
    random.shuffle(zipped_dev)
    # random.shuffle(zipped_test)

    unzipped_train = list(zip(*zipped_train))
    unzipped_dev = list(zip(*zipped_dev))
    unzipped_test = list(zip(*zipped_test))

    return (
        list(unzipped_train[0]),
        list(unzipped_train[1]),
        list(unzipped_dev[0]),
        list(unzipped_dev[1]),
        list(unzipped_test[0]),
        list(unzipped_test[1]),
    )


def setup_model(args, num_labels):
    if args.model == "bert":
        model = BertClassificationModel(
            args.model_name_or_path, args.emotion_model_name_or_path, num_labels
        )
    elif args.model == "bilstm":
        model = BiLSTM(
            args.model_name_or_path,
            args.emotion_model_name_or_path,
            args.hidden_dim,
            args.num_layers,
            num_labels,
            args.dropout,
        )
    elif args.model == "bilstm_attn":
        model = BiLSTMAttn(
            args.model_name_or_path,
            args.emotion_model_name_or_path,
            args.hidden_dim,
            args.num_layers,
            num_labels,
            args.dropout,
        )
    elif args.model == "mlp":
        model = MLP(
            args.model_name_or_path,
            args.emotion_model_name_or_path,
            num_labels,
            args.dropout,
        )
    elif args.model == "emobert":
        model = EmoBERTMLP(args.emotion_model_name_or_path, num_labels, args.dropout)

    return model


def train(model, train_dataloader, eval_dataloader, args):
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # create optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # create scheduler for optimizer
    lr_scheduler = tfoptim.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int((len(train_dataloader) * args.num_train_epochs) / 10),
        num_training_steps=len(train_dataloader) * args.num_train_epochs,
    )

    progress_bar = tqdm(range(max_train_steps))
    global_step = 0
    best_val_loss = float("inf")
    best_model = None
    best_dev_results = {}

    train_loss = 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {key: batch[key].to(device) for key in batch}
            outputs = model(**batch)

            loss = outputs[0]
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

        eval_loss, eval_results = evaluate(model, eval_dataloader)

        logger.info(f"| epoch {epoch:3d} | eval loss {eval_loss:5.2f}")
        wandb.log({"eval_loss": eval_loss})

        if eval_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_val_loss = eval_loss
            best_dev_results = eval_results

    return best_model, best_dev_results


def train_main(args, seed=42):
    set_seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = read_data(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    val_encodings = tokenizer(X_val, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    train_dataset = PHMDataset(train_encodings, y_train)
    validation_dataset = PHMDataset(val_encodings, y_val)
    test_dataset = PHMDataset(test_encodings, y_test)

    label_list = list(set(y_train))
    num_labels = len(label_list)

    model = setup_model(args, num_labels)
    model.to(device)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
    )

    logger.info("=" * 40)
    logger.info(f"  Random seed = {seed}")
    logger.info(args)
    logger.info("=" * 40)

    best_model, best_dev_results = train(model, train_dataloader, val_dataloader, args)

    _, test_results = evaluate(best_model, test_dataloader)

    logger.info(f"***** Eval results on test dataset*****")
    for key in sorted(test_results.keys()):
        logger.info("  {} = {}".format(key, str(test_results[key])))

    results = {"best_dev_results": best_dev_results, "test_results": test_results}

    return results


def evaluate(model, dataloader):
    model.eval()
    eval_loss = 0.0
    y_pred = None
    y_true = None

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = {key: batch[key].to(device) for key in batch}
            outputs = model(**batch)

            eval_loss += outputs[0].item()
            if y_pred is None:
                y_pred = outputs[1].argmax(dim=-1).detach().cpu().numpy()
                y_true = batch["labels"].detach().cpu().numpy()
            else:
                y_pred = np.append(
                    y_pred, outputs[1].argmax(dim=-1).detach().cpu().numpy(), axis=0
                )
                y_true = np.append(y_true, batch["labels"].detach().cpu().numpy())

    results = compute_metrics(y_true, y_pred)
    eval_loss = eval_loss / len(dataloader)
    return eval_loss, results


def tune(parameters, args):
    def get_score_for_parameters(parameters, args):
        for param in parameters:
            setattr(args, param, parameters[param])

        avg_dev = defaultdict(float)
        avg_eval = defaultdict(float)

        # for i in range(args.num_restarts):
        results = train_main(args)
        for collector, metrics in zip(
            [avg_dev, avg_eval],
            [results["best_dev_results"], results["test_results"]],
        ):
            for metric in metrics:
                collector[metric] += metrics[metric] / args.num_restarts

        return {
            "dev f1": (avg_dev["macro_f1"], 0),
            "test f1": (avg_eval["macro_f1"], 0),
            "dev accuracy": (avg_dev["accuracy"], 0),
            "test accuracy": (avg_eval["accuracy"], 0),
        }

    best_parameters, values, experiment, best_model = ax.service.managed_loop.optimize(
        parameters=parameters,
        evaluation_function=lambda params: get_score_for_parameters(params, args),
        objective_name="dev f1",
        total_trials=args.trials,
    )

    df = experiment.fetch_data().df

    # record final info about best parameter settings
    logging.info("=" * 30)
    logging.info("Finished optimizing!")
    logging.info("BEST PARAMETER SETTINGS...")
    for param in best_parameters:
        logging.info(
            "{param_name}: {param_value}".format(
                param_name=param, param_value=best_parameters[param]
            )
        )
    logging.info("BEST DEV SCORE:")
    logging.info(max(df[df["metric_name"] == "dev f1"]["mean"]))
    logging.info("=" * 30)

    print(best_parameters)

    # save best parameters
    dataset = args.data_dir.rstrip("/").split("/")[-1]
    with open(f"./{args.model}_{dataset}_params.json", "w") as f:
        f.write(json.dumps(best_parameters))

    return best_parameters


def main():
    args = parse_args()
    init_logger()

    wandb.login()
    wandb.init(project="phm-classification")

    run_name = (
        wandb.run.name
        if wandb.run.name
        else datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    )
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, run_name)
    else:
        args.output_dir = os.path.join("runs", run_name)
        os.makedirs(args.output_dir, exist_ok=True)

    if args.tune:
        if args.model == "bert" or args.model == "mlp" or args.model == "emobert":
            parameters = [
                {
                    "name": "learning_rate",
                    "type": "range",
                    "bounds": [1e-6, 1e-3],
                    "log_scale": True,
                },
                {
                    "name": "dropout",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",
                },
                {"name": "batch_size", "type": "choice", "values": [32, 64]},
            ]
        elif args.model == "bilstm" or args.model == "bilstm-attn":
            parameters = [
                {
                    "name": "dropout",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",
                },
                {"name": "batch_size", "type": "choice", "values": [32, 64, 128]},
                {"name": "hidden_dim", "type": "choice", "values": [128, 256, 512]},
                {
                    "name": "learning_rate",
                    "type": "choice",
                    "values": [0.01, 0.001, 0.0001],
                },
            ]
        else:
            raise ValueError("Don't know how to tune that yet.")

        best_parameters = tune(parameters, args)
    else:
        logging.info("***** Loading best parameters *****")
        dataset = args.data_dir.rstrip("/").split("/")[-1]
        with open(f"./{args.model}_{dataset}_params.json", "r") as f:
            best_parameters = json.load(f)

    for param_name in best_parameters:
        if param_name == "dropout":
            args.dropout = best_parameters[param_name]
        elif param_name == "batch_size":
            args.batch_size = best_parameters[param_name]
        if param_name == "hidden_dim":
            args.hidden_dim = best_parameters[param_name]
        elif param_name == "learning_rate":
            args.learning_rate = best_parameters[param_name]

    seed_results = []
    for i in range(args.num_restarts):
        logging.info(f"***** Running Training #{i+1} of {args.num_restarts} *****")
        test_results = train_main(args, seed=SEEDS[i])
        seed_results.append(test_results["test_results"])

    logger.info(f"***** Average eval results on test dataset *****")
    for key in sorted(seed_results[0].keys()):
        avg_value = np.mean([result[key] for result in seed_results])
        logger.info("  {} = {}".format(key, str(avg_value)))


if __name__ == "__main__":
    main()
