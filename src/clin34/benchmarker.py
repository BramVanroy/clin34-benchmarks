import json
import string
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from random import randrange
from typing import Any, Literal

import outlines
import torch
from datasets import Dataset, load_dataset
from jinja2 import Template
from outlines import models
from outlines.models.transformers import Transformers as OutlinesHFModel
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from sklearn.metrics import classification_report, f1_score
from tqdm import trange
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from transformers import logging as hf_logging
from transformers.pipelines.pt_utils import KeyDataset

from clin34.significance import add_confidence
from clin34.utils import is_jinja_template


yaml = YAML()

hf_logging.set_verbosity_error()


@dataclass
class Benchmarker:
    dataset_name: str
    labels2idx: dict[str, int]
    prompt: str
    model_name: str
    output_dir: Path
    dataset_config: str = None
    dataset_split: str = "test"
    label_column: str = "label"
    convert_labels_to_int: bool = False
    text_column: str = None
    f1_average: str = "macro"
    system_message: str = ""
    use_chat_template: bool = True
    bnb_config: dict[str, Any] = None
    model_kwargs: dict[str, Any] = None
    device: str | int | torch.device = "auto"
    formatted_output_colname: str = "formatted"
    trust_remote_code: bool = False
    save_config_as: Literal["json", "yaml"] = "yaml"
    batch_size: int = 1
    max_tokens: int = 10
    num_runs: int = 3
    num_workers: int = 1
    process_id: int = 0
    verbose: bool = True

    dataset: Dataset | KeyDataset = field(default=None, init=False)
    hf_tokenizer: PreTrainedTokenizer = field(default=None, init=False)
    outlines_model: OutlinesHFModel = field(default=None, init=False)
    hf_model: PreTrainedModel = field(default=None, init=False)
    true_label_idxs: list[int] = field(default=None, init=False)
    idx2label: dict[int, str] = field(default=None, init=False)
    _for_saving_bnb_config: dict[str, Any] = field(default=None, init=False)
    _for_saving_model_kwargs: dict[str, Any] = field(default=None, init=False)

    def __post_init__(self):
        if self.prompt and self.text_column:
            raise ValueError(
                "Cannot have both 'prompt' and 'text_column'. If you use a prompt, the fields will automatically be"
                " filled out with the appropriate columns from the dataset. If you do not use a prompt, you need to"
                " pass the 'text_column' that will be used instead."
            )

        if self.system_message and not self.use_chat_template:
            raise ValueError("Cannot have 'system_message' without 'use_chat_template'")

        self.change_model(
            model_name=self.model_name,
            bnb_config=self.bnb_config,
            model_kwargs=self.model_kwargs,
            use_chat_template=self.use_chat_template,
            trust_remote_code=self.trust_remote_code,
            device=self.device,
        )

        self.change_dataset(
            dataset_name=self.dataset_name,
            labels2idx=self.labels2idx,
            prompt=self.prompt,
            dataset_config=self.dataset_config,
            dataset_split=self.dataset_split,
            label_column=self.label_column,
            system_message=self.system_message,
            trust_remote_code=self.trust_remote_code,
        )

        self.output_dir = Path(self.output_dir)

    @property
    def label_idxs(self):
        return list(self.labels2idx.values())

    @property
    def label_names(self):
        return list(self.labels2idx.keys())

    def change_dataset(
        self,
        dataset_name: str,
        labels2idx: dict[str, int],
        prompt: str,
        dataset_config: None | str,
        dataset_split: str,
        label_column: str,
        system_message: str,
        trust_remote_code: bool,
    ):
        self.dataset_name = dataset_name
        self.labels2idx = labels2idx
        self.idx2label = {idx: label for label, idx in labels2idx.items()}
        self.prompt = prompt
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.label_column = label_column
        self.system_message = system_message

        self.dataset = load_dataset(
            self.dataset_name, self.dataset_config, split=self.dataset_split, trust_remote_code=trust_remote_code
        )

        if self.convert_labels_to_int:
            self.dataset = self.dataset.map(
                lambda item: {label_column: labels2idx[item[label_column]]}, num_proc=self.num_workers
            )

        if self.verbose:
            print(f"DATASET SIZE: {len(self.dataset):,}")

        self.true_label_idxs = self.dataset[self.label_column]

        counts = Counter(self.true_label_idxs)
        if self.verbose:
            for label, label_idx in labels2idx.items():
                print(f"Dataset no. occurrences for {label}: {counts[label_idx]:,}")

        if any(c == 0 for c in counts.values()):
            raise ValueError(
                "Some labels have no occurrences in the dataset. This is possible if your `labels2idx`"
                " contains the wrong possible labels, or if the given `label_column` contains strings rather than"
                " integer labels. If the latter is the case, enable `convert_labels_to_int=true`."
            )

        self._format_dataset()

    def _format_dataset(self):
        ds_columns = self.dataset.column_names
        if self.formatted_output_colname in ds_columns:
            raise ValueError(
                f"Column name '{self.formatted_output_colname}' already exists in the dataset."
                f" Please choose a different 'formatted_output_colname'."
            )

        prompt_fields = []
        use_jinja = True
        if not is_jinja_template(self.prompt):
            # If this is not a Jinja template, but just a legacy string with formattable fields, e.g. `Text: {text}`
            str_formatter = string.Formatter()
            prompt_fields = [
                fld[1] for fld in str_formatter.parse(self.prompt) if fld and len(fld) >= 2 and fld[1] is not None
            ]

            if not prompt_fields:
                self.dataset = KeyDataset(self.dataset, self.text_column)
                return

            use_jinja = False

        self.dataset = self.dataset.map(
            _fill_out_prompt,
            fn_kwargs={
                "column_name_formatted": self.formatted_output_colname,
                "prompt": self.prompt,
                "use_jinja": use_jinja,
                "prompt_fields": prompt_fields,
            },
            batched=True,
            batch_size=10_000,
            desc="Applying prompt",
            num_proc=self.num_workers,
        )
        self.dataset = KeyDataset(self.dataset, self.formatted_output_colname)

        if self.verbose:
            rand_idx = randrange(0, len(self.dataset))
            random_sample = self.dataset[rand_idx]
            print(f"Applied prompt to random example {rand_idx} of {self.dataset_name}:\n{random_sample}")

    def change_model(
        self,
        model_name: str,
        bnb_config: dict[str, Any],
        model_kwargs: dict[str, Any],
        use_chat_template: bool,
        trust_remote_code: bool,
        device: str | int | torch.device = "auto",
    ):
        self._for_saving_bnb_config = deepcopy(self.bnb_config)
        self._for_saving_model_kwargs = deepcopy(self.model_kwargs)
        model_kwargs = model_kwargs or {}
        model_kwargs["trust_remote_code"] = trust_remote_code

        if (
            "torch_dtype" in model_kwargs
            and isinstance(model_kwargs["torch_dtype"], str)
            and model_kwargs["torch_dtype"] != "auto"
        ):
            model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])

        if bnb_config:
            bnb_config = BitsAndBytesConfig(**bnb_config)
            model_kwargs["quantization_config"] = bnb_config

        self.bnb_config = bnb_config or {}
        self.model_kwargs = model_kwargs

        self.model_name = model_name
        self.device = device
        self.use_chat_template = use_chat_template

        self.outlines_model = models.transformers(model_name, model_kwargs=model_kwargs, device=device)
        self.hf_model = self.outlines_model.model
        self.hf_tokenizer = self.outlines_model.tokenizer.tokenizer

    @torch.inference_mode
    def process_dataset(self):
        run_results = {}
        short_model_name = self.model_name.split("/")[-1]
        short_dataset_name = self.dataset_name.split("/")[-1]

        # TODO: rather than doing this in a loop, with full forward passes,
        # we can only redo the sampling itself. This would be much faster and 
        # is even available with Outlines in the samplers. However, that requires
        # a full rework of this loop (wrt batching).
        for run_idx in range(1, self.num_runs + 1):
            pdout = self.output_dir.joinpath(f"run_{run_idx}")
            pdout.mkdir(exist_ok=True, parents=True)

            pred_idxs = []
            with pdout.joinpath("results.jsonl").open("w", encoding="utf-8") as fhout:
                for start_idx in trange(
                    0,
                    len(self.dataset),
                    self.batch_size,
                    desc=f"{short_model_name} run {run_idx}/{self.num_runs} ({short_dataset_name})",
                    position=self.process_id,
                    leave=False,
                ):
                    orig_prompts = self.dataset[start_idx : start_idx + self.batch_size]
                    gold_label_idxs = self.true_label_idxs[start_idx : start_idx + self.batch_size]

                    if self.use_chat_template:
                        prompts = []
                        for prompt in orig_prompts:
                            chat = [{"role": "user", "content": prompt}]
                            if self.system_message:
                                chat = [{"role": "system", "content": self.system_message}] + chat

                            prompts.append(
                                self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                            )
                    else:
                        prompts = orig_prompts

                    generations = outlines.generate.choice(self.outlines_model, self.label_names)(prompts)

                    if self.batch_size == 1:
                        pred_labels = [generations.replace(" ", "").strip()]
                    else:
                        pred_labels = [l.replace(" ", "").strip() for l in generations]

                    for prompt, gold_label_idx, pred_label in zip(orig_prompts, gold_label_idxs, pred_labels):
                        pred_idx = self.labels2idx[pred_label]
                        result = {
                            "prompt": prompt,
                            "gold_label_idx": gold_label_idx,
                            "pred_idx": pred_idx,
                            "gold_label": self.idx2label[gold_label_idx],
                            "pred_label": pred_label,
                        }

                        fhout.write(f"{json.dumps(result)}\n")
                        pred_idxs.append(pred_idx)

            f1 = f1_score(self.true_label_idxs, pred_idxs, average=self.f1_average, labels=self.label_idxs)
            result_str = (
                f"F1 ({self.f1_average}) score on {self.dataset_name} with {self.model_name}"
                f" (use_chat_template={self.use_chat_template}): {f1:.4f}"
            )
            clf_report = classification_report(
                self.true_label_idxs,
                pred_idxs,
                target_names=self.label_names,
                labels=self.label_idxs,
                digits=4,
            )

            pdout.joinpath("report.txt").write_text(result_str + "\n\n" + clf_report, encoding="utf-8")

            clf_report_json = classification_report(
                self.true_label_idxs,
                pred_idxs,
                target_names=self.label_names,
                labels=self.label_idxs,
                output_dict=True,
            )
            run_results[run_idx] = clf_report_json

            pdout.joinpath("scores.json").write_text(json.dumps(clf_report_json, indent=4), encoding="utf-8")

        # Calculate confidence intervals of "accuracy", "macro average" f1, and "weighted avg" f1 of `run_results`
        run_results = add_confidence(run_results)

        if self.verbose:
            heading = f"{self.model_name.split('/')[-1]} on {self.dataset_name.split('/')[-1]}"
            print(f"{heading}\n{'='*len(heading)}")

            for metric in ("accuracy", "macro avg", "weighted avg"):
                print(f"{metric}: {run_results[metric]['mean']*100:.4f} ± {run_results[metric]['ci95']*100:.4f}")

        self.output_dir.joinpath("agg_scores.json").write_text(json.dumps(run_results, indent=4), encoding="utf-8")

        config = deepcopy(self.config())
        if self.save_config_as == "json":
            self.output_dir.joinpath("config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
        elif self.save_config_as == "yaml":
            config["prompt"] = LiteralScalarString(config["prompt"])
            with self.output_dir.joinpath("config.yaml").open("w", encoding="utf-8") as fhout:
                yaml.dump(config, fhout)

    def config(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "labels2idx": self.labels2idx,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "output_dir": str(self.output_dir),
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "label_column": self.label_column,
            "text_column": self.text_column,
            "f1_average": self.f1_average,
            "system_message": self.system_message,
            "use_chat_template": self.use_chat_template,
            "bnb_config": self._for_saving_bnb_config,
            "model_kwargs": self._for_saving_model_kwargs,
            "device": self.device,
            "formatted_output_colname": self.formatted_output_colname,
            "trust_remote_code": self.trust_remote_code,
            "save_config_as": self.save_config_as,
            "batch_size": self.batch_size,
            "num_runs": self.num_runs,
            "verbose": self.verbose,
            "num_workers": self.num_workers,
        }

    @classmethod
    def from_json(cls, config_file: PathLike | str, **kwargs):
        with Path(config_file).open("r", encoding="utf-8") as fhin:
            config = json.load(fhin)

        config = {**config, **kwargs}
        return cls(**config)

    @classmethod
    def from_yaml(cls, config_file: PathLike | str, **kwargs):
        with Path(config_file).open("r", encoding="utf-8") as fhin:
            config = yaml.load(fhin)

        config = {**config, **kwargs}
        return cls(**config)


def _fill_out_prompt(
    samples, prompt: str | Template, prompt_fields: list[str], column_name_formatted: str, use_jinja: bool
) -> dict[str, str]:
    num_items = len(next(iter(samples.values())))
    samples = [{colname: col[sample_idx] for colname, col in samples.items()} for sample_idx in range(num_items)]

    if use_jinja:
        prompt = Template(prompt)
        # Jinja2 will ignore any fields that are not present in the sample (unlike formattable strings, which will
        # throw an error). This is why we need to filter out the fields that are not present in the sample for
        # formattable strings but not for Jinja.
        return {column_name_formatted: [prompt.render(**sample) for sample in samples]}
    else:
        return {
            column_name_formatted: [prompt.format(**{fld: sample[fld] for fld in prompt_fields}) for sample in samples]
        }
