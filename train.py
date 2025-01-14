# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

import argparse
import os
import logging
import sys
from dataclasses import dataclass

from config import Config
import numpy as np
import torch
import transformers
from transformers.trainer_utils import is_main_process
from optimum.habana import GaudiConfig, GaudiSeq2SeqTrainer, GaudiSeq2SeqTrainingArguments    
from optimum.habana.utils import set_seed
from models import load_model
from dataset.AudioCaps import AudioCaps
import json

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

epoch = 0

def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    return parser.parse_args()

@dataclass
class DataCollatorForCaptioning:
    tokenizer: transformers.PreTrainedTokenizer
    max_txt_len: int = 128

    def __call__(self, samples):
        samples_spectrogram = [s["spec"].unsqueeze(0) for s in samples]
        samples_clip_features = [torch.tensor(s["clip_feat"]).unsqueeze(0) for s in samples]
        labels = [s["text"] for s in samples]
        vid_id = [s["vid_id"] for s in samples]
        torch_id = [s["torch_id"] for s in samples]

        to_regress_tokens = self.tokenizer(
            labels,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        )

        return {
            "spec": torch.cat(samples_spectrogram, dim=0),
            "clip_feat": torch.cat(samples_clip_features, dim=0),
            "labels": to_regress_tokens.input_ids,
            "labels_attention_mask": to_regress_tokens.attention_mask,
            "vid_id": vid_id,
            "torch_id": torch.tensor(torch_id),
        }


def main():
    # load config
    cfg = Config(parse_args())
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    set_seed(run_config.seed)

    if run_config.do_debug:
        run_config.output_dir = './debug'
        run_config.num_workers = 1
        os.environ["WANDB_DISABLED"] = "true"
        run_config.train_batch_size = 1
    else:
        os.environ["WANDB_PROJECT"] = run_config.wandb.wandb_project


    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg.pretty_print()

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True
    
    training_args = GaudiSeq2SeqTrainingArguments(
        per_device_train_batch_size=run_config['train_batch_size'],
        per_device_eval_batch_size=run_config['eval_batch_size'],
        gradient_accumulation_steps=run_config['accum_grad_iters'],
        warmup_ratio=run_config.optims['warmup_ratio'],
        num_train_epochs=run_config.optims['max_epoch'],
        learning_rate=run_config.optims['init_lr'],
        weight_decay=run_config.optims['weight_decay'],
        bf16=True,
        # fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        # eval_strategy="epoch",
        do_eval=False,
        # save_strategy="epoch",
        # logging_strategy="epoch",
        dataloader_num_workers=0 if run_config.do_debug else run_config.num_workers,
        output_dir=run_config['output_dir'],
        save_steps=1500,
        save_total_limit=30,
        load_best_model_at_end=False,
        report_to=None if run_config.do_debug else "wandb",
        run_name=None if run_config.do_debug else run_config.wandb.wandb_name,
        remove_unused_columns=False,
        label_names=["text"],
        ddp_find_unused_parameters=True,
        use_habana=True,
        use_lazy_mode=False,
        predict_with_generate=True,
        include_inputs_for_metrics=True,
    )

    # build model
    model = load_model(model_config)
    tokenizer = model.llama_tokenizer

    # build datasets
    datasets = {
        "train": AudioCaps(data_config, data_config.train_data_path),
        "valid": AudioCaps(data_config, data_config.valid_data_path),
    }

    
    if run_config.do_debug:
        datasets["train"].annotation = datasets["train"].annotation[:1]
        # datasets["valid"].annotation = datasets["valid"].annotation[:10]

    data_collator = DataCollatorForCaptioning(tokenizer=tokenizer, max_txt_len=256)   

    with open('/mnt/lynx2/datasets/audiocaps/torch2iid.json') as f:
        torch2iid =json.load(f)
    
    def compute_metrics(eval_preds):
        preds, _, torch_ids = eval_preds
        preds = np.where(preds < tokenizer.vocab_size, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        coco_result_json, ids = [], []
        for torch_id, pred in zip(torch_ids, decoded_preds):
            try:
                iid = torch2iid[str(torch_id)]
            except:
                continue
            if iid in ids:
                continue
            coco_result_json.append({
                "image_id": iid,
                "caption": pred
            })
            ids.append(iid)
        print(len(coco_result_json))

        global epoch
        epoch += 1

        save_path = os.path.join(training_args.output_dir, f"epoch{epoch}_coco_result.json")
        with open(save_path, 'w') as f:
            json.dump(coco_result_json, f)

        annotation_file = data_config.annot_path

        coco = COCO(annotation_file)
        coco_result = coco.loadRes(save_path)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params['image_id'] = coco_result.getImgIds()
        coco_eval.evaluate()

        metrics = {}
        for metric_name, score in coco_eval.eval.items():
            metrics[metric_name] = score
        print(metrics)

        return metrics

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {True}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    trainer = GaudiSeq2SeqTrainer(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['valid'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if not run_config.do_eval:
        trainer.train()
    else:
        trainer._load_from_checkpoint(resume_from_checkpoint=model_config.resume_from)
        trainer.predict(test_dataset=datasets['valid'])


    
    # state_dict = torch.load(model_config.resume_from, map_location='cpu')
    # msg = trainer.model.load_state_dict(state_dict['model'], strict=False)
    # result = trainer.predict(test_dataset=datasets['valid'])
    # print(result)

if __name__ == "__main__":
    main()