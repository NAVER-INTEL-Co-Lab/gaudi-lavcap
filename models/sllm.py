import os
import random
import json
import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, LlamaTokenizer, LlamaForCausalLM, StoppingCriteriaList, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model
from optimum.habana.transformers.generation import GaudiGenerationConfig

from models.conv import Conv1dSubsampler
from models.utils import StoppingCriteriaSub

class SLLMConfig(PretrainedConfig):
    model_type = "sllm"

    def __init__(self, **kwargs):
        self.llama_path = kwargs.get("llama_path")

        self.speech_llama_proj_model = kwargs.get("speech_llama_proj_model", "")
        self.freeze_speech_llama_proj = kwargs.get("freeze_speech_llama_proj", False)
        self.freeze_speech_conv = kwargs.get("freeze_speech_conv", False)

        self.lora = kwargs.get("lora", True)
        self.lora_rank = kwargs.get("lora_rank", 8)
        self.lora_alpha = kwargs.get("lora_alpha", 32)
        self.lora_dropout = kwargs.get("lora_dropout", 0.1)

        self.multi_prompt = kwargs.get("multi_prompt", False)
        self.prompt_path = kwargs.get("prompt_path", "")
        self.prompt_template = kwargs.get("prompt_template", "")
        self.max_txt_len = kwargs.get("max_txt_len", 128)
        self.end_sym = kwargs.get("end_sym", "</s>")
        self.low_resource = kwargs.get("low_resource", False)
        self.device_8bit = kwargs.get("device_8bit", 0)

        self.generation_config = kwargs.get("generate", None)

        

class SLLM(PreTrainedModel):
    config_class = SLLMConfig

    @property
    def device(self):
        return list(self.parameters())[0].device

    def __init__(self, config: SLLMConfig):
        super().__init__(config)

        self.speech_dim = 1280

        self.llama_path = config.llama_path

        self.speech_llama_proj_model = config.speech_llama_proj_model
        self.freeze_speech_llama_proj = config.freeze_speech_llama_proj
        self.freeze_speech_conv = config.freeze_speech_conv

        self.lora = config.lora
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout

        self.multi_prompt = config.multi_prompt
        self.prompt_path = config.prompt_path
        self.prompt_template = config.prompt_template
        self.max_txt_len = config.max_txt_len
        self.end_sym = config.end_sym
        self.low_resource = config.low_resource
        self.device_8bit = config.device_8bit

        self.generation_config = GaudiGenerationConfig(
            max_new_tokens=config.generation_config.max_new_tokens,
            static_shapes=False, ## GaudiGenerationMixin.generate에서 model_input_name 가 input_embeds면 False로 해야하는듯
            num_beams=config.generation_config.get("num_beams", 1),
            do_sample=config.generation_config.get("do_sample", False),
        )

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.llama_path)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) ## pad 원래 없음??
        self.llama_tokenizer.padding_side = "right"


        logging.info('Loading LLaMA Model')
        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "trust_remote_code": None,
            "use_cache": True,
            "token": None,
        }
        llama_config = AutoConfig.from_pretrained(self.llama_path, **config_kwargs)
        self.llama_model = LlamaForCausalLM.from_pretrained(
            self.llama_path,
            config=llama_config,
            device_map='hpu'
        )
        
        self.config = self.llama_model.config

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for param in self.llama_model.parameters():
            param.requires_grad = False
        
        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            logging.info('LoRA Training')

        self.ln_speech = nn.LayerNorm(self.speech_dim)

        self.speech_conv = Conv1dSubsampler(
            in_channels=self.speech_dim,
            mid_channels=self.speech_dim,
            out_channels=768,
            kernel_sizes=[3, 3, 3, 3],
        )
        self.speech_llama_proj = nn.Linear(768, self.llama_model.config.hidden_size)

        if self.freeze_speech_conv:
            for name, param in self.speech_conv.named_parameters():
                param.requires_grad = False
            for param in self.ln_speech.parameters():
                param.requires_grad = False
            self.speech_conv.eval()
            self.ln_speech.eval()
        
        if self.freeze_speech_llama_proj:
            for param in self.speech_llama_proj.parameters():
                param.requires_grad = False
            self.speech_llama_proj.eval()
        

        # prepare prompts
        self.prompt_dict = {}
        if self.prompt_path:
            try:
                raw_prompts = json.load(open(self.prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(self.prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if ("<SpeechHere>" in raw_prompt) or ("<KeywordHere>" in raw_prompt)]
                self.prompt_dict[task] = [self.prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

    def _encode_auditory_feature(self, speech_embeds):
        speech_embeds = self.ln_speech(speech_embeds)
        speech_embeds = self.speech_conv(speech_embeds)
        llama_input = self.speech_llama_proj(speech_embeds)
        llama_atts = torch.ones(llama_input.size()[:-1], dtype=torch.long).to(llama_input.device)
        return llama_input, llama_atts
    
    def prompt_wrap(self, embeds, atts, prompt, task, multi_prompt=False):
        batch_size = embeds.shape[0]
        if task == "keyword":
            p_before, p_after = prompt.split("<KeywordHere>")
        else:
            p_before, p_after = prompt.split("<SpeechHere>") 

        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False
        ).to(embeds.device)


        p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
        
        p_before_att = p_before_tokens.attention_mask.expand(batch_size, -1)
        p_after_att = p_after_tokens.attention_mask.expand(batch_size, -1)
        wrapped_atts = torch.cat([p_before_att, atts, p_after_att], dim=1)
        return wrapped_embeds, wrapped_atts
      
    def forward(self, **kwargs):
        speech_embeds, speech_atts = self._encode_auditory_feature(kwargs['spectrogram'])

        prompt = random.choice(self.prompt_dict[kwargs['task'][0]])

        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, kwargs['task'], self.multi_prompt)
        
        to_regress_tokens = kwargs['labels']
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens)
        targets = to_regress_tokens.masked_fill(
            to_regress_tokens == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(speech_embeds.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.dtype,
            device=to_regress_tokens.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, kwargs['labels_attention_mask']], dim=1)

        # calulate loss
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}
    
    def generate(self, **kwargs):
        prompt = "[INST] <<SYS>>\n \n<</SYS>>\n\n###Human: <Speech><SpeechHere></Speech> Recognize the speech and give me the transcription.\n [/INST] ###Assistant: \n"

        speech_embeds, speech_atts = self._encode_auditory_feature(kwargs['spectrogram'])
        speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, kwargs['task'][0], False)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=speech_embeds.device
        ) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).to("hpu")]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            attention_mask=attns,
            stopping_criteria=stopping_criteria,
            generation_config=self.generation_config
        )
        return outputs



    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config, *model_args, **kwargs):
        cfg = SLLMConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        cfg.update(config.to_dict())
        model = cls(cfg)

        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        return model
    
    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters()
        }
        state_dict = self.state_dict() # 
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                if 'speech_conv' in k:
                    continue
                if 'ln_speech' in k:
                    continue
                if 'speech_llama_proj' in k:
                    continue
                # delete parameters that do not require gradient
                del state_dict[k]
        torch.save(state_dict, model_path)