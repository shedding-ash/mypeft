import os
from dataclasses import dataclass, field
from typing import Literal, Optional
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from peft import CLoraConfig, get_peft_model


@dataclass
class ScriptArguments(CLoraConfig):
    # model configs
    lora_rank1: Optional[str] = field(
        default=None, metadata={"help": "The name or path of the fp32/16 base model."}
    )
    lora_rank2: Optional[str] = field(
        default=None, metadata={"help": "The name or path of the fp32/16 base model."}
    )
    