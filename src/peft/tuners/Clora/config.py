# Copyright 2023-present the HuggingFace Inc. team.
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

import warnings
from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType

@dataclass
class CLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.CLora`].

    Args:
        r1 (:obj:`int`, `optional`, defaults to 8):
            The dimension of the first matrix in the CLora model.
        r2 (:obj:`int`, `optional`, defaults to 8):
            The dimension of the second matrix in the CLora model.
    """
    r1: int = field(default=8, metadata={"help": "Lora1 matrix dimension."})
    r2: int = field(default=8, metadata={"help": "Lora2 matrix dimension."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CLORA

        if self.use_dora:
            raise ValueError(f"{self.peft_type} does not support DoRA.")

        if self.loftq_config:
            raise ValueError(f"{self.peft_type} does not support LOFTQ.")

