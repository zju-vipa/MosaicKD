# Copyright 2020 Zhejiang Lab. All Rights Reserved.
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
# =============================================================

import numpy as np
import torch
from .stream_metrics import Metric
from typing import Callable

__all__=['Accuracy', 'TopkAccuracy']

class Accuracy(Metric):
    def __init__(self):
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs = outputs.max(1)[1]
        self._correct += ( outputs.view(-1)==targets.view(-1) ).sum()
        self._cnt += torch.numel( targets )

    def get_results(self):
        return (self._correct / self._cnt).detach().cpu()
    
    def reset(self):
        self._correct = self._cnt = 0.0


class TopkAccuracy(Metric):
    def __init__(self, topk=5):
        self._topk = topk
        self.reset()
    
    @torch.no_grad()
    def update(self, outputs, targets):
        _, outputs = outputs.topk(self._topk, dim=1, largest=True, sorted=True)
        correct = outputs.eq( targets.view(-1, 1).expand_as(outputs) )
        self._correct += correct[:, :self._topk].view(-1).float().sum(0).item()
        self._cnt += len(targets)
    
    def get_results(self):
        return self._correct / self._cnt

    def reset(self):
        self._correct = 0.0
        self._cnt = 0.0