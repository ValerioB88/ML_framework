from typing import Any

import torch.nn as nn
import torch
from models.model_utils import Flatten, conv_block
import torch.nn.functional as F
import numpy as np
from framework_utils import make_cuda

class InvarianceSupervisedModel(nn.Module):
    def __init__(self, inv_model, superv_model):
        self.inv_model = inv_model
        self.superv_model = superv_model

    def forward(self, input):
        x = self.inv_model(input)
        x = self.superv_model(x)
        return x