import torch.nn as nn
from typing import Any

import torch
from models.model_utils import Flatten, conv_block
import torch.nn.functional as F
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class SequenceNtrain1cand(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self):
        super().__init__()
        size_embedding_frame = 64
        self.image_embedding_training = nn.Sequential(conv_block(3, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             nn.AdaptiveAvgPool2d(1),
                                             Flatten())  # output = 1 x 64
        self.image_embedding_candidates = nn.Sequential(conv_block(3, 64, bias=False),
                                                      conv_block(64, 64, bias=False),
                                                      conv_block(64, 64, bias=False),
                                                      conv_block(64, 64, bias=False),
                                                      nn.AdaptiveAvgPool2d(1),
                                                      Flatten())  # output = 1 x 64

        hidden_size = 64
        size_embedding_output = 256
        self.encoder_fr2seq = EncoderRNN(input_size=64, hidden_size=hidden_size)
        # you could put another linear layer that transforms the hidden output from the first encoder to something for the second one
        self.encoder_seq2obj = EncoderRNN(input_size=hidden_size, hidden_size=hidden_size)

        self.relation_net = nn.Sequential(nn.Linear(64, 256),
                                          nn.BatchNorm1d(256),
                                          nn.LeakyReLU(True),
                                          nn.Linear(256, 1),
                                          nn.Sigmoid())

        self._initialize_weights()

    def forward(self, frame, hidden):
        pass


class SequenceMatchingNetSimple(nn.Module):
    # Very simple version. We compute an embedding of the image, feed it into an RNN, take the last output hidden activation
    # feed it into a linear classifier to get a class
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self):
        super().__init__()
        size_embedding_frame = 64
        self.image_embedding = nn.Sequential(conv_block(3, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             nn.AdaptiveAvgPool2d(1),
                                             Flatten())  # output = 1 x 64

        hidden_size = 64
        size_embedding_output = 256
        self.encoder_fr2seq = EncoderRNN(input_size=64, hidden_size=hidden_size)
        # you could put another linear layer that transforms the hidden output from the first encoder to something for the second one
        self.encoder_seq2obj = EncoderRNN(input_size=hidden_size, hidden_size=hidden_size)

        self.relation_net = nn.Sequential(nn.Linear(64, 256),
                                          nn.BatchNorm1d(256),
                                          nn.LeakyReLU(True),
                                          nn.Linear(256, 1),
                                          nn.Sigmoid())

        self._initialize_weights()
    def forward(self, frame, hidden):
        pass

##

