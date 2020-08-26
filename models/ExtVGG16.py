from torchvision.models.vgg import VGG, make_layers
import torch.nn as nn
import torch

def vgg16np(**kwargs):
    # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    cfg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    kwargs['init_weights'] = True
    model = VGG_NoPooling(make_layers(cfg, batch_norm=False), **kwargs)
    return model


class VGG_NoPooling(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 224 * 224, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
