from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn
from torchvision.models.resnet import resnet18


def build_res_unet(n_input=1, n_output=2, size=256, device="cpu"):
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    generator = DynamicUnet(body, n_output, (size, size)).to(device)
    return generator


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(2, 2), padding=1),
        )

    def forward(self, x):
        return self.model(x)
