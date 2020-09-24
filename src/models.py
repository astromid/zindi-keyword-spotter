import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import sampler
from torchaudio.transforms import MelSpectrogram


class PalSolModel(nn.Module):

    def __init__(self, num_classes: int, sample_rate: int) -> None:
        super().__init__()
        self.melspectrogram = MelSpectrogram(sample_rate=sample_rate)
        self.norm_input = nn.BatchNorm2d(num_features=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)

        self.fc = nn.Linear(in_features=1536, out_features=256)
        self.fc_norm = nn.BatchNorm1d(num_features=256)
        self.linear = nn.Linear(in_features=256, out_features=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.melspectrogram(x)
        x = x.unsqueeze(1)
        x = self.norm_input(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.drop1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.drop2(x)

        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = self.drop2(x)

        x = F.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.drop2(x)

        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = self.drop2(x)

        x = self.fc(x.flatten(start_dim=1))
        x = self.fc_norm(x)
        return self.linear(x)

