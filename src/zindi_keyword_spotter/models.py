import torch
import torch.nn.functional as F
from torch import nn
from torch.tensor import Tensor
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


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
    
    def forward(self, x: Tensor) -> Tensor:
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


class ScaleBlock(nn.Module):

    def __init__(self, n_in: int, n_out:int, bottleneck: int = 16) -> None:
        super().__init__()
        self.scale_block = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=bottleneck, out_channels=n_out, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.scale_block(x)


class ResBlock(nn.Module):

    def __init__(self, n_in: int, bottleneck: int, n_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=bottleneck, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=bottleneck)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=bottleneck, out_channels=n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=n_out)
        self.scale = ScaleBlock(n_in=n_out, n_out=n_out)
    
    def forward(self, x: Tensor) -> Tensor:
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        conv2_out = self.bn2(self.conv2(conv1_out))
        scale_out = self.scale(conv2_out)
        out = x + scale_out * conv2_out
        return self.relu(out)


class SeResNet3(nn.Module):

    def __init__(
        self,
        num_classes: int,
        hop_length: int,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        power: float,
        normalize: bool,
        use_decibels: bool,
    ) -> None:
        super().__init__()
        self.use_decibels = use_decibels

        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            normalized=normalize,
        )
        self.amplitude2db = AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(num_features=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.res1 = ResBlock(n_in=16, bottleneck=16, n_out=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.res2 = ResBlock(n_in=32, bottleneck=32, n_out=32)
        self.res3 = ResBlock(n_in=32, bottleneck=32, n_out=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.res4 = ResBlock(n_in=64, bottleneck=64, n_out=64)
        self.res5 = ResBlock(n_in=64, bottleneck=64, n_out=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.res6 = ResBlock(n_in=128, bottleneck=128, n_out=128)
        self.res7 = ResBlock(n_in=128, bottleneck=128, n_out=128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)

        self.logits = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        mel = self.melspectrogram(x)
        if self.use_decibels:
            mel = self.amplitude2db(mel)
        # (N, H, W) -> (N, C, H, W) & bn
        norm_x = self.input_bn(mel.unsqueeze(1))

        out = self.conv1(norm_x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.res1(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.dropout(out, p=0.1)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.res2(out)
        out = self.res3(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.dropout(out, p=0.2)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.res4(out)
        out = self.res5(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.dropout(out, p=0.2)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = self.res6(out)
        out = self.res7(out)
        out = F.dropout(out, p=0.2)

        out = self.conv5(out)
        out = self.bn5(out)
        out = F.relu(out)
        # global avg pooling - get mean from each channel
        # (N, C, H, W) -> (N, C)
        feature_vector = out.mean(dim=(2, 3))
        return self.logits(feature_vector)


class ResNest(nn.Module):
    def __init__(
            self,
            num_classes: int,
            hop_length: int,
            sample_rate: int,
            n_mels: int,
            n_fft: int,
            power: float,
            normalize: bool,
            use_decibels: bool,
            resnest_name: str,
            pretrained: bool = True
    ) -> None:
        """
        :param resnest_name: one of ['resnest101', 'resnest200', 'resnest269', 'resnest50', 'resnest50_fast_1s1x64d',
            'resnest50_fast_1s2x40d', 'resnest50_fast_1s4x24d', 'resnest50_fast_2s1x64d', 'resnest50_fast_2s2x40d',
            'resnest50_fast_4s1x64d', 'resnest50_fast_4s2x40d']
        """
        super().__init__()
        self.use_decibels = use_decibels
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            normalized=normalize,
        )
        self.amplitude2db = AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(num_features=1)

        self.resnest = torch.hub.load('zhanghang1989/ResNeSt', resnest_name, pretrained=pretrained)
        self.resnest.fc = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        mel = self.melspectrogram(x)
        if self.use_decibels:
            mel = self.amplitude2db(mel)
        # (N, H, W) -> (N, C, H, W) & bn
        norm_x = self.input_bn(mel.unsqueeze(1))
        norm_x = norm_x.repeat(1, 3, 1, 1)
        out = self.resnest(norm_x)
        return out


class WideConvolutionsModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        hop_length: int,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        power: float,
        normalize: bool,
        use_decibels: bool,
    ) -> None:
        super().__init__()
        self.use_decibels = use_decibels
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            normalized=normalize,
        )
        self.amplitude2db = AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(num_features=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[7, 3])
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 7])
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 10])
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[7, 1])
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.logits = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        mel = self.melspectrogram(x)
        if self.use_decibels:
            mel = self.amplitude2db(mel)
        # (N, H, W) -> (N, C, H, W) & bn
        norm_x = self.input_bn(mel.unsqueeze(1))
        out = self.conv1(norm_x)
        out = self.bn1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=[1, 3])

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=[1, 4])

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=[1, 10])

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=[7, 1])
        # global avg pooling - get mean from each channel
        # (N, C, H, W) -> (N, C)
        feature_vector = out.mean(dim=(2, 3))
        return self.logits(feature_vector)


