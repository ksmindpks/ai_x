import torch
import torch.nn as nn
from torchvision import models

def build_efficientnetb3_model(num_classes, pretrained=True):
    """
    EfficientNet-B3 기반 브랜드 분류 모델 생성
    :param num_classes: 출력 클래스 수 (예: 9개 브랜드)
    :param pretrained: ImageNet 사전학습 가중치 사용 여부
    :return: nn.Module 모델
    """
    model = models.efficientnet_b3(pretrained=pretrained)

    # 기존 classifier 구조: (Dropout(p=0.3), Linear(1536, 1000))
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    return model