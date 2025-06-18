from .resnet import (
    ResNet, ResNet_backbone,
    ResNet18, ResNet18_backbone,
    ResNet34, ResNet34_backbone,
    ResNet50, ResNet50_backbone,
    ResNet101, ResNet101_backbone,
    ResNet152, ResNet152_backbone,
)

from .ddrnet import (
    DDRNet23, DDRNet23_backbone,
    DDRNet23_slim, DDRNet23_slim_backbone,
    DDRNet23_base, DDRNet23_base_backbone,
    DDRNet39, DDRNet39_backbone,
    DDRNet39_base, DDRNet39_base_backbone,
)

from .bit import (
    ResnetV2, ResnetV2_backbone,
    BiT_R50x1, BiT_R50x1_backbone,
    BiT_R50x3, BiT_R50x3_backbone,
    BiT_R101x1, BiT_R101x1_backbone,
    BiT_R101x3, BiT_R101x3_backbone,
    BiT_R152x4, BiT_R152x4_backbone,
)

from .res2net import (
    Res2Net, Res2Net_backbone,
    Res2Net50, Res2Net50_backbone,
    Res2Net50_26w4s, Res2Net50_26w4s_backbone,
    Res2Net50_26w6s, Res2Net50_26w6s_backbone,
    Res2Net50_26w8s, Res2Net50_26w8s_backbone,
    Res2Net50_48w2s, Res2Net50_48w2s_backbone,
    Res2Net50_14w8s, Res2Net50_14w8s_backbone,
    Res2Net101, Res2Net101_backbone,
    Res2Net101_26w4s, Res2Net101_26w4s_backbone,
    Res2Net101_26w6s, Res2Net101_26w6s_backbone,
    Res2Net101_26w8s, Res2Net101_26w8s_backbone,
    Res2Net101_48w2s, Res2Net101_48w2s_backbone,
    Res2Net101_14w8s, Res2Net101_14w8s_backbone,
    Res2Net152, Res2Net152_backbone,
    Res2Net152_26w4s, Res2Net152_26w4s_backbone,
    Res2Net152_26w6s, Res2Net152_26w6s_backbone,
    Res2Net152_26w8s, Res2Net152_26w8s_backbone,
    Res2Net152_48w2s, Res2Net152_48w2s_backbone,
    Res2Net152_14w8s, Res2Net152_14w8s_backbone,
)