from .resnet import (ResNetA, ResNetB,
                     ResNet18, ResNet18_backbone,
                     ResNet34, ResNet34_backbone,
                     ResNet50, ResNet50_backbone,
                     ResNet101, ResNet101_backbone,
                     ResNet152, ResNet152_backbone)
from .bit import (ResnetV2,
                  BiT_S_R50x1, BiT_S_R50x3,
                  BiT_M_R50x1, BiT_M_R50x3,
                  BiT_S_R101x1, BiT_S_R101x3,
                  BiT_M_R101x1, BiT_M_R101x3,
                  BiT_S_R152x4, BiT_M_R152x4)
from .res2net import (Res2Net,
                      Res2Net50, Res2Net50_backbone,
                      Res2Net101, Res2Net101_backbone,
                      Res2Net152, Res2Net152_backbone)
