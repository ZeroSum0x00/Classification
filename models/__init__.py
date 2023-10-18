# train-test processing model
from .classification import CLS


# convolutional base architectures
from .architectures import (AlexNet,
                            VGG, VGG11, VGG13, VGG16, VGG19,
                            ResNetA, ResNetB, ResNet18, ResNet18_backbone, ResNet34, ResNet34_backbone, ResNet50, ResNet50_backbone,  ResNet101, ResNet101_backbone, ResNet152, ResNet152_backbone,
                            GoogleNet, Inception_v1_naive, Inception_v1,
                            Inception_v3,
                            Inception_v4, Inception_Resnet_v1, Inception_Resnet_v2,
                            SqueezeNet,
                            MobileNet_v1,
                            MobileNet_v2,
                            DenseNet, DenseNet121, DenseNet121_backbone, DenseNet169, DenseNet169_backbone, DenseNet201, DenseNet201_backbone, DenseNet264, DenseNet264_backbone,
                            EfficientNet, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
                            Res2Net, Res2Net50, ResNet50_backbone, Res2Net101, ResNet101_backbone, Res2Net152, ResNet152_backbone,
                            Xception,
                            DDRNet23_slim, DDRNet23, DDRNet39,
                            ResnetV2, BiT_S_R50x1, BiT_S_R50x3, BiT_M_R50x1, BiT_M_R50x3, BiT_S_R101x1, BiT_S_R101x3, BiT_M_R101x1, BiT_M_R101x3, BiT_S_R152x4, BiT_M_R152x4,
                            ConvNext, ConvNextT, ConvNextT_backbone, ConvNextS, ConvNextS_backbone, ConvNextB, ConvNextB_backbone, ConvNextL, ConvNextL_backbone, ConvNextXL, ConvNextXL_backbone,
)


# transfomer base architectures
from .architectures import (MLPMixer, MLPMixer_S16, MLPMixer_S32, MLPMixer_B16, MLPMixer_B32, MLPMixer_L16, MLPMixer_L32, MLPMixer_H14,
                            gMLP, gMLP_T16, gMLP_S16, gMLP_B16,
                            ResMLP, ResMLP_S12, ResMLP_S24, ResMLP_S36, ResMLP_B24,
                            WaveMLP, WaveMLP_T, WaveMLP_S, WaveMLP_M, WaveMLP_B,
                            ViT, ViTBase16, ViTBase32, ViTLarge16, ViTLarge32, ViTHuge16, ViTHuge32,
                            DeiT,
                            Swin, SwinT, SwinS, SwinB, SwinL,
                            Swin_v2, SwinT_v2, SwinS_v2, SwinB_v2, SwinL_v2,
)