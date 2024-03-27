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
                            DarkNet53, DarkNet53_backbone,
                            CSPDarkNet53, CSPDarkNet53_backbone,
                            DarkNetC3, DarkNetC3_nano, DarkNetC3_nano_backbone, DarkNetC3_small, DarkNetC3_small_backbone, DarkNetC3_medium, DarkNetC3_medium_backbone, DarkNetC3_large, DarkNetC3_large_backbone, DarkNetC3_xlarge, DarkNetC3_xlarge_backbone,
                            EfficientRep, EfficientLite, EfficientRep_nano, EfficientRep_nano_backbone, EfficientRep6_nano, EfficientRep6_nano_backbone, EfficientRep_small, EfficientRep_small_backbone, EfficientRep6_small, EfficientRep6_small_backbone, EfficientRep_medium, EfficientRep_medium_backbone, EfficientRep6_medium, EfficientRep6_medium_backbone, EfficientRep_large, EfficientRep_large_backbone, EfficientRep6_large, EfficientRep6_large_backbone, EfficientMBLA_small, EfficientMBLA_small_backbone, EfficientMBLA_medium, EfficientMBLA_medium_backbone, EfficientMBLA_large, EfficientMBLA_large_backbone, EfficientMBLA_xlarge, EfficientMBLA_xlarge_backbone, EfficientLite_small, EfficientLite_small_backbone, EfficientLite_medium, EfficientLite_medium_backbone, EfficientLite_large, EfficientLite_large_backbone,
                            DarkNetELAN_A, DarkNetELAN_B, DarkNetELAN_C, DarkNetELAN_D, DarkNetELAN_E, DarkNetELAN_tiny, DarkNetELAN_tiny_backbone, DarkNetELAN_nano, DarkNetELAN_nano_backbone, DarkNetELAN_small, DarkNetELAN_small_backbone, DarkNetELAN_medium, DarkNetELAN_medium_backbone, DarkNetELAN_large, DarkNetELAN_large_backbone, DarkNetELAN_xlarge, DarkNetELAN_xlarge_backbone, DarkNetELAN_huge, DarkNetELAN_huge_backbone,
                            DarkNetC2, DarkNetC2_nano, DarkNetC2_nano_backbone, DarkNetC2_small, DarkNetC2_small_backbone, DarkNetC2_medium, DarkNetC2_medium_backbone, DarkNetC2_large, DarkNetC2_large_backbone, DarkNetC2_xlarge, DarkNetC2_xlarge_backbone,
                            DarkNetELAN4_A, DarkNetELAN4_B, DarkNetELAN4_small, DarkNetELAN4_small_backbone, DarkNetELAN4_base, DarkNetELAN4_base_backbone, DarkNetELAN4_Large,
                            ResnetV2, BiT_S_R50x1, BiT_S_R50x3, BiT_M_R50x1, BiT_M_R50x3, BiT_S_R101x1, BiT_S_R101x3, BiT_M_R101x1, BiT_M_R101x3, BiT_S_R152x4, BiT_M_R152x4,
                            RepVGG, RepVGG_A0, RepVGG_A1, RepVGG_A2, RepVGG_B0, RepVGG_B1, RepVGG_B1g2, RepVGG_B1g4, RepVGG_B2, RepVGG_B2g2, RepVGG_B2g4, RepVGG_B3, RepVGG_B3g2, RepVGG_B3g4, repvgg_reparameter,
                            ConvNext, ConvNextT, ConvNextT_backbone, ConvNextS, ConvNextS_backbone, ConvNextB, ConvNextB_backbone, ConvNextL, ConvNextL_backbone, ConvNextXL, ConvNextXL_backbone,
)


# transfomer base architectures
from .architectures import (MLPMixer, MLPMixer_S16, MLPMixer_S32, MLPMixer_B16, MLPMixer_B32, MLPMixer_L16, MLPMixer_L32, MLPMixer_H14,
                            gMLP, gMLP_T16, gMLP_S16, gMLP_B16,
                            ResMLP, ResMLP_S12, ResMLP_S24, ResMLP_S36, ResMLP_B24,
                            WaveMLP, WaveMLP_T, WaveMLP_S, WaveMLP_M, WaveMLP_B,
                            ViT, ViTBase16, ViTBase32, ViTLarge16, ViTLarge32, ViTHuge16, ViTHuge32,
                            BEiT, BEiTBase16, BEiTBase32, BEiTLarge16, BEiTLarge32, BEiTHuge16, BEiTHuge32,
                            DeiT, DeiT_Ti, DeiT_S, DeiT_B,
                            ViT_BEiT, ViT_BEiT_Tiny16, ViT_BEiT_Tiny32, ViT_BEiT_Base16, ViT_BEiT_Base32, ViT_BEiT_Large14, ViT_BEiT_Large16, ViT_BEiT_Large32, ViT_BEiT_Huge14, ViT_BEiT_Huge16, ViT_BEiT_Huge32,
                            FlexiViT, FlexiViT_Small16, FlexiViT_Base16, FlexiViT_Large16,
                            MetaTransformer, MetaTransformer_Base16, MetaTransformer_Large14, MetaTransformer_Large16,
                            DINOv2, DINOv2_Small14, DINOv2_Base14, DINOv2_Large14, DINOv2_Huge14, DINOv2_Gaint14,
                            EVA, EVA_Large14, EVA_Gaint14,
                            EVA02, EVA02_Tiny14, EVA02_Small14, EVA02_Base14, EVA02_Large14,
                            Swin, SwinT, SwinS, SwinB, SwinL,
                            Swin_v2, SwinT_v2, SwinS_v2, SwinB_v2, SwinL_v2,
                            ViTImageEncoder, ViTImageEncoder_base, ViTImageEncoder_large, ViTImageEncoder_huge,
)
