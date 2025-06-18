# YOLO v1
from .darknet import (
    # DarkNet, DarkNet_backbone,
    DarkNet_base, DarkNet_base_backbone,
)

# YOLO v2
from .darknet19 import (
    DarkNet19, DarkNet19_backbone,
    DarkNet19_base, DarkNet19_base_backbone,
)

# YOLO v3
from .darknet53 import (
    DarkNet53, DarkNet53_backbone,
    DarkNet53_base, DarkNet53_base_backbone,
)

# YOLO v4
from .cspdarknet53 import (
    CSPDarkNet53, CSPDarkNet53_backbone,
    CSPDarkNet53_base, CSPDarkNet53_base_backbone,
)

# YOLO v5
from .darknet_c3 import (
    DarkNetC3, DarkNetC3_backbone,
    DarkNetC3_nano, DarkNetC3_nano_backbone,
    DarkNetC3_small, DarkNetC3_small_backbone,
    DarkNetC3_medium, DarkNetC3_medium_backbone,
    DarkNetC3_large, DarkNetC3_large_backbone,
    DarkNetC3_xlarge, DarkNetC3_xlarge_backbone,
)

# YOLO v6
from .efficient_rep import (
    EfficientLite, EfficientLite_backbone,
    EfficientRep, EfficientRep_backbone,
    EfficientLite_small, EfficientLite_small_backbone,
    EfficientLite_medium, EfficientLite_medium_backbone,
    EfficientLite_large, EfficientLite_large_backbone,
    EfficientRep_nano, EfficientRep_nano_backbone,
    EfficientRep6_nano, EfficientRep6_nano_backbone,
    EfficientMBLA_small, EfficientMBLA_small_backbone,
    EfficientRep_small, EfficientRep_small_backbone,
    EfficientRep6_small, EfficientRep6_small_backbone,
    EfficientMBLA_medium, EfficientMBLA_medium_backbone,
    EfficientRep_medium, EfficientRep_medium_backbone,
    EfficientRep6_medium, EfficientRep6_medium_backbone,
    EfficientMBLA_large, EfficientMBLA_large_backbone,
    EfficientRep_large, EfficientRep_large_backbone,
    EfficientRep6_large, EfficientRep6_large_backbone,
    EfficientMBLA_xlarge, EfficientMBLA_xlarge_backbone,
)

# YOLO v7
from .darknet_elan import (
    DarkNetELAN_A, DarkNetELAN_A_backbone,
    DarkNetELAN_B, DarkNetELAN_B_backbone,
    DarkNetELAN_C, DarkNetELAN_C_backbone,
    DarkNetELAN_D, DarkNetELAN_D_backbone,
    DarkNetELAN_tiny, DarkNetELAN_tiny_backbone,
    DarkNetELAN_nano, DarkNetELAN_nano_backbone,
    DarkNetELAN_small, DarkNetELAN_small_backbone,
    DarkNetELAN_medium, DarkNetELAN_medium_backbone,
    DarkNetELAN_large, DarkNetELAN_large_backbone,
    DarkNetELAN_xlarge, DarkNetELAN_xlarge_backbone,
    DarkNetELAN_huge, DarkNetELAN_huge_backbone,
)

# YOLO v8
from .darknet_c2 import (
    DarkNetC2, DarkNetC2_backbone,
    DarkNetC2_nano, DarkNetC2_nano_backbone,
    DarkNetC2_small, DarkNetC2_small_backbone,
    DarkNetC2_medium, DarkNetC2_medium_backbone,
    DarkNetC2_large, DarkNetC2_large_backbone,
    DarkNetC2_xlarge, DarkNetC2_xlarge_backbone,
)

# YOLO v9
from .darknet_elan4 import (
    DarkNetELAN4_A, DarkNetELAN4_A_backbone,
    DarkNetELAN4_B, DarkNetELAN4_B_backbone,
    DarkNetELAN4_small, DarkNetELAN4_small_backbone, 
    DarkNetELAN4_base, DarkNetELAN4_base_backbone, 
    DarkNetELAN4_large, DarkNetELAN4_large_backbone,
    DarkNetELAN4_xlarge, DarkNetELAN4_xlarge_backbone,
)

# YOLO v10
from .darknet_cib import (
    DarkNetCIB, DarkNetCIB_backbone,
    DarkNetCIB_nano, DarkNetCIB_nano_backbone,
    DarkNetCIB_small, DarkNetCIB_small_backbone,
    DarkNetCIB_medium, DarkNetCIB_medium_backbone,
    DarkNetCIB_base, DarkNetCIB_base_backbone,
    DarkNetCIB_large, DarkNetCIB_large_backbone,
    DarkNetCIB_xlarge, DarkNetCIB_xlarge_backbone,
)