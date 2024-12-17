from .mlp_mixer import (MLPMixer, MLPMixer_S16,
                        MLPMixer_S32, MLPMixer_B16,
                        MLPMixer_B32, MLPMixer_L16,
                        MLPMixer_L32, MLPMixer_H14)
from .gated_mlp import gMLP, gMLP_T16, gMLP_S16, gMLP_B16
from .res_mlp import ResMLP, ResMLP_S12, ResMLP_S24, ResMLP_S36, ResMLP_B24
from .wave_mlp import WaveMLP, WaveMLP_T, WaveMLP_S, WaveMLP_M, WaveMLP_B