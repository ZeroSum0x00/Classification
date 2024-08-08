"""
  # Description:
    - The following table comparing the params of the DarkNet 53 with CIB Block (YOLOv10 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

       -----------------------------------------
      |      Model Name       |     Params      |
      |-----------------------------------------|
      |    DarkNetCIB nano    |    1,464,536    |
      |-----------------------------------------|
      |    DarkNetCIB small   |    4,422,600    |
      |-----------------------------------------|
      |    DarkNetCIB medium  |    9,071,896    |
      |-----------------------------------------|
      |    DarkNetCIB large   |   15,499,432    |
      |-----------------------------------------|
      |    DarkNetCIB xlarge  |   15,526,360    |
       -----------------------------------------

  # Reference:
    - Source: https://github.com/THU-MIG/yolov10

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file

from .darknet53 import ConvolutionBlock
from .darknet_c3 import SPP, SPPF
from .darknet_c2 import C2f, LightConvolutionBlock, Bottleneck2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape



class SCDown(LightConvolutionBlock):

    def __init__(self,
                 filters,
                 kernel_size,
                 downsample=False,
                 activation='relu', 
                 normalizer='batch-norm',
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         kernel_size,
                         activation,
                         normalizer,
                         *args,
                         **kwargs)
        self.downsample = downsample

    def build(self, input_shape):
        super().build(input_shape)
        self.conv2 = ConvolutionBlock(self.filters, self.kernel_size, downsample=self.downsample, groups=self.filters, activation=None, normalizer=self.normalizer)


class RepVGGDW(tf.keras.layers.Layer):
    
    def __init__(self,
                 filters,
                 activation='silu', 
                 normalizer='batch-norm',
                 *args, 
                 **kwargs):
        super(RepVGGDW, self).__init__(*args, **kwargs)
        self.filters       = filters
        self.activation    = activation
        self.normalizer    = normalizer

    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(self.filters, (7, 7), groups=self.filters, activation=None, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, (3, 3), groups=self.filters, activation=None, normalizer=self.normalizer)
        self.activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        out = x1 + x2
        out = self.activ(out)
        return out


class CIB(tf.keras.layers.Layer):

    def __init__(self, 
                 filters, 
                 expansion   = 1.0,
                 shortcut    = True,
                 use_reparam = False,
                 activation  ='relu', 
                 normalizer  ='batch-norm',
                 **kwargs):
        super(CIB, self).__init__(**kwargs)
        self.filters     = filters
        self.expansion   = expansion
        self.shortcut    = shortcut  
        self.use_reparam = use_reparam     
        self.activation  = activation
        self.normalizer  = normalizer

    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        if self.use_reparam:
            middle = RepVGGDW(2 * hidden_dim, activation=self.normalizer, normalizer=self.normalizer)
        else:
            middle = ConvolutionBlock(2 * hidden_dim, (3, 3), groups=2 * hidden_dim, activation=self.normalizer, normalizer=self.normalizer)

        self.block = Sequential([
            ConvolutionBlock(self.c, (3, 3), groups=self.c, activation=self.activation, normalizer=self.normalizer),
            ConvolutionBlock(2 * hidden_dim, (1, 1), activation=self.activation, normalizer=self.normalizer),
            middle,
            ConvolutionBlock(self.filters, (1, 1), activation=self.activation, normalizer=self.normalizer),
            ConvolutionBlock(self.filters, (3, 3), groups=self.filters, activation=self.activation, normalizer=self.normalizer)
        ])

    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x


class C2fCIB(C2f):

    def __init__(self,
                 filters, 
                 iters,
                 expansion   = 0.5,
                 shortcut    = True,
                 use_reparam = False,
                 activation  = 'silu', 
                 normalizer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         iters,
                         expansion,
                         shortcut,
                         activation,
                         normalizer,
                         *args,
                         **kwargs)
        self.use_reparam = use_reparam

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.blocks = [CIB(hidden_dim, shortcut=self.shortcut, use_reparam=self.use_reparam, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)]


class SimpleAttention(tf.keras.layers.Layer):

    def __init__(self, 
                 dim,
                 num_heads  = 8,
                 attn_ratio = 0.5,
                 activation = None, 
                 normalizer = 'batch-norm',
                 **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        self.dim        = dim
        self.num_heads  = num_heads
        self.attn_ratio = attn_ratio
        self.activation = activation
        self.normalizer = normalizer
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        self.h = dim + nh_kd * 2

    def build(self, input_shape):
        self.qkv = ConvolutionBlock(self.h, (1, 1), activation=self.activation, normalizer=self.normalizer)
        self.proj = ConvolutionBlock(self.dim, (1, 1), activation=self.activation, normalizer=self.normalizer)
        self.pe  = ConvolutionBlock(self.dim, (3, 3), groups=self.dim, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        _, H, W, C = inputs.shape
        N = H * W

        x = self.qkv(inputs, training=training)
        x = tf.reshape(x, shape=[-1, N, self.num_heads, self.key_dim*2 + self.head_dim])
        q, k, v = tf.split(x, num_or_size_splits=[self.key_dim, self.key_dim, self.head_dim], axis=-1)
        attn = tf.transpose(q, perm=[0, 2, 1, 3]) @ tf.transpose(k, perm=[0, 2, 3, 1])
        attn *= self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.transpose(v, perm=[0, 2, 3, 1]) @ tf.transpose(attn, perm=[0, 1, 2, 3])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, H, W, C])
        x += self.pe(tf.reshape(v, shape=[-1, H, W, C]))
        x = self.proj(x, training=training)
        return x


class PSA(tf.keras.layers.Layer):

    def __init__(self, 
                 filters, 
                 expansion   = 0.5,
                 activation  ='silu', 
                 normalizer  ='batch-norm',
                 **kwargs):
        super(PSA, self).__init__(**kwargs)
        self.filters     = filters
        self.expansion   = expansion
        self.activation  = activation
        self.normalizer  = normalizer

    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.c * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim * 2, (1, 1), activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.c, (1, 1), activation=self.activation, normalizer=self.normalizer)
        self.attn = SimpleAttention(hidden_dim, num_heads=hidden_dim // 64, attn_ratio=0.5)
        self.ffn = Sequential([
            ConvolutionBlock(hidden_dim * 2, (1, 1), activation=self.activation, normalizer=self.normalizer),
            ConvolutionBlock(hidden_dim, (1, 1), activation=None, normalizer=self.normalizer)
        ])

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        a, b = tf.split(x, num_or_size_splits=2, axis=-1)

        b += self.attn(b, training=training)
        b += self.ffn(b, training=training)        

        x = concatenate([a, b], axis=-1)
        x = self.conv2(x, training=training)
        return x


def DarkNetCIB_A(cib_block,
                 spp_block,
                 layers,
                 filters,
                 scale_ratio=1,
                 use_reparam=False,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 activation='silu',
                 normalizer='batch-norm',
                 final_activation="softmax",
                 classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=640,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    l0, l1, l2, l3 = layers

    if isinstance(cib_block, (list, tuple)):
        cib_block1, cib_block2 = cib_block
    else:
        cib_block1 = cib_block2 = cib_block

    img_input = Input(shape=input_shape)
    
    x = ConvolutionBlock(filters, 3, downsample=True, activation=activation, normalizer=normalizer, name='stem')(img_input)
    
    x = ConvolutionBlock(filters * 2, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage1.block1')(x)
    x = cib_block1(filters * 2, l0, activation=activation, normalizer=normalizer, name='stage1.block2')(x)

    x = ConvolutionBlock(filters * 4, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage2.block1')(x)
    x = cib_block1(filters * 4, l1, activation=activation, normalizer=normalizer, name='stage2.block2')(x)
    
    x = SCDown(filters * 8, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage3.block1')(x)
    x = cib_block1(filters * 8, l2, activation=activation, normalizer=normalizer, name='stage3.block2')(x)

    x = SCDown(int(filters * 16 * scale_ratio), 3, downsample=True, activation=activation, normalizer=normalizer, name='stage4.block1')(x)
    
    if cib_block2 == C2fCIB:
        x = cib_block2(int(filters * 16 * scale_ratio), l3, use_reparam=use_reparam, activation=activation, normalizer=normalizer, name='stage4.block2')(x)
    else:
        x = cib_block2(int(filters * 16 * scale_ratio), l3, activation=activation, normalizer=normalizer, name='stage4.block2')(x)
    
    x = spp_block(int(filters * 16 * scale_ratio), name='stage4.block3')(x)

    x = PSA(int(filters * 16 * scale_ratio), activation=activation, normalizer=normalizer, name='stage4.block4')(x)
                                                                   

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
            
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if layers == [1, 2, 2, 1] and filters == 16:
        model = Model(inputs, x, name='DarkNet-CIB-Nano')
    elif layers == [1, 2, 2, 1] and filters == 32:
        model = Model(inputs, x, name='DarkNet-CIB-Small')
    elif layers == [2, 4, 4, 2] and filters == 48:
        model = Model(inputs, x, name='DarkNet-CIB-Medium')
    elif layers == [2, 4, 4, 2] and filters == 64:
        model = Model(inputs, x, name='DarkNet-CIB-Base')
    elif layers == [3, 6, 6, 3] and filters == 64:
        model = Model(inputs, x, name='DarkNet-CIB-Large')
    else:
        model = Model(inputs, x, name='DarkNet-CIB-A')

    if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image data format convention '
                      '(`image_data_format="channels_first"`). '
                      'For best performance, set '
                      '`image_data_format="channels_last"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
    return model


def DarkNetCIB_B(cib_block,
                 spp_block,
                 layers,
                 filters,
                 scale_ratio=1,
                 use_reparam=False,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 activation='silu',
                 normalizer='batch-norm',
                 final_activation="softmax",
                 classes=1000):

    if weights not in {'imagenet', None}:
      raise ValueError('The `weights` argument should be either '
                        '`None` (random initialization) or `imagenet` '
                        '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=640,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    l0, l1, l2, l3 = layers

    if isinstance(cib_block, (list, tuple)):
        cib_block1, cib_block2 = cib_block
    else:
        cib_block1 = cib_block2 = cib_block

    img_input = Input(shape=input_shape)
    
    x = ConvolutionBlock(filters, 3, downsample=True, activation=activation, normalizer=normalizer, name='stem')(img_input)
    
    x = ConvolutionBlock(filters * 2, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage1.block1')(x)
    x = cib_block1(filters * 2, l0, activation=activation, normalizer=normalizer, name='stage1.block2')(x)

    x = ConvolutionBlock(filters * 4, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage2.block1')(x)
    x = cib_block1(filters * 4, l1, activation=activation, normalizer=normalizer, name='stage2.block2')(x)
    
    x = SCDown(filters * 8, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage3.block1')(x)
    
    if cib_block2 == C2fCIB:
        x = cib_block2(filters * 8, l2, use_reparam=use_reparam, activation=activation, normalizer=normalizer, name='stage3.block2')(x)
    else:
        x = cib_block2(filters * 8, l2, activation=activation, normalizer=normalizer, name='stage3.block2')(x)
    
    x = SCDown(int(filters * 16 * scale_ratio), 3, downsample=True, activation=activation, normalizer=normalizer, name='stage4.block1')(x)

    if cib_block2 == C2fCIB:
        x = cib_block2(int(filters * 16 * scale_ratio), l3, use_reparam=use_reparam, activation=activation, normalizer=normalizer, name='stage4.block2')(x)
    else:
        x = cib_block2(int(filters * 16 * scale_ratio), l3, activation=activation, normalizer=normalizer, name='stage4.block2')(x)
    
    x = spp_block(int(filters * 16 * scale_ratio), name='stage4.block3')(x)
    x = PSA(int(filters * 16 * scale_ratio), activation=activation, normalizer=normalizer, name='stage4.block4')(x)
                                                                   
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
            
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if layers == [3, 6, 6, 3] and filters == 80:
        model = Model(inputs, x, name='DarkNet-CIB-XLarge')
    else:
        model = Model(inputs, x, name='DarkNet-CIB-B')

    if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image data format convention '
                      '(`image_data_format="channels_first"`). '
                      'For best performance, set '
                      '`image_data_format="channels_last"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
    return model


def DarkNetCIB_nano(cib_block=C2f,
                    spp_block=SPPF,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    normalizer='batch-norm',
                    final_activation="softmax",
                    classes=1000) -> Model:

    model = DarkNetCIB_A(cib_block=cib_block,
                         spp_block=spp_block,
                         layers=[1, 2, 2, 1],
                         filters=16,
                         scale_ratio=1,
                         use_reparam=False,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes)

    return model

def DarkNetCIB_nano_backbone(cib_block=C2f,
                             spp_block=SPP,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             normalizer='batch-norm',
                             custom_layers=None) -> Model:
    
    """
        - Used in YOLOv10 version nano
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10n.yaml
    """
    
    model = DarkNetCIB_nano(cib_block=cib_block,
                            spp_block=spp_block,
                            include_top=include_top, 
                            weights=weights,
                            activation=activation,
                            normalizer=normalizer,
                            input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block4").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetCIB_small(cib_block=[C2f, C2fCIB],
                     spp_block=SPPF,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     normalizer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:

    model = DarkNetCIB_A(cib_block=cib_block,
                         spp_block=spp_block,
                         layers=[1, 2, 2, 1],
                         filters=32,
                         scale_ratio=1,
                         use_reparam=True,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes)

    return model


def DarkNetCIB_small_backbone(cib_block=[C2f, C2fCIB],
                              spp_block=SPP,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              normalizer='batch-norm',
                              custom_layers=None) -> Model:
    
    """
        - Used in YOLOv10 version small
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10s.yaml
    """
    
    model = DarkNetCIB_small(cib_block=cib_block,
                             spp_block=spp_block,
                             include_top=include_top, 
                             weights=weights,
                             activation=activation,
                             normalizer=normalizer,
                             input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block4").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetCIB_medium(cib_block=[C2f, C2fCIB],
                      spp_block=SPPF,
                      include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      activation='silu',
                      normalizer='batch-norm',
                      final_activation="softmax",
                      classes=1000) -> Model:

    model = DarkNetCIB_A(cib_block=cib_block,
                         spp_block=spp_block,
                         layers=[2, 4, 4, 2],
                         filters=48,
                         scale_ratio=0.75,
                         use_reparam=False,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes)

    return model


def DarkNetCIB_medium_backbone(cib_block=[C2f, C2fCIB],
                               spp_block=SPP,
                               input_shape=(640, 640, 3),
                               include_top=False, 
                               weights='imagenet', 
                               activation='silu',
                               normalizer='batch-norm',
                               custom_layers=None) -> Model:
    
    """
        - Used in YOLOv10 version medium
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10m.yaml
    """
    
    model = DarkNetCIB_medium(cib_block=cib_block,
                              spp_block=spp_block,
                              include_top=include_top, 
                              weights=weights,
                              activation=activation,
                              normalizer=normalizer,
                              input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block4").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetCIB_base(cib_block=[C2f, C2fCIB],
                    spp_block=SPPF,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    normalizer='batch-norm',
                    final_activation="softmax",
                    classes=1000) -> Model:

    model = DarkNetCIB_A(cib_block=cib_block,
                         spp_block=spp_block,
                         layers=[2, 4, 4, 2],
                         filters=64,
                         scale_ratio=0.5,
                         use_reparam=False,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes)

    return model


def DarkNetCIB_base_backbone(cib_block=[C2f, C2fCIB],
                             spp_block=SPP,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             normalizer='batch-norm',
                             custom_layers=None) -> Model:
    
    """
        - Used in YOLOv10 version base
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10b.yaml
    """
    
    model = DarkNetCIB_base(cib_block=cib_block,
                            spp_block=spp_block,
                            include_top=include_top, 
                            weights=weights,
                            activation=activation,
                            normalizer=normalizer,
                            input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block4").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetCIB_large(cib_block=[C2f, C2fCIB],
                     spp_block=SPPF,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     normalizer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:

    model = DarkNetCIB_A(cib_block=cib_block,
                         spp_block=spp_block,
                         layers=[3, 6, 6, 3],
                         filters=64,
                         scale_ratio=0.5,
                         use_reparam=False,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes)

    return model


def DarkNetCIB_large_backbone(cib_block=[C2f, C2fCIB],
                              spp_block=SPP,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              normalizer='batch-norm',
                              custom_layers=None) -> Model:
    
    """
        - Used in YOLOv10 version large
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10l.yaml
    """
    
    model = DarkNetCIB_large(cib_block=cib_block,
                             spp_block=spp_block,
                             include_top=include_top, 
                             weights=weights,
                             activation=activation,
                             normalizer=normalizer,
                             input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block4").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetCIB_xlarge(cib_block=[C2f, C2fCIB],
                      spp_block=SPPF,
                      include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      activation='silu',
                      normalizer='batch-norm',
                      final_activation="softmax",
                      classes=1000) -> Model:

    model = DarkNetCIB_B(cib_block=cib_block,
                         spp_block=spp_block,
                         layers=[3, 6, 6, 3],
                         filters=80,
                         scale_ratio=0.5,
                         use_reparam=False,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes)

    return model


def DarkNetCIB_xlarge_backbone(cib_block=[C2f, C2fCIB],
                               spp_block=SPP,
                               input_shape=(640, 640, 3),
                               include_top=False, 
                               weights='imagenet', 
                               activation='silu',
                               normalizer='batch-norm',
                               custom_layers=None) -> Model:
    
    """
        - Used in YOLOv10 version xlarge
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10x.yaml
    """
    
    model = DarkNetCIB_xlarge(cib_block=cib_block,
                              spp_block=spp_block,
                              include_top=include_top, 
                              weights=weights,
                              activation=activation,
                              normalizer=normalizer,
                              input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block4").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')
