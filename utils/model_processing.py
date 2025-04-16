import inspect
import warnings
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.backend import is_keras_tensor


def _obtain_input_shape(
    input_shape,
    default_size,
    min_size,
    data_format,
    require_flatten,
    weights=None,
):
    """
    Taken from:
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py

    Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if tuple(input_shape) != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                        (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                        (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def process_model_input(
    input_data,
    include_head=True,
    default_size=224,
    min_size=32,
    weights=None,
):
    if isinstance(input_data, (list, tuple)):
        if len(input_data) == 2:
            input_data = (*input_data, 3)

        input_shape = _obtain_input_shape(
            input_shape=input_data,
            default_size=default_size,
            min_size=min_size,
            data_format=K.image_data_format(),
            require_flatten=include_head,
            weights=weights
        )
        return Input(shape=input_shape)

    elif K.is_keras_tensor(input_data):
        return Input(tensor=input_data)

    elif isinstance(input_data, tf.Tensor):
        return input_data

    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")


def create_layer_instance(block, *args, **kwargs):
    layer_name = kwargs.pop("name", None)

    if inspect.isclass(block):
        valid_params = inspect.signature(block.__init__).parameters
    elif inspect.isfunction(block):
        valid_params = inspect.signature(block).parameters
    else:
        raise ValueError("block must be either a class or a function")
        
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return block(*args, **filtered_kwargs, name=layer_name)


def compute_padding(kernel_size, dilation_rate):
    k = np.array(kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size])
    d = np.array(dilation_rate if isinstance(dilation_rate, (list, tuple)) else [dilation_rate, dilation_rate])
    padding = ((k - 1) * d) // 2
    return int(padding[0]), int(padding[1])


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def drop_connect_rates_split(num_blocks, start=0.0, end=0.0):
    """split drop connect rate in range `(start, end)` according to `num_blocks`"""
    cum_split = [sum(num_blocks[: id + 1]) for id, _ in enumerate(num_blocks[:-1])]
    drop_connect_rates = np.split(np.linspace(start, end, sum(num_blocks)), cum_split)
    return [ii.tolist() for ii in drop_connect_rates]