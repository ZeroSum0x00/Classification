import inspect
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from .logger import logger



def is_valid_image_tensor(input_obj):
    # Lấy shape nếu là tensor, ngược lại coi như tuple
    try:
        shape = tuple(input_obj.shape)
    except AttributeError:
        try:
            shape = tuple(input_obj)
        except:
            return False  # Không phải dạng shape hợp lệ

    # Định nghĩa các shape hợp lệ
    valid_shapes = [
        (lambda s: len(s) == 3 and s[2] == 3),     # (h, w, 3)
        (lambda s: len(s) == 4 and s[3] == 3),     # (None, h, w, 3)
        (lambda s: len(s) == 3 and s[0] == 3),     # (3, h, w)
        (lambda s: len(s) == 4 and s[1] == 3),     # (None, 3, h, w)
    ]

    return any(check(shape) for check in valid_shapes)

    
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

    Internal utility to compute/validate a model"s input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
            If weights="imagenet" input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    no_valid_size = None
    if isinstance(default_size, int):
        valid_sizes = no_valid_size = [default_size, default_size]
    elif isinstance(default_size, (list, tuple)):
        temp_size = []
        for size in default_size:
            if isinstance(size, int):
                if not no_valid_size:
                    no_valid_size = [size, size]
                temp_size.append([size, size])
            elif isinstance(size, (list, tuple)):
                if not no_valid_size:
                    no_valid_size = size
                temp_size.append(size)
    
        valid_sizes = temp_size
    else:
        raise ValueError("`default_size` must be an int or a list/tuple of ints.")

    if is_valid_image_tensor(input_shape):
        if weights != "imagenet" and len(input_shape) == 3:
            if data_format == "channels_first":
                if input_shape[0] not in {1, 3}:
                    logger.warning(
                        "This model usually expects 1 or 3 input channels. "
                        "However, it was passed an input_shape with " +
                        str(input_shape[0]) + " input channels."
                    )
            else:
                if input_shape[-1] not in {1, 3}:
                    logger.warning(
                        "This model usually expects 1 or 3 input channels. "
                        "However, it was passed an input_shape with " +
                        str(input_shape[-1]) + " input channels."
                    )
        default_shape = input_shape
    else:
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            if data_format == "channels_first":
                default_shape = (3, *input_shape)
            else:
                default_shape = (*input_shape, 3)
        else:
            logger.warning(
                "Invalid `input_shape` provided. Expected a 2D shape (height, width) or a 3D shape "
                "(channels, height, width) or (height, width, channels) depending on `data_format`. "
                "Received: {}. Defaulting to fallback shape.".format(input_shape)
            )

            if data_format == "channels_first":
                default_shape = (3, *no_valid_size)
            else:
                default_shape = (*no_valid_size, 3)

    check_valid_shape = False
    if isinstance(valid_sizes[0], list):
        for valid_size in valid_sizes:
            if weights == "imagenet" and require_flatten:
                if is_valid_image_tensor(input_shape):
                    if data_format == "channels_first":
                        if tuple(input_shape) == (input_shape[-1], *valid_size):
                            check_valid_shape = True
                            break
                    else:
                        if tuple(input_shape) == (*valid_size, input_shape[-1]):
                            check_valid_shape = True
                            break
            else:
                check_valid_shape = True
    else:
        if weights == "imagenet" and require_flatten:
            if is_valid_image_tensor(input_shape):
                if data_format == "channels_first":
                    if tuple(input_shape) == (input_shape[-1], *valid_sizes):
                        check_valid_shape = True
                else:
                    if tuple(input_shape) == (*valid_sizes, input_shape[-1]):
                        check_valid_shape = True
        else:
            check_valid_shape = True
            
    if not check_valid_shape:
        raise ValueError("When setting `include_head=True` "
                         "and loading `imagenet` weights, "
                         "`inputs` should be in " +
                         str(default_size) + ".")
    else:
        return default_shape
    
    if input_shape:
        if data_format == "channels_first":
            if len(input_shape) != 3:
                raise ValueError("`inputs` must be a tuple of three integers.")
            if input_shape[0] != 3 and weights == "imagenet":
                raise ValueError("The input must have 3 channels; got `input_shape=" +
                                 str(input_shape) + "`")
            if ((input_shape[1] is not None and input_shape[1] < min_size) or
                (input_shape[2] is not None and input_shape[2] < min_size)):
                raise ValueError("Input size must be at least " +
                                 str(min_size) + "x" + str(min_size) +
                                 "; got `input_shape=" + str(input_shape) + "`")
        else:
            if len(input_shape) != 3:
                raise ValueError("`inputs` must be a tuple of three integers.")
            if input_shape[-1] != 3 and weights == "imagenet":
                raise ValueError("The input must have 3 channels; got `input_shape=" +
                                 str(input_shape) + "`")
            if ((input_shape[0] is not None and input_shape[0] < min_size) or
                (input_shape[1] is not None and input_shape[1] < min_size)):
                raise ValueError("Input size must be at least " +
                                 str(min_size) + "x" + str(min_size) +
                                 "; got `input_shape=" + str(input_shape) + "`")
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == "channels_first":
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)

    if require_flatten:
        if None in input_shape:
            raise ValueError("If `include_head` is True, "
                             "you should specify a static `inputs`. "
                             "Got `input_shape=" + str(input_shape) + "`")
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

    strides = kwargs.get("strides", None)
    pool_size = kwargs.get("pool_size", None)
    if strides == (2, 2) and pool_size == (2, 2) and "strides" in valid_params and "pool_size" in valid_params:
        kwargs["pool_size"] = (1, 1)

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
    img_dim = 2 if K.image_data_format() == "channels_first" else 1
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


def check_regularizer(value):
    if isinstance(value, (int, float)) and value > 0:
        return l2(value)
    elif isinstance(value, l2):
        return value
    else:
        return None

def validate_conv_arg(value):
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return tuple(value)
    else:
        raise ValueError(f"value must be an int or a tuple/list of 2 ints, got: {value}")


def create_model_backbone(model_fn, custom_layers=None, *args, **kwargs):
    model = model_fn(include_head=False, *args, **kwargs)

    custom_layers = custom_layers or []
    output_custom_layer = []
    for layer in custom_layers:
        if not model.get_layer(layer):
            raise ValueError(f"Layer '{layer}' not found in model.")

        output_custom_layer.append(model.get_layer(layer).output)
        
    final_output_layer = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*output_custom_layer, final_output_layer], name=f"{model.name}_backbone")
