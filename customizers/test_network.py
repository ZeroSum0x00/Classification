from tensorflow.keras.applications import ConvNeXtTiny


def TestNet(inputs=(224, 224, 3), include_head=True, weights=None, num_classes=1000):
    print(num_classes)
    return ConvNeXtTiny(input_shape=inputs, include_top=include_head, weights=weights, classes=num_classes)
