# models.py

from keras import layers, models

def build_cnn(input_shape, num_classes):
    """
    Builds and returns a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape: tuple, the shape of the input data (timesteps, features)
    - num_classes: int, the number of output classes

    Returns:
    - model: Keras Model object
    """
    model = models.Sequential()
    # Block 1
    model.add(layers.Conv1D(filters=32,
                            kernel_size=3,
                            input_shape=input_shape,
                            activation='relu',
                            padding='same'))
    model.add(layers.Conv1D(filters=32,
                            kernel_size=3,
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling1D(pool_size=3))
    model.add(layers.Dropout(0.1))

    # Block 2
    model.add(layers.Conv1D(filters=64,
                            kernel_size=3,
                            activation='relu',
                            padding='same'))
    model.add(layers.Conv1D(filters=64,
                            kernel_size=3,
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.1))

    # Block 3
    model.add(layers.Conv1D(filters=128,
                            kernel_size=3,
                            activation='relu',
                            padding='same'))
    model.add(layers.Conv1D(filters=128,
                            kernel_size=3,
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling1D(pool_size=5))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def residual_block(x, filters, kernel_size=3, stride=1):
    """
    A standard residual block for 1D ResNet.

    Parameters:
    - x: input tensor
    - filters: int, number of filters for the convolutional layers
    - kernel_size: int, size of the convolutional kernel
    - stride: int, stride for the convolution

    Returns:
    - output tensor for the block
    """
    shortcut = x

    # First convolutional layer
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second convolutional layer
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adjust the shortcut if necessary
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the shortcut to the main path
    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)

    return x

def build_resnet1d(input_shape, num_classes):
    """
    Builds and returns a 1D ResNet model.

    Parameters:
    - input_shape: tuple, the shape of the input data (timesteps, features)
    - num_classes: int, the number of output classes

    Returns:
    - model: Keras Model object
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)

    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=128)

    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=256)

    x = residual_block(x, filters=512, stride=2)
    x = residual_block(x, filters=512)

    # Global average pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model