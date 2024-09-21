# models.py

from keras import layers, models
from keras.regularizers import l2

def build_cnn(input_shape, num_classes, filters, kernel_sizes, dropout_rates, l2_reg=0.001):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for f, k, d in zip(filters, kernel_sizes, dropout_rates):
        x = keras.layers.Conv1D(filters=f, kernel_size=k, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(d)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def conv_block(x, filters, kernel_size=3, stride=1, l2_reg=0.001):
    """
    Basic residual block for ResNet18 and ResNet34.

    Parameters:
    - x (tensor): Input tensor.
    - filters (int): Number of filters.
    - kernel_size (int): Kernel size for Conv layers.
    - stride (int): Stride for the first Conv layer.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - tensor: Output tensor after applying the residual block.
    """
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                 kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def bottleneck_block(x, filters, kernel_size=3, stride=1, l2_reg=0.001):
    """
    Bottleneck residual block for ResNet50.

    Parameters:
    - x (tensor): Input tensor.
    - filters (int): Number of filters for the first two Conv layers.
    - kernel_size (int): Kernel size for the middle Conv layer.
    - stride (int): Stride for the first Conv layer.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - tensor: Output tensor after applying the bottleneck block.
    """
    shortcut = x

    # 1x1 Conv
    x = layers.Conv1D(filters, 1, strides=stride, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 3x3 Conv
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 1x1 Conv
    x = layers.Conv1D(filters * 4, 1, strides=1, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters * 4:
        shortcut = layers.Conv1D(filters * 4, 1, strides=stride, padding='same',
                                 kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_resnet18_1d(input_shape, num_classes, l2_reg=0.001):
    """
    Builds ResNet18 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - model (keras.Model): Compiled ResNet18 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                      kernel_regularizer=l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(2):
        x = conv_block(x, filters=64, l2_reg=l2_reg)

    for _ in range(2):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=128, stride=stride, l2_reg=l2_reg)

    for _ in range(2):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=256, stride=stride, l2_reg=l2_reg)

    for _ in range(2):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=512, stride=stride, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

def build_resnet34_1d(input_shape, num_classes, l2_reg=0.001):
    """
    Builds ResNet34 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - model (keras.Model): Compiled ResNet34 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                      kernel_regularizer=l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(3):
        x = conv_block(x, filters=64, l2_reg=l2_reg)

    for _ in range(4):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=128, stride=stride, l2_reg=l2_reg)

    for _ in range(6):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=256, stride=stride, l2_reg=l2_reg)

    for _ in range(3):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=512, stride=stride, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

def build_resnet50_1d(input_shape, num_classes, l2_reg=0.001):
    """
    Builds ResNet50 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - model (keras.Model): Compiled ResNet50 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                      kernel_regularizer=l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(3):
        x = bottleneck_block(x, filters=64, l2_reg=l2_reg)

    for _ in range(4):
        stride = 2 if _ == 0 else 1
        x = bottleneck_block(x, filters=128, stride=stride, l2_reg=l2_reg)

    for _ in range(6):
        stride = 2 if _ == 0 else 1
        x = bottleneck_block(x, filters=256, stride=stride, l2_reg=l2_reg)

    for _ in range(3):
        stride = 2 if _ == 0 else 1
        x = bottleneck_block(x, filters=512, stride=stride, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model
