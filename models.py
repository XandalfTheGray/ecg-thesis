# models.py
from keras import layers, models
from keras.regularizers import l2
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization

def build_cnn(input_shape, num_classes, filters, kernel_sizes, dropout_rates, l2_reg=0.001, activation='softmax'):
    """
    Builds a Convolutional Neural Network (CNN) with two Conv1D layers per block and MaxPooling.
    """
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(len(filters)):
        # First Conv1D layer
        x = Conv1D(filters=filters[i],
                   kernel_size=kernel_sizes[i],
                   activation='relu',
                   padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        # Second Conv1D layer
        x = Conv1D(filters=filters[i],
                   kernel_size=kernel_sizes[i],
                   activation='relu',
                   padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        # MaxPooling
        pool_size = 3 if filters[i] == 32 else 2 if filters[i] == 64 else 5
        x = MaxPooling1D(pool_size=pool_size)(x)
        # Dropout
        x = Dropout(dropout_rates[i])(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rates[-1])(x)
    outputs = Dense(num_classes, activation=activation)(x)
    
    model = models.Model(inputs, outputs)
    return model

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

def build_resnet18_1d(input_shape, num_classes, l2_reg=0.001, activation='softmax'):
    """
    Builds ResNet18 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.
    - activation (str): Activation function for the output layer ('softmax' or 'sigmoid').

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
    outputs = layers.Dense(num_classes, activation=activation)(x)
    model = models.Model(inputs, outputs)
    return model

def build_resnet34_1d(input_shape, num_classes, l2_reg=0.001, activation='softmax'):
    """
    Builds ResNet34 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.
    - activation (str): Activation function for the output layer ('softmax' or 'sigmoid').

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
    outputs = layers.Dense(num_classes, activation=activation)(x)
    model = models.Model(inputs, outputs)
    return model

def build_resnet50_1d(input_shape, num_classes, l2_reg=0.001, activation='softmax'):
    """
    Builds ResNet50 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.
    - activation (str): Activation function for the output layer ('softmax' or 'sigmoid').

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
    outputs = layers.Dense(num_classes, activation=activation)(x)
    model = models.Model(inputs, outputs)
    return model
