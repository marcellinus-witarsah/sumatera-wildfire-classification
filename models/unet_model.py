import tensorflow as tf
import math

def double_conv_block(x, n_filters):
    """
    :param x: previous tf keras layer
    :param n_filters: amount of filters
    :return: two consecutive Conv2D layer
    """
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation=tf.nn.relu,
                               padding='same',kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation=tf.nn.relu,
                               padding='same', kernel_initializer='he_normal')(x)
    return x
    
def downsample_conv_block(x, n_filters):
    """
    :param x: previous tf keras layer
    :param n_filters: amount of filters
    :return: two consecutive Conv2D layer followed by MaxPooling2D and Dropout layer
    """
    c = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c)
    p = tf.keras.layers.Dropout(0.3)(p)
    return c, p


def upsample_conv_block(x, conv_features, n_filters):
    """
    :param x: previous tf keras layer
    :param conv_features: previous Conv2D layer on the contracting path
    :param n_filters: amount of filters
    :return: Conv2DTranspose (upsampling) layer followed by concatenate layer 
             (Conv2DTranspose and Conv2D on the contracting path) and Dropout layer
    """
    x = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                        kernel_size=(3, 3),
                                        strides=(2, 2),
                                        padding='same',)(x)
    x = tf.keras.layers.concatenate([x, conv_features])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

    
def get_model(input_shape, starting_filter, base=2):
    """
    :param input_shape: input shape of the model (width, height, channels)
    :param starting_filter: starting filter for convolution layer to be doubled
    :param base: base of power equation (later will be multiplied to the starting filter)  
    :return: U-Net model
    """
    # input to the model     
    inputs = tf.keras.Input(shape=input_shape)
    
    # encoder: contracting path - downsample
    # downsample conv block - 1
    c1, p1 = downsample_conv_block(inputs, n_filters=int(starting_filter*math.pow(base,0)))
    # downsample conv block - 2
    c2, p2 = downsample_conv_block(p1, n_filters=int(starting_filter*math.pow(base,1)))
    # downsample conv block - 3
    c3, p3 = downsample_conv_block(p2, n_filters=int(starting_filter*math.pow(base,2)))
    # downsample conv block - 4
    c4, p4 = downsample_conv_block(p3, n_filters=int(starting_filter*math.pow(base,3)))
    
    # downsample conv block - 5 (Bottleneck)
    bottleneck = double_conv_block(p4, n_filters=int(starting_filter*math.pow(base,4)))
    
    # decoder: symmetric expanding path - upsample
    # upsample conv block - 1
    u1 = upsample_conv_block(bottleneck, c4, n_filters=int(starting_filter*math.pow(base,3)))
    # upsample conv block - 2
    u2 = upsample_conv_block(u1, c3, n_filters=int(starting_filter*math.pow(base,2)))
    # upsample conv block - 3
    u3 = upsample_conv_block(u2, c2, n_filters=int(starting_filter*math.pow(base,1)))
    # upsample conv block - 4
    u4 = upsample_conv_block(u3, c1, n_filters=int(starting_filter*math.pow(base,0)))
    
    # output
    outputs = tf.keras.layers.Conv2D(filters=1,
                                     kernel_size=(1, 1),
                                     activation='sigmoid')(u4)
    
    unet_model = tf.keras.Model(inputs, outputs, name='unet-model')
    return unet_model