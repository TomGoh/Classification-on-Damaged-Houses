import tensorflow as tf
import numpy as np



def Build_Layer(input_channel, channel, block_num, strides=1):
    downsample = None
    if strides != 1 or input_channel != channel:
        downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=strides, use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   bias_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        ])

    layers_list = []
    layers_list.append(Residual_Block(channel, downsample=downsample, strides=strides))
    for i in range(1, block_num):
        layers_list.append(Residual_Block(channel))
    return tf.keras.Sequential(layers_list)


def build_ResNet18():
    inputs = tf.keras.layers.Input(shape=(128, 128, 3), dtype='float32')
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=2, padding='same', use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               bias_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = Build_Layer(input_channel=x.shape[-1], channel=32, block_num=1)(x)
    x = Build_Layer(input_channel=x.shape[-1], channel=64, block_num=1, strides=2)(x)
    x = Build_Layer(input_channel=x.shape[-1], channel=128, block_num=1, strides=2)(x)
    x = Build_Layer(input_channel=x.shape[-1], channel=256, block_num=1, strides=2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(x)

    ResNet18 = tf.keras.Model(inputs=inputs, outputs=outputs)
    return ResNet18

class Residual_Block(tf.keras.layers.Layer):

    def __init__(self, output_channel, strides=1, downsample=None, **kwargs):
        super(Residual_Block, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(output_channel, kernel_size=(3, 3), strides=strides, padding='same',
                                            use_bias=False)
        self.conv2 = tf.keras.layers.Conv2D(output_channel, kernel_size=(3, 3), strides=1, padding='same',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.downsample = downsample
        self.relu = tf.keras.layers.ReLU()
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.add([identity, x])
        x = self.relu(x)

        return x