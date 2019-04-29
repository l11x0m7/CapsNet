# -*- encoding:utf -*-

# Refer to XifengGuo's CapsNet-Keras(https://github.com/XifengGuo/CapsNet-Keras)

from __future__ import print_function

import keras
import keras.backend as K
from keras import layers
from keras import utils
from keras.utils import conv_utils
from keras import initializers
from keras.layers import Input, Conv2D
from keras.models import Model, Sequential
import os
import numpy as np


def softmax(x, axis=-1):
    """
    Self-defined softmax function
    """
    x = K.exp(x - K.max(x, axis=axis, keepdims=True))
    x /= K.sum(x, axis=axis, keepdims=True)
    return x

def margin_loss(y, pred):
    """
    For the first part of loss(classification loss)
    """
    return K.mean(K.sum(y * K.square(K.maximum(0.9 - pred, 0)) + \
        0.5 *  K.square((1 - y) * K.maximum(pred - 0.1, 0)), axis=1))


def squash(s, axis=-1):
    """
    Squash function. This could be viewed as one kind of activations.
    """
    squared_s = K.sum(K.square(s), axis=axis, keepdims=True)
    scale = squared_s / (1 + squared_s) / K.sqrt(squared_s + K.epsilon())
    return scale * s

class Length(layers.Layer):
    """
    Compute the lengths of capsules. 
    The values could be viewed as probability.
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), axis=-1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


class Mask(layers.Layer):
    """
    Mask for the true answer or the predicted answer.
    """
    def call(self, inputs, **kwargs):
        # inputs -> (X, y), then output the mask of y
        # inputs -> X, then output the mask of prediction
        if type(inputs) is list or tuple:
            inputs, mask = inputs
        else:
            pred = K.sqrt(K.sum(K.square(inputs), axis=-1) + K.epsilon())
            mask = K.one_hot(indices=K.argmax(pred, 1), num_classes=pred.get_shape().as_list()[1])
        return K.batch_flatten(inputs * K.expand_dims(mask, axis=-1))

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is list or tuple:
            return (input_shape[0][0], input_shape[0][1] * input_shape[0][2])
        else:
            return (input_shape[0], input_shape[1] * input_shape[2])



class DigiCaps(layers.Layer):
    """
    Compute the operations between two layers of capsules.
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3, 
        kernel_initializer='glorot_uniform', name='digitcaps', **kwargs):
        super(DigiCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                    self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W_cap')

    def call(self, inputs, **kwargs):
        # (batch_size, 1, input_num_capsule, input_dim_capsule)
        expand_inputs = K.expand_dims(inputs, axis=1)
        # (batch_size, num_capsule, input_num_capsule, input_dim_capsule)
        expand_inputs = K.tile(expand_inputs, (1, self.num_capsule, 1, 1))
        # (batch_size, num_capsule, input_num_capsule, dim_capsule)
        u_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, axes=[2, 3]), expand_inputs)

        if self.num_routing <= 0:
            self.num_routing = 3
        # (batch_size, num_capsule, input_num_capsule)
        b = K.zeros((K.shape(u_hat)[0], self.num_capsule, self.input_num_capsule))
        for i in xrange(self.num_routing):
            # (batch_size, num_capsule, input_num_capsule)
            c = softmax(b, axis=1)
            # (batch_size, num_capsule, dim_capsule)
            s = K.batch_dot(c, u_hat, axes=[2, 2])
            squashed_s = squash(s)
            if i < self.num_routing - 1:
                # (batch_size, num_capsule, input_num_capsule)
                b += K.batch_dot(squashed_s, u_hat, axes=[2, 3])
        return squashed_s

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class PrimaryCapsules(layers.Layer):
    """
    Convert the input into the capsule format.
    """
    def __init__(self, filters, kernel_size, dim_capsule, padding='valid', strides=(1, 1), **kwargs):
        super(PrimaryCapsules, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        if type(self.kernel_size) is int:
            self.kernel_size = (self.kernel_size, self.kernel_size)
        self.dim_capsule = dim_capsule
        self.padding = padding
        self.strides = strides


    def build(self, input_shape):
        assert len(input_shape) == 4
        self.conv1 = Conv2D(filters=self.filters * self.dim_capsule, 
                            kernel_size=self.kernel_size, 
                            strides=self.strides, 
                            padding=self.padding, 
                            name='primarycap_conv2d')


    def call(self, inputs):
        output = self.conv1(inputs)
        output = layers.Reshape(target_shape=[-1, self.dim_capsule], name='primarycap_reshape')(output)
        return squash(output)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i])
            new_space.append(new_dim)

        return (None, np.prod(new_space) * self.filters, self.dim_capsule)



def CapsuleNet(input_shape, n_class, num_routing):
    """
    The whole capsule network for MNIST recognition.
    """
    # (None, H, W, C)
    x = Input(input_shape)

    conv1 = Conv2D(filters=256, kernel_size=9, padding='valid', activation='relu', name='init_conv')(x)

    # (None, num_capsules, capsule_dim)
    prim_caps = PrimaryCapsules(filters=32, kernel_size=9, dim_capsule=8, padding='valid', strides=(2, 2))(conv1)
    # (None, n_class, dim_vector)
    digit_caps = DigiCaps(num_capsule=n_class, dim_capsule=16, 
            num_routing=num_routing, name='digitcaps')(prim_caps)

    # (None, n_class)
    pred = Length(name='out_caps')(digit_caps)

    # (None, n_class)
    y = Input(shape=(n_class, ))

    # (None, n_class * dim_vector)
    masked = Mask()([digit_caps, y])  

    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(784, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[28, 28, 1], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return Model([x, y], [pred, x_recon])



if __name__ == '__main__':
    batch_size = 128
    num_classes = 10
    img_rows, img_cols = 28, 28
    lam_recon = 0.392

    mnist_data = np.load('data/mnist.npz')
    x_train, y_train, x_test, y_test = mnist_data['x_train'], mnist_data['y_train'], \
                                        mnist_data['x_test'], mnist_data['y_test']

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    model = CapsuleNet((img_rows, img_cols, 1), num_classes, 3)
    model.compile(loss=margin_loss,
              optimizer='adam',
              loss_weights=[1., lam_recon],
              metrics=['accuracy'])
    model.fit([x_train, y_train], [y_train, x_train], 
              batch_size=batch_size,
              epochs=15,
              validation_data=([x_test, y_test], [y_test, x_test]))


