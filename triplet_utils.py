import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import concatenate
import keras.backend as K

class Identity(Layer):

    def __init__(self, inp_0, **kwargs):
        self.output_dim = inp_0
        super(Identity, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self):
        return tuple(self.output_dim)




def dec_value(x, y, epsilon=1e-8):

    # Constants
    gamma = 1.0
    ten_01 = tf.convert_to_tensor([0, 1.])
    ten1m1 = tf.convert_to_tensor([1.0, -1])
    ten_00 = tf.convert_to_tensor([1., 0])
    epsilon = 1e-8
    const_0 = tf.constant(gamma * (1. + epsilon), dtype="float32")



    pos1, neg1 = tf.split(x, axis=1, num_or_size_splits=2)

    pos = concatenate([tf.constant(1.) + K.zeros_like(pos1), pos1])
    neg = concatenate([tf.constant(1.) + K.zeros_like(neg1), neg1])

    mm = K.equal(y, ten_00)
    zz = K.switch(mm, tf.add(K.zeros_like(y), tf.convert_to_tensor([0., 1.])),
                  tf.add(K.zeros_like(y), tf.convert_to_tensor([1., -1])))

    pos_zz = tf.multiply(pos, zz)

    pos_zz = tf.reduce_sum(pos_zz, axis=-1, keepdims=True)

    mm = K.equal(y, ten_01)
    zz = K.switch(mm, tf.add(K.zeros_like(y), tf.convert_to_tensor([0., 1.])),
                  tf.add(K.zeros_like(y), tf.convert_to_tensor([1, -1.])))
    neg_zz = tf.multiply(neg, zz)
    neg_zz = tf.reduce_sum(neg_zz, axis=-1, keepdims=True)

    const_0 = tf.constant(gamma * (1. + epsilon), dtype="float32")

    a = -tf.log(const_0 - pos_zz)
    b = -tf.log(const_0 - neg_zz)
    conca_val = concatenate([a, b])

    return conca_val


class Decide(Layer):
    def __init__(self, yy):
        self.y = yy
        super(Decide, self).__init__()

    def call(self, x):
        return dec_value(x, self.y)

    def get_output_shape_for(self):
        return (1)


