from keras.layers import Layer, concatenate
import tensorflow as tf


def normal_array(x):
    al = tf.nn.l2_normalize(x,axis=1 )
    return al

class norm_vec(Layer):

    def __init__(self,   **kwargs):
        super(norm_vec, self).__init__(**kwargs)

    def build(self, input_shape):
         super(norm_vec, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x ):
        output =  normal_array(x)
        return output

    def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            assert input_shape[-1]

            tmp_0 = input_shape[-1]
            output_shape = (input_shape[0],tmp_0)
            return output_shape
if __name__ =="__main__":

    import numpy as np
    from keras.layers import Dense,Conv1D
    from keras.models import Sequential
    xx = np.random.randn(1000,15)
    # print (xx)
    # print ("ttt",np.mean(xx,axis=1),np.std(xx,axis=1))
    dd = Sequential()
    dd.add(Dense(15, input_shape=(15,)))
    dd.add(norm_vec())

    # dd.add(Dense(5, activation="relu"))
    kk = np.linalg.norm(xx, ord=2, axis=0)
    print ("ll ",kk)
    xx = np.random.randint(7, size=(1000, 15))

    yy =dd.predict(xx)
    print ("s1", type(yy),yy.shape)
    cntr = 0.
    a=522
    for j in range(15):
        cntr += yy[a,j] * yy[a,j]
    print("cnt = ",cntr)
    exit(332)

    print (yy[0,:])
    print ("yyy", type(yy))
    print(dd.summary())
    print (yy.shape)
    kk = np.linalg.norm(yy, ord=2, axis=0)
    print("zzz", kk.shape,kk)
