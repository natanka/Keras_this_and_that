from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, binary_crossentropy

from keras.layers import Layer, concatenate
from keras.activations import relu,softmax

import tensorflow as tf
import numpy as np
import keras.backend as K


def gen_stat(x):
   average = tf.reduce_mean(x,axis=1,keepdims=True)

   zsq = (tf.square(x-average))
   std_dev = tf.sqrt(tf.reduce_mean(zsq,axis=1,keepdims=True))
   return concatenate([average, std_dev],axis=2)

class xvpool(Layer):

    def __init__(self,   **kwargs):
        super(xvpool, self).__init__(**kwargs)

    def build(self, input_shape):
         super(xvpool, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x ):
        output =  gen_stat(x)
        return output

    def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            assert input_shape[-1]
            output_shape = (input_shape[0],2*input_shape[2])
            return output_shape


if __name__ =="__main__":
    # from keras.optimizers import Adam
    # from keras.losses import categorical_crossentropy, binary_crossentropy
    # x0= np.array([1,2,3])
    # y0 = np.array([-1,2,-3])
    # x1 =  tf.convert_to_tensor(x0)
    # y1 = tf.convert_to_tensor(y0)
    # # z= tf.concat([x1,y1],axis=1)
    # print (K.eval(z))
    # exit(33)
    import numpy as np
    from keras.layers import Dense,Conv1D
    from keras.models import Sequential
    ss=2500
    input_s= (ss,270,39)
    xx = np.random.randn(input_s[0],input_s[1],input_s[2])
    dd = Sequential()
    dd.add(Conv1D(filters=800, kernel_size=3, input_shape=(input_s[1], input_s[2])))
    dd.add(Conv1D(filters=800, kernel_size=3 ))

    dd.add(xvpool())
    dd.add(Dense(2, activation="sigmoid"))
    dd.compile(loss=binary_crossentropy, optimizer=Adam())
    yy = np.random.randint(2, size=ss)

    yy = K.one_hot(yy, 2)
    yy = K.eval(yy)

    dd.fit(x=xx,y=yy,batch_size=100,nb_epoch=10)
    yy = dd.predict(xx)
    print(yy.shape)
    print(dd.summary())
    exit(44)
    print (dd.output_shape)
    m= tf.convert_to_tensor(yy)
    print (m)
    y2 = gen_stat(m)
    print ("l",y2.shape)

    exit(22)

    # print (xx)
    print ("ttt",np.mean(xx,axis=1),np.std(xx,axis=1))
    m0 =np.mean(xx,axis=1)
    s0 =np.std(xx,axis=1)
    m0 =np.expand_dims(m0,axis=1)
    s0 = np.expand_dims(s0, axis=1)
    mm0= np.dstack((m0,s0))
    print (mm0.shape)
    # xx =np.expand_dims(xx,axis=3)

    xt = tf.convert_to_tensor(xx)

    print (xt)
    dd =Sequential()
    dd.add(Conv1D(filters=10,kernel_size=3,input_shape=(7,4)))
    dd.add(xvpool(input_shape=(3,4)))
    yy =dd.predict(xx)
    print (yy.shape)
    exit(22)
    # dd.add(Dense(5,activation="relu"))
    # print (dd.summary())
    # exit(333)
    dd.add(xvpool(input_shape=(4,)))
    # print ("hihi")
    # print ("moshe",dd.layers[0].compute_output_shape((10,4)))
    # print (dd.summary())
    exit(222)
    ss= dd.predict(xx)
    zz=[]
    for i in range (10):
         zz.append([[ss[i,0,:],[mm0[i,0,:]]]],)
         print(zz[i])
    print (type(zz[0]))
    print (ss.shape,type(ss))
    # zz =gen_stat(xx)
    #
    # print ("uuu",K.eval(zz))
