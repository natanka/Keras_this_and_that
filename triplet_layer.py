import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import concatenate
# from Speaker_Change.Konwledge_transfer import all_enums
import numpy as np
from keras.layers import concatenate


def loss_func_a (y_true, y_pred):

    pos, neg = tf.split(y_pred,axis=1,num_or_size_splits=2)
    return neg + pos



def spec_loss(x  ):
    anchor ,positive, neg  = tf.split(x, axis=1, num_or_size_splits=3)

    pos_sq = tf.square(tf.subtract(anchor, positive))
    pos_sq = tf.reduce_mean(pos_sq, axis=-1,keepdims=True)

    neg_sq = tf.square(tf.subtract(anchor, neg))
    neg_sq = tf.reduce_mean(neg_sq, axis=-1, keepdims=True)

    uu=concatenate(  [pos_sq, neg_sq])
    return uu
def spec_cosine_loss(x  ):
    al, p1, n1  = tf.split(x, axis=1, num_or_size_splits=3)

    pos_sq = tf.losses.cosine_distance(al,p1,dim=1,reduction="none")
    neg_sq = tf.losses.cosine_distance(al, n1,dim=1,reduction="none")

    uu=concatenate(  [pos_sq, neg_sq])
    return uu





class trip_loss_layer(Layer):

    def __init__(self, output_dim, dist_met,  **kwargs ):
        self.output_dim = output_dim
        self.dist_met =dist_met
        super(trip_loss_layer, self).__init__(**kwargs)

    def build(self, input_shape):
         super(trip_loss_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x ):
        if self.dist_met.name=="l2":
           return spec_loss(x)
        else :
           return spec_cosine_loss (x)

    def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            assert input_shape[-1]
            output_shape = list(input_shape)
            output_shape[-1] = self.output_dim
            return tuple(output_shape)
if __name__ == '__main__' :
  # xx =np.array([4.,4.,4.,4.,4.,4.,4.,4.,4.,4., 5.,5.,5.,5.,5.,5.,5.,5.,5.,5. ,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.])
  xx =np.array([4.,5.,6.,7.,8., 5.,6.,7.,8,9,8.,7.,6.,5.,4.])

  y1 =3*xx

  xx =np.vstack((xx,y1))
  y1 = 2 * y1
  xx =np.vstack((xx,y1))
  print (xx.shape)
  print ("yyy",K.eval(spec_cosine_loss(xx)))
  exit(33)
  rr1 = spec_loss(xx,10,10)

  print (K.eval(rr1))
  rr2 = spec_loss(xx, 10, 20)

  print(K.eval(rr2))
  print ("ok")
  y_pred=tf.stack([rr1,rr2],axis=1)
  print ("hhhhh",K.eval(y_pred))
  y_true =tf.convert_to_tensor([[1.,0], [0,1.],[0,1.]])
  print ("jjjjj",y_true,y_pred,K.int_shape(y_pred),K.int_shape(y_true))

  # y_true =np.reshape(y_true,(1,2))

  print ("uuu ",K.eval(loss_func_a(y_true, y_pred, epsilon=1e-8)))
  exit(11)
  xx= trip_loss_layer(output_dim=2,)
  print (xx)