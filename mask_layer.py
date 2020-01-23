from keras.engine.topology import Layer
import keras.backend as K


def sub_sampling_structure(i_context, amount_of_packets,tmp_offset=-1):
    inp_context =[i - i_context[0] for i in i_context]
    mx_ind =max(inp_context)
    if tmp_offset<=0 :
        offset =2*mx_ind
    else:
        offset =mx_ind+tmp_offset

    cond_line =[1 if j % offset in inp_context else 0 for j in  range(amount_of_packets)]
    return cond_line

def tdnn_pre_proc(x_inputs,cond_line ):
    zz =K.permute_dimensions(x_inputs,(0,2,1))
    z2= K.permute_dimensions(zz* cond_line,(0,2,1))
    return z2




class my_mask(Layer):

    def __init__(self, input_context, sub_sampling, tmp_offset=-1 ,**kwargs):
          self.sub_sampling = sub_sampling
          self.cond_line =[]
          if self.sub_sampling:
                   self.cond_line= sub_sampling_structure(input_context,tmp_offset,)
          self.offset = tmp_offset
          self.input_context =input_context
          super(my_mask, self).__init__()  # Be sure to call this at the end
    def build(self, input_shape):

        self.cond_line = sub_sampling_structure(self.input_context,input_shape[1], self.offset )



    def call(self, x ):

        if not(self.sub_sampling) :
            return x
        return tdnn_pre_proc(x, self.cond_line)

    def compute_output_shape(self, input_shape):
        return input_shape



if __name__ =="__main__":
    from keras.optimizers import Adam
    from keras.losses import categorical_crossentropy,binary_crossentropy
    from keras.layers import Dense
    from Knowledge_transfer.Train.pooling_layer import xvpool

    import numpy as np

    aa= [-1,2]

    from keras.models import Sequential
    from keras.layers import Conv1D

    xx = np.arange(100*11*4)
    xx = np.reshape(xx, (100, 11, 4))
    yy = np.random.randint(2, size=100)

    yy = K.one_hot(yy, 2)
    yy = K.eval(yy)

    print ()
    mm = Sequential()
    mm.add(Conv1D(filters=12, kernel_size=2,input_shape=(11,4)))
    mm.add(my_mask([-1, 2 ], True))
    mm.add(Conv1D(filters=8, kernel_size=2))
    mm.add(my_mask([-1, 2 ], True))
    mm.add(Conv1D(filters=10, kernel_size=2))
    mm.add(xvpool())
    mm.add(Dense(2,activation="sigmoid"))
    mm.compile(loss=binary_crossentropy, optimizer=Adam())

    print (mm.summary())
    mm.fit(xx,yy,nb_epoch=2,steps_per_epoch=60)
    print(yy.shape, " La La ")


    mm.add(my_mask([-1,1, 2],True,1))
    yy = mm.predict(xx)
    print (yy)
    # exit(44)
    mm=Sequential()
    mm.add(Conv1D(filters=5,kernel_size=2,input_shape=(20,4)))


    mm.add(my_mask([-1,1, 2],False))

    mm.add(Conv1D(filters=8, kernel_size=2))
    mm.add(my_mask([-1,  2], True,tmp_offset=1))
    mm.build()
    print (mm.summary())
    xx =np.random.randn(10,20,4)
    yy = mm.predict(xx)
    print (xx.shape,yy.shape)
    # mm.add(Conv1D(filters=10, kernel_size=2))

    print("fianlly", mm.output_shape)




    exit(22)
