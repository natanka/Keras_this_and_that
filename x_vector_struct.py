from keras.layers import Dense, Conv1D,Flatten
from keras.models import Sequential
from Knowledge_transfer.Train.pooling_layer import xvpool
from Knowledge_transfer.Train.x_vector.mask_layer import my_mask
import  Konwledge_transfer.Train.train_const as alg_con
from keras.activations import relu,softmax

def construc_x_vector(input_shape ,xvec_flg):
    model =Sequential()
    model.add(Conv1D(filters=400, kernel_size=3,   input_shape=input_shape))
    model.add(my_mask([-1,2],False))
    model.add(Conv1D(filters=400, kernel_size=3 ))
    model.add(my_mask([-2,2],False))


    model.add(Conv1D(filters=400, kernel_size=3 ))
    model.add(my_mask([-2,2],True))
    model.add(Conv1D(filters=400, kernel_size=3 ))
    # model.add(Flatten())
    model.add(xvpool())

    model.add(Dense(500,activation =relu))
    model.add(Dense(alg_con.embedding_dim,  activation="sigmoid",name="xvector"+str(xvec_flg)))
    return model


if __name__ == "__main__":
    import numpy as np
    import keras.backend as K
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy,categorical_crossentropy

    ss = 2500
    input_s = (ss, 270, 39)
    xx = np.random.randn(input_s[0], input_s[1], input_s[2])
    data_dic = np.load(alg_con.train_file)
    xx = data_dic['arr_0']
    yy = data_dic['arr_1']

    # yy = np.random.randint(2, size=ss)
    #
    # yy = K.one_hot(yy, 2)
    # yy = K.eval(yy)

    model = Sequential()
    model.add(Conv1D(filters=800, kernel_size=3,   input_shape=(270,39)))
    model.add(my_mask([-1, 2], True))
    model.add(Conv1D(filters=800, kernel_size=3 ))
    model.add(my_mask([-2, 2], False))

    model.add(Conv1D(filters=400, kernel_size=3 ))
    model.add(my_mask([-2, 2], True))
    model.add(Conv1D(filters=400, kernel_size=3 ))

    model.add(xvpool(name="gg"))
    model.add(Dense(2,activation="sigmoid"))
    model.compile(loss=categorical_crossentropy, optimizer=Adam())
    from keras.models import Model

    m1 = Model(inputs= model.input, outputs=model.get_layer("gg").output)
    print (m1.summary())
    model.fit(xx[:600, 0, :, :], yy[:600, :], epochs=1, batch_size=50)
    print(m1.predict(xx[:20, 0, :, :]))
    print("that's it")
    exit(1211)

    # data_dic = np.load(alg_con.train_file)
    # xx = data_dic['arr_0']
    # yy = data_dic['arr_1']
    xx= np.random.rand(600,1,270,39)
    yy= np.random.randint(2,size=600)
    # yy = np.random.randint(2, size=1300)

    yy = K.one_hot(yy, 2)
    yy = K.eval(yy)

    model.fit(xx[:600,0,:,:], yy[:600,:], epochs=1, batch_size=50)
    print (m1.predict(xx[:20,0,:,:]))
    print ("that's it")
    exit(222)
    xx = np.arange(400 * 30 * 4)
    xx = np.reshape(xx, (400, 30, 4))
    yy = np.random.randint(alg_con.embedding_dim, size=400)

    yy = K.one_hot(yy,alg_con.embedding_dim)
    yy = K.eval(yy)
    mm =construc_x_vector((30,4))
    mm.compile(loss =categorical_crossentropy,optimizer=Adam())
    mm.fit(xx,yy,epochs=3,batch_size=50)
