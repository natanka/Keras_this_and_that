import numpy as np
from keras.layers import Input,Dense
from keras.models import Model

# A toy data creation
def create_toy_folder():
    xx_train =np.random.rand(100,32)
    yy_train = np.random.randint(2,size=100)
    for j in range(100):
        np.save('folder1\\aa_x_'+str(j)+'.npy',xx_train[j,:])
        np.save('folder2\\aa_y_' + str(j) + '.npy', yy_train[j])

    xx_test = np.random.rand(50, 32)
    yy_test = np.random.randint(2,size=50)
    print (yy_test)
    for j in range(50):
        np.save('foldertest\\aa_x_'+str(j)+'.npy',xx_test[j,:])
        np.save('foldertest\\aa_y_' + str(j) + '.npy', yy_test[j])

    return xx_train,yy_train,xx_test,yy_test


class  organizer():
    def __init__(self,folder, amount_files, batch_size,feat_dim):
        self.mode_folder =folder
        self.y_folder = folder+"_y"
        self.nfiles =amount_files
        self.batch_size = batch_size
        self.feat_dim =feat_dim
        self.xtrain = np.empty((0, self.feat_dim))
        self.ytrain = []


    def __len__(self):
        return len(self.list_IDs)



    def get_data_item(self, index):
        x = np.expand_dims(np.load(self.mode_folder +"\\aa_x_"+str(index) + '.npy'),axis=0) #expand_dims only if needed
        y = np.load(self.y_folder + "\\aa_y_" + str(index) + '.npy').item()  # reading ndarray as scalar since np.load brings only arrays
        return x,y

    def data_genereator(self):

        counter = 0
        while True:
            for index in range(self.nfiles):
                if self.xtrain.shape[0]==self.batch_size:
                    self.xtrain = np.empty((0,  self.feat_dim))
                    self.ytrain= []

                x,y =self.get_data_item(index)

                self.xtrain = np.concatenate(( self.xtrain, x), axis=0)
                self.ytrain.append(y)
                counter = counter + 1
                if (counter % self.batch_size == 0):
                    x= self.xtrain

                    y= self.ytrain
                    yield x,    np.array(y)

    def get_data_test(self, index):
        x = np.expand_dims(np.load(self.mode_folder + "\\aa_x_" + str(index) + '.npy'),
                           axis=0)  # expand_dims only if needed

        return x
    def data_genereator_test(self):

        counter = 0
        while True:
            for index in range(self.nfiles):
                if self.xtrain.shape[0]==self.batch_size:
                    self.xtrain = np.empty((0,  self.feat_dim))


                x  =self.get_data_test(index)

                self.xtrain = np.concatenate(( self.xtrain, x), axis=0)

                counter = counter + 1
                if (counter % self.batch_size == 0):
                    x= self.xtrain


                    yield x








if __name__ == '__main__':
    # The way  I created the toy data
    # x_train,ytr,xte,yte =create_toy_folder()

    train_p ="your train data x (y if needed"
    test_p ="your test data x"
    ntest = 50
    batch_size =8
    nitems=100
    feat_dim=32


    organ_train =  organizer(train_p, nitems, 8, feat_dim)
    organ_test = organizer(test_p, ntest, 8, feat_dim )
    gen =organ_train.data_genereator()

    inp = Input(shape=( 32,))
    x =Dense(5,activation="relu")(inp)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inp], outputs=[x])
    model.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['accuracy'])
    print (model.summary())
   #Train
    model.fit_generator(gen, steps_per_epoch=nitems // batch_size, verbose=1)

    #Test
    gent = organ_test.data_genereator_test()
    score = model.predict_generator(gent, ntest // batch_size, verbose=1)
    print (score)
    print ("haha")