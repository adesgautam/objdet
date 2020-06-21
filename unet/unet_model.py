import keras

from keras import Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, UpSampling2D, Concatenate, Input, MaxPooling2D, Dropout, Dense, Flatten #Conv2DTranspose, BatchNormalization

def unet(nclasses, input_shape):
	model = Sequential()

	inp = Input(shape=input_shape)

	x = Conv2D(64, kernel_size=(3,3), padding = 'same',  activation='relu', kernel_initializer = 'he_normal')(inp)
	x1 = Conv2D(64, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)

	x = MaxPooling2D(pool_size=(2,2))(x1)

	x = Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)
	x2 = Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)

	x = MaxPooling2D(pool_size=(2,2))(x2)

	x = Conv2D(256, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)
	x3 = Conv2D(256, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)

	x = MaxPooling2D(pool_size=(2,2))(x3)

	x = Conv2D(512, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)
	x4 = Conv2D(512, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)

	x = MaxPooling2D(pool_size=(2,2))(x4)

	x = Conv2D(1024, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)
	x5 = Conv2D(1024, kernel_size=(3,3), padding = 'same', activation='relu')(x)

	# Upsampling
	y = UpSampling2D(size=(2,2))(x5)
	y4 = Conv2D(512, kernel_size=(2,2), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(y)

	# Concat
	concat4 = Concatenate(axis=3)([x4, y4])

	y = Conv2D(512, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(concat4)
	y = Conv2D(512, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(y)

	y = UpSampling2D(size=(2,2))(y)
	y3 = Conv2D(256, kernel_size=(2,2), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(y)

	concat3 = Concatenate(axis=3)([x3, y3])

	y = Conv2D(256, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(concat3)
	y = Conv2D(256, kernel_size=(3,3), padding = 'same',  activation='relu', kernel_initializer = 'he_normal')(y)

	y = UpSampling2D(size=(2, 2))(y)
	y2 = Conv2D(128, kernel_size=(2,2), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(y)

	concat2 = Concatenate(axis=3)([x2, y2])

	y = Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(concat2)
	y = Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(y)

	y = UpSampling2D(size=(2, 2))(y)
	y1 = Conv2D(64, kernel_size=(2,2), padding = 'same', activation='relu', kernel_initializer = 'he_normal')(y)

	concat1 = Concatenate(axis=3)([x1, y1])

	y = Conv2D(64, kernel_size=(3,3), padding = 'same',  activation='relu', kernel_initializer = 'he_normal')(concat1)
	y = Conv2D(64, kernel_size=(3,3), padding = 'same',  activation='relu', kernel_initializer = 'he_normal')(y)

	y = Conv2D(2, kernel_size=(3,3), padding = 'same',  activation='relu', kernel_initializer = 'he_normal')(y)

	out = Conv2D(nclasses, kernel_size=(1,1), padding = 'same',  activation='sigmoid')(y)

	model = Model(input=inp, output=out)

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

	# model.summary()

	return model