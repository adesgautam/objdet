import keras

from keras import Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, UpSampling2D, Concatenate, Input, MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2DTranspose, BatchNormalization

def conv_block(input, n_filters, kernel_size=3 batchnorm=True):
	x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size), padding = 'same', kernel_initializer = 'he_normal')(input)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size), padding = 'same', kernel_initializer = 'he_normal')(input)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def unet(nclasses, input_shape, dropout=0.1):
	model = Sequential()

	inp = Input(shape=input_shape)

	x1 = conv_block(input=inp, n_filters = 64, kernel_size=3, batchnorm=True)
	x = MaxPooling2D(pool_size=(2,2))(x1)
	x = Dropout(dropout)(x)

	x2 = conv_block(input=x, n_filters = 128, kernel_size=3, batchnorm=True)
	x = MaxPooling2D(pool_size=(2,2))(x2)
	x = Dropout(dropout)(x)

	x3 = conv_block(input=x, n_filters = 256, kernel_size=3, batchnorm=True)
	x = MaxPooling2D(pool_size=(2,2))(x3)
	x = Dropout(dropout)(x)

	x4 = conv_block(input=x, n_filters = 512, kernel_size=3, batchnorm=True)
	x = MaxPooling2D(pool_size=(2,2))(x4)
	x = Dropout(dropout)(x)

	x5 = conv_block(input=x, n_filters = 1024, kernel_size=3, batchnorm=True)

	# Upsampling
	y4 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(x5)
	concat4 = Concatenate(axis=3)([x4, y4])
	drp4 = Dropout(dropout)(concat4)
	y = conv_block(input=drp4, n_filters = 512, kernel_size=3, batchnorm=True)

	y3 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(y)
	concat3 = Concatenate(axis=3)([x3, y3])
	drp3 = Dropout(dropout)(concat3)
	y = conv_block(input=drp3, n_filters = 256, kernel_size=3, batchnorm=True)

	y2 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(y)
	concat2 = Concatenate(axis=3)([x2, y2])
	drp2 = Dropout(dropout)(concat2)
	y = conv_block(input=drp2, n_filters = 128, kernel_size=3, batchnorm=True)

	y2 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(y)
	concat1 = Concatenate(axis=3)([x1, y1])
	drp1 = Dropout(dropout)(concat1)
	y = conv_block(input=drp1, n_filters = 64, kernel_size=3, batchnorm=True)

	# if using nclasses > 1 use softmax activation and categorical_crossentropy loss
	out = Conv2D(nclasses, kernel_size=(1,1), padding = 'same',  activation='sigmoid')(y)

	model = Model(input=inp, output=out)

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

	# model.summary()

	return model