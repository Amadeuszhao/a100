from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten,Dropout
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from load_cifar100 import load_cifar100_data
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
import os
def resnet50_model(num_classes):
	
	
	# create the base pre-trained model
	base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

	# add a global spatial average pooling layer
	x = base_model.output
	print("Shape:", x.shape)
	x_newfc = x = GlobalAveragePooling2D()(x)
	# and a logistic layer -- let's say we have 100 classes
	x_newfc = Dense(512)(x_newfc)
	x_newfc = Dropout(0.5)(x_newfc)

	x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

	# this is the model we will train
	# Create another model with our customized softmax
	model = Model(inputs=base_model.input, outputs=x_newfc)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional resnet50 layers
	for layer in base_model.layers:
		layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	# Learning rate is changed to 0.001
	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',  'top_k_categorical_accuracy'])
	return model
model = resnet50_model(100)
model.summary()