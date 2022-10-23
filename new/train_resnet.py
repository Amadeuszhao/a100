from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from load_cifar100 import load_cifar100_data
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
import os
#K.set_image_dim_ordering('tf')

def resnet50_model(num_classes):
	
	
	# create the base pre-trained model
	base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

	# add a global spatial average pooling layer
	x = base_model.output
	print("Shape:", x.shape)
	x_newfc = x = GlobalAveragePooling2D()(x)

	# and a logistic layer -- let's say we have 100 classes
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
  

if __name__ == '__main__':

    # Example to fine-tune on samples from Cifar100

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 100 
    nb_epoch = 50
    MODEL_PATH ='/data/data/zhz/zw/models'

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar100_data(img_rows, img_cols)

    # Load our model
    model = resnet50_model(num_classes)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=8, verbose=1, factor=0.1, min_lr=0.00000001)
    chechpointer = ModelCheckpoint(os.path.join(MODEL_PATH, 'resnet/resnet_cifar100_{epoch:03d}_{val_accuracy:04f}.h5'),monitor='val_accuracy',save_weights_only=False,period=1,save_best_only=True)
    
    print('当前学习率为：', learning_rate_reduction)
    # 定义早停回调函数，当监测的验证集精度连续5次没有优化，则停止网络训练，保存现有模型
    es = EarlyStopping(monitor='val_loss', patience=15)
    callback = [learning_rate_reduction, es , chechpointer]

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              callbacks=[callback]
              )

  
    
    # Make predictions
    predictions_valid = model.predict(X_valid, verbose=1)

    # Cross-entropy loss score
    scores = log_loss(Y_valid, predictions_valid)
    print("Cross-entropy loss score",scores)
    
    ## evaluate modelon test data:
    score = model.evaluate(X_valid, Y_valid, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))