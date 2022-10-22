import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# retrian the mentioned above model with cifar100
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
import keras
import tensorflow as tf
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import pickle
from keras.utils import np_utils
import tensorflow as tf
NUM_CLASSES = 100
WEIGHT_DECAY = 1e-4
MODEL_PATH ='/data/data/zhz/zw/models'
opt =  tf.keras.optimizers.SGD(lr=0.01, decay = 1e-8)
# Freeze all the layers
# for layer in conv.layers[:]:
#     layer.trainable = False
#model.summary()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train)
    
y_test = np_utils.to_categorical(y_test)
from tensorflow import keras

from tensorflow.keras.layers import  Flatten,Dense ,Dropout,Dense
# Create the model
from tensorflow.keras.layers import Input
IMAGE_SIZE = 32
# 数据集加载
#help(tf.keras.datasets)
model = VGG19(weights=None, include_top=True, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),classes=NUM_CLASSES)
model.summary()

# 编译网络（定义损失函数、优化器、评估指标）
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 定义学习率回调函数（监测验证集精度，根据所设参数，按照标准衰减学习率）
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=8, verbose=1, factor=0.1, min_lr=0.00000001)
chechpointer = ModelCheckpoint(os.path.join(MODEL_PATH, 'vgg19/vgg19_cifar10_{epoch:03d}_{val_accuracy:04f}.h5'),monitor='val_accuracy',save_weights_only=False,period=1,save_best_only=True)

print('当前学习率为：', learning_rate_reduction)
# 定义早停回调函数，当监测的验证集精度连续5次没有优化，则停止网络训练，保存现有模型
es = EarlyStopping(monitor='val_loss', patience=10)
# 回调函数联合
callback = [learning_rate_reduction, es , chechpointer]

# 定义训练过程可视化函数（训练集损失、验证集损失、训练集精度、验证集精度）
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

def train_model(model):
    start_time = datetime.datetime.now()

    
    # 开始网络训练（定义训练数据与验证数据、定义训练代数，定义训练批大小）
    train_history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                              epochs=128,  verbose=2, callbacks=[callback])

    elapsed_time = datetime.datetime.now() - start_time
    print('训练时间：', elapsed_time)

    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')


if __name__ == '__main__':
    train_model(model)              # 选择要训练的网络
