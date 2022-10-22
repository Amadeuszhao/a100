"""
库导入
"""
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import pickle
from keras.utils import np_utils
import tensorflow as tf

"""
数据集下载与加载（利用KerasAPI）
"""
# x_train_original和y_train_original代表训练集的图像与标签, x_test_original与y_test_original代表测试集的图像与标签
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data()


"""
打印数据集信息
"""
# 文件处理函数(可视化图像原始标签)，调用该函数的位置在后续的函数load_data()中
def load_file(filename):
    with open(filename, 'rb') as datasets:
        data = pickle.load(datasets)
    return data


"""
数据集图像可视化（调用这两个函数的部分在后续load_data()中）
"""
# 单张图像可视化（通过索引选择一张图像可视化）
# mode=0时，选择原始训练集的数据可视化，mode为其他时，选择原始测试集的数据可视化
def mnist_visualize_single(mode, idx):
    if mode == 0:
        plt.imshow(x_train_original[idx], cmap=plt.get_cmap('gray'))        # 显示函数
        title = 'label=' + str(y_train_original[idx])                       # 标签名称（这里是原始编码的标签，即0~9）
        plt.title(title)
        plt.xticks([])  # 不显示x轴
        plt.yticks([])  # 不显示y轴
        plt.show()      # 图像显示
    else:
        plt.imshow(x_test_original[idx], cmap=plt.get_cmap('gray'))
        title = 'label=' + str(y_test_original[idx])
        plt.title(title)
        plt.xticks([])  # 不显示x轴
        plt.yticks([])  # 不显示y轴
        plt.show()


# 多张图像可视化
# 函数的start与end参数表示可视化从start开始，从end结束，例如start=4，end=8表示可视化索引为4、5、6、7的图像（注：以strat开始、以end-1结束）
# 函数的length与width参数表示绘图框显示图像的情况，例如length=3，width=3表示绘制一个3×3（共9个）的画板，画板用来放置可视化的图像
def mnist_visualize_multiple(mode, start, end, length, width):
    if mode == 0:
        for i in range(start, end):
            plt.subplot(length, width, 1 + i)
            plt.imshow(x_train_original[i], cmap=plt.get_cmap('gray'))
            title = 'label=' + str(y_train_original[i])
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
        plt.show()
    else:
        for i in range(start, end):
            plt.subplot(length, width, 1 + i)
            plt.imshow(x_test_original[i], cmap=plt.get_cmap('gray'))
            title = 'label=' + str(y_test_original[i])
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
        plt.show()


"""
分配验证集并可视化各部分数量
"""
def val_set_alloc():
    # 原始原始数据集数据量
    print('原始训练集图像的尺寸：', x_train_original.shape)
    print('原始训练集标签的尺寸：', y_train_original.shape)
    print('原始测试集图像的尺寸：', x_test_original.shape)
    print('原始测试集标签的尺寸：', y_test_original.shape)
    print('===============================')

    # 验证集分配（从测试集中抽取，因为训练集数据量不够）
    x_val = x_train_original[:5000]
    y_val = y_train_original[:5000]
    x_test = x_test_original[5000:]
    y_test = y_test_original[5000:]
    x_train = x_train_original[5000:]
    y_train = y_train_original[5000:]

    # 打印验证集分配后的各部分数据数据量
    print('训练集图像的尺寸：', x_train.shape)
    print('训练集标签的尺寸：', y_train.shape)
    print('验证集图像的尺寸：', x_val.shape)
    print('验证集标签的尺寸：', y_val.shape)
    print('测试集图像的尺寸：', x_test.shape)
    print('测试集标签的尺寸：', y_test.shape)
    print('===============================')

    return x_train, y_train, x_val, y_val, x_test, y_test


"""
图像数据与标签数据预处理
"""
def data_process(x_train, y_train, x_val, y_val, x_test, y_test):

    # 这里把数据从unint类型转化为float32类型, 提高训练精度。
    # x_train = x_train.astype('float32')
    # x_val = x_val.astype('float32')
    # x_test = x_test.astype('float32')

  

    # 图像标签一共有10个类别即0-9，这里将其转化为独热编码（One-hot）向量
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    y_test = np_utils.to_categorical(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


"""
加载数据（即将上述定义的所有函数通过该函数进行整合，输出最终的图像数据与标签）
"""
def load_data():

    # 打印数据集信息
    (x_train, y_train), (x_test_original, y_test_original) = cifar100.load_data()
    
    x_train = tf.image.resize(x_train,(224,224))
    x_test_original = tf.image.resize(x_test_original,(224,224))
    x_val = x_test_original[:5000]
    y_val = y_test_original[:5000]
    x_test = x_test_original[5000:]
    y_test = y_test_original[5000:]
    # 数据预处理（图像数据、标签数据）
    x_train, y_train, x_val, y_val, x_test, y_test = data_process(x_train, y_train, x_val, y_val, x_test, y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    print(x_train.shape)