# coding=utf-8

import os
# 图像读取库
from PIL import Image
# 矩阵运算库
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 数据文件夹
data_dir = "test"
# 训练还是测试True False
train = False
# 模型文件路径
model_path = "model/image_model"


# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels


fpaths, datas, labels = read_data(data_dir)

# 计算有多少类图片
num_classes = len(set(labels))

# 定义placeholder，存放输入和标签
datas_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
labels_placeholder = tf.compat.v1.placeholder(tf.int32, [None])

# 存放DropOut参数的容器，训练时为0.25，测试时为0
dropout_placeholdr = tf.compat.v1.placeholder(tf.float32)

# 定义卷积层，20个卷积核，卷积核大小为5，用Relu激活
conv0 = tf.compat.v1.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool0 = tf.compat.v1.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 定义卷积层，40个卷积核，卷积核大小为4，用Relu激活
conv1 = tf.compat.v1.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool1 = tf.compat.v1.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 定义卷积层，60个卷积核，卷积核大小为4，用Relu激活
conv2 = tf.compat.v1.layers.conv2d(pool0, 60, 3, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool2 = tf.compat.v1.layers.max_pooling2d(conv2, [2, 2], [2, 2])

# 将三维特征转换为1维向量
flatten = tf.compat.v1.layers.flatten(pool2)

# 全连接层，转换为长度为400的特征向量
fc = tf.compat.v1.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.compat.v1.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.compat.v1.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.compat.v1.arg_max(logits, 1)

# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)
# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

# 用于保存和载入模型
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    if train:
        print("train mode")
        # 如果是训练，初始化参数
        sess.run(tf.compat.v1.global_variables_initializer())
        # 定义输入和Label以填充容器，训练时dropout为0.5
        train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0.5
        }
        for step in range(150):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("train finished saved model to{}".format(model_path))
    else:
        print("test mode")
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("from{}load the model".format(model_path))
        # label和名称的对照关系
        label_name_dict = {
            0: "bus",
            1: "taxi",
            2: "truck",
            3: "family sedan",
            4: "minibus",
            5: "jeep",
            6: "SUV",
            7: "heavy truck",
            8: "racing car",
            9: "fire engine"
        }
        # 定义输入和Label以填充容器，测试时dropout为0
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实Label与预测模型Label
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
            # 将Label id转换为label名
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))

        # 正确次数
        correct_number = 0
        # 计算正确率
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
            if real_label == predicted_label:
                correct_number += 1

        correct_rate = correct_number / 200
        print('correct: {:.2%}'.format(correct_rate))
