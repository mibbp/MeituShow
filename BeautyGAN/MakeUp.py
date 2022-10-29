
import numpy as np

import glob
from imageio import imread, imsave
import cv2
import argparse
import sys
#忽略
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'test.jpg'), help='path to the no_makeup image')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

imgpath = "D:/Users/mibbp/PycharmProjects/pythonProject1/BeautyGAN/imgs/no_makeup/test.jpg"

def main():

    batch_size = 0
    img_size = 256

    no_makeup = cv2.resize(imread(imgpath), (img_size, img_size))
    X_img = np.expand_dims(preprocess(no_makeup), 0)
    makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
    result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
    result[img_size: 2 * img_size, :img_size] = no_makeup / 255.

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.import_meta_graph(os.path.join('model', 'model.meta'))
    saver.restore(sess, tf.train.latest_checkpoint('model'))

    graph = tf.compat.v1.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    Y = graph.get_tensor_by_name('Y:0')
    Xs = graph.get_tensor_by_name('generator/xs:0')

    # for i in range(len(makeups)):
    # operation = 0
    i = 0
    file_object = open("D:/MakeupOperation.txt", 'r')  # 创建一个文件对象，也是一个可迭代对象
    try:
        all_the_text = file_object.read()  # 结果为str类型
        # print(type(all_the_text))
        # print("all_the_text=", all_the_text)
    finally:
        file_object.close()

    i = int(all_the_text)
    print(i)

    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    Xs_ = deprocess(Xs_)



    result = makeup / 255.
    result = Xs_[0]
    result = (result * 255.0).astype('uint8')

    # print(makeups)
    imsave('D:/Users/mibbp/PycharmProjects/pythonProject1/BeautyGAN/result.jpg', result)
    # Img = cv2.imread("D:/Users/mibbp/PycharmProjects/pythonProject1/BeautyGAN/result.jpg")
    # cv2.imshow("change", Img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
