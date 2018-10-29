# -*-coding:utf-8-*-

"""
Retrain the YOLO model for my own dataset.
"""

from __future__ import print_function
import os
from utils import *
from model import *

def _main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    annotation_path = 'model_data/train_shuffle.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/food_classes.txt'
    anchors_path = 'model_data/yolo_anchors_416_416.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw
    is_tiny_model = len(anchors) == 6
    if is_tiny_model:
        image_input, y_true1, y_true2, loss, learning_rate = create_tiny_yolov3(input_shape, anchors, num_classes)
    else:
        image_input, y_true1, y_true2, y_true3, loss, learning_rate = create_yolov3(input_shape, anchors, num_classes)

    config = tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('restore model', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    print('Starting training')
    last_avg_val_cost = np.inf
    lr = 1e-4
    patience = 0
    if is_tiny_model:
        batch_size = 32
        for epoch in range(100):
            train_cost = 0.
            for step in range(len(lines[:num_train])//batch_size):
                image, y = data_generator(step, lines[:num_train], batch_size, input_shape, anchors, num_classes)
                sess.run(optimizer, feed_dict={image_input:image, y_true1:y[0], y_true2:y[1], learning_rate:lr})
                cost = sess.run(loss, feed_dict={image_input:image, y_true1:y[0], y_true2:y[1], learning_rate:lr})
                train_cost += cost
                print("Epoch:", '%03d' % (epoch + 1), "Step:", '%03d' % step, "Loss:", str(cost))
            avg_train_cost = train_cost*1./(len(lines[:num_train])//batch_size)
            print("Epoch:", '%03d' % (epoch + 1), "Train Loss:", str(avg_train_cost), ' '),

            val_cost = 0.
            for step in range(len(lines[num_train:])//batch_size):
                image, y = data_generator(step, lines[num_train:], batch_size, input_shape, anchors, num_classes)
                cost = sess.run(loss, feed_dict={image_input:image, y_true1:y[0], y_true2:y[1], learning_rate:0.})
                val_cost += cost
            avg_val_cost = val_cost*1./(len(lines[num_train:])//batch_size)
            print('Val Loss:', str(avg_val_cost))
            if avg_val_cost < last_avg_val_cost:
                patience = 0
                last_avg_val_cost = avg_val_cost
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
            else:
                patience += 1
                if patience == 3:
                    patience = 0
                    lr *= 0.1
    else:
        batch_size = 16
        for epoch in range(100):
            train_cost = 0.
            for step in range(len(lines[:num_train]) // batch_size):
                image, y = data_generator(step, lines[:num_train], batch_size, input_shape, anchors, num_classes)
                sess.run(optimizer, feed_dict={image_input: image, y_true1: y[0], y_true2: y[1], y_true3:y[2], learning_rate: lr})
                cost = sess.run(loss, feed_dict={image_input: image, y_true1: y[0], y_true2: y[1], y_true3:y[2], learning_rate: lr})
                train_cost += cost
                print("Epoch:", '%03d' % (epoch + 1), "Step:", '%03d' % step, "Loss:", str(cost))
            avg_train_cost = train_cost * 1. / (len(lines[:num_train]) // batch_size)
            print("Epoch:", '%03d' % (epoch + 1), "Train Loss:", str(avg_train_cost), ' '),

            val_cost = 0.
            for step in range(len(lines[num_train:]) // batch_size):
                image, y = data_generator(step, lines[num_train:], batch_size, input_shape, anchors, num_classes)
                cost = sess.run(loss, feed_dict={image_input: image, y_true1: y[0], y_true2: y[1], y_true3:y[2], learning_rate: 0.})
                val_cost += cost
            avg_val_cost = val_cost * 1. / (len(lines[num_train:]) // batch_size)
            print('Val Loss:', str(avg_val_cost))
            if avg_val_cost < last_avg_val_cost:
                patience = 0
                last_avg_val_cost = avg_val_cost
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
            else:
                patience += 1
                if patience == 3:
                    patience = 0
                    lr *= 0.1

def get_classes(class_path):
    file = open(class_path, 'r')
    class_name = [c.strip() for c in file.readlines()]
    return class_name

def get_anchors(anchors_path):
    file = open(anchors_path)
    anchors = file.readline()
    anchors = [float(a) for a in anchors.split(',')]
    return np.reshape(np.asarray(anchors), (-1, 2))


if __name__ == '__main__':
    _main()