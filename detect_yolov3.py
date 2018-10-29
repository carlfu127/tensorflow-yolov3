# -*-coding:utf-8-*-
import os
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
import cv2
import colorsys
from model import *
from utils import letterbox_image

class YOLO(object):
    _defaults = {
        "model_path": 'logs/000/',
        "anchors_path": 'model_data/yolo_anchors_416_416.txt',
        "classes_path": 'model_data/food_classes.txt',
        "score" : 0.15,
        "iou" : 0.25,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting

        self.image_input = tf.placeholder(tf.float32, shape=[None, self.model_image_size[0], self.model_image_size[1], 3])
        self.yolo_output = tiny_yolov3(self.image_input, num_anchors // 2, num_classes) if is_tiny_version else \
            yolov3(self.image_input, num_anchors // 3, num_classes)
        try:
            saver = tf.train.Saver(max_to_keep=3)
            model_file = tf.train.latest_checkpoint(self.model_path)
            saver.restore(self.sess, model_file)
            print('{} model, anchors, and classes loaded.'.format(model_file))
        except:
            print("No model loaded!!!")

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = tf.placeholder(tf.int32, [None, 2])
        boxes, scores, classes = yolo_eval(self.yolo_output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.image_input: image_data,
                self.input_image_shape: np.asarray([image.size[1], image.size[0]]).reshape([-1, 2])})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        output = []
        for i, c in reversed(list(enumerate(out_classes))):
            pre = []
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            pre.append(c)
            pre.append(score)
            pre.append(left)
            pre.append(top)
            pre.append(right)
            pre.append(bottom)
            output.append(pre)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image, output

    def close_session(self):
        self.sess.close()

def detect_img(yolo):

    img = ''#image path
    save_dir, name = os.path.split(img)
    print(name)
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image, output = yolo.detect_image(image)
        # fo = open('det_label/%s.txt' % name, 'w')
        # for pre in output:
        #     for index in range(len(pre)):
        #         if index == len(pre)-1:
        #             fo.write(str(pre[index]) + '\n')
        #         else:
        #             fo.write(str(pre[index]) + ' ')
        # fo.close()
        r_image.show()
        # cv2.waitKey(10000)
        # r_image.close()
        # r_image.save('output/%s'  name)
    yolo.close_session()

if __name__ == '__main__':
    # class YOLO defines the default value
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    detect_img(YOLO())
