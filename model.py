# -*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import batch_norm

def create_yolov3(input_shape, anchors, num_classes):
    h, w = input_shape
    num_anchors = len(anchors)
    image_input = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    size = {0: 32, 1: 16, 2: 8}
    y_true1 = tf.placeholder(tf.float32, [None, h // size[0], w // size[0],
                                          num_anchors // 3, 5 + num_classes])
    y_true2 = tf.placeholder(tf.float32, [None, h // size[1], w // size[1],
                                          num_anchors // 3, 5 + num_classes])
    y_true3 = tf.placeholder(tf.float32, [None, h // size[2], w // size[2],
                                          num_anchors // 3, 5 + num_classes])
    y_true = [y_true1, y_true2, y_true3]
    learning_rate = tf.placeholder(tf.float32, shape=[])
    y_predict = yolov3(image_input, num_anchors // 3, num_classes)
    loss = yolo_loss(y_predict, y_true, anchors, num_classes)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    return image_input, y_true1, y_true2, y_true3, loss, learning_rate

def create_tiny_yolov3(input_shape, anchors, num_classes):

    h, w = input_shape
    num_anchors = len(anchors)
    image_input = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    size = {0:32, 1:16, 2:8}
    y_true1 = tf.placeholder(tf.float32, [None, h//size[0], w//size[0],
                                          num_anchors//2, 5+num_classes])
    y_true2 = tf.placeholder(tf.float32, [None, h//size[1], w//size[1],
                                          num_anchors//2, 5+num_classes])
    y_true = [y_true1, y_true2]
    learning_rate = tf.placeholder(tf.float32, shape=[])
    y_predict = tiny_yolov3(image_input, num_anchors//2, num_classes)
    loss = yolo_loss(y_predict, y_true, anchors, num_classes)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    return image_input, y_true1, y_true2, loss, learning_rate

def conv2d(x, num_outputs, kernel_size, stride=(1, 1)):

    return layers.conv2d(x, num_outputs, kernel_size, stride=stride, activation_fn=None,
                      weights_regularizer=regularizers.l2_regularizer(5e-4), biases_initializer=None)

def conv2d_bn_leaky(x, num_outputs, kernel_size, stride=(1, 1)):
    '''Convolution2D followed by BatchNormalization and LeakyReLU.'''
    x = conv2d(x, num_outputs, kernel_size, stride=stride)
    x = layers.batch_norm(x)
    x = tf.nn.leaky_relu(x, alpha=0.1)
    return x

def max_pool2d(x, kernel_size, stride=2):

    return layers.max_pool2d(x, kernel_size, stride, padding='same')

def up_sample(x, size=2):
    '''Upsampling layer for 2D inputs.'''
    h = x.shape[1]
    w = x.shape[2]
    return tf.image.resize_nearest_neighbor(x, [h*size, w*size])

def concatenate(x1, x2):
    return tf.concat([x1, x2], axis=-1)

def residual_block(x, num_output, num_block):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    x = conv2d_bn_leaky(x, num_output, (3, 3), stride=(2, 2))
    for i in range(num_block):
        y = conv2d_bn_leaky(x, num_output//2, (1, 1))
        y = conv2d_bn_leaky(y, num_output, (3, 3))
        x = tf.add(x, y)
    return x

def make_last_layers(x, num_outputs, num_anchors, num_classes):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = conv2d_bn_leaky(x, num_outputs, (1, 1))
    x = conv2d_bn_leaky(x, num_outputs*2, (3, 3))
    x = conv2d_bn_leaky(x, num_outputs, (1, 1))
    x = conv2d_bn_leaky(x, num_outputs*2, (3, 3))
    x = conv2d_bn_leaky(x, num_outputs, (1, 1))

    y = conv2d_bn_leaky(x, num_outputs*2, (3, 3))
    y = conv2d(y, num_anchors*(num_classes+5), (1, 1))

    return x, y

def yolov3(image_input, num_anchors, num_classes):
    '''Create YOLO_V3 model CNN body in Tensorflow.'''
    x = conv2d_bn_leaky(image_input, 32, (3, 3))
    x = residual_block(x, 64, 1)
    x = residual_block(x, 128, 2)
    x1 = residual_block(x, 256, 8)
    x2 = residual_block(x1, 512, 8)
    x3 = residual_block(x2, 1024, 4)

    x, y1 = make_last_layers(x3, 512, num_anchors, num_classes)
    x = conv2d_bn_leaky(x, 256, (1, 1))
    x = up_sample(x)
    x = concatenate(x, x2)

    x, y2 = make_last_layers(x, 256, num_anchors, num_classes)
    x = conv2d_bn_leaky(x, 128, (1, 1))
    x = up_sample(x)
    x = concatenate(x, x1)

    x, y3 = make_last_layers(x, 128, num_anchors, num_classes)
    return [y1, y2, y3]

def tiny_yolov3(image_input, num_anchors, num_classes):
    '''Create Tiny YOLO_V3 model CNN body in Tensorflow.'''
    x1 = conv2d_bn_leaky(image_input, 16, (3, 3))
    x1 = max_pool2d(x1, (2, 2))
    x1 = conv2d_bn_leaky(x1, 32, (3, 3))
    x1 = max_pool2d(x1, (2, 2))
    x1 = conv2d_bn_leaky(x1, 64, (3, 3))
    x1 = max_pool2d(x1, (2, 2))
    x1 = conv2d_bn_leaky(x1, 128, (3, 3))
    x1 = max_pool2d(x1, (2, 2))
    x1 = conv2d_bn_leaky(x1, 256, (3, 3))

    x2 = max_pool2d(x1, (2, 2))
    x2 = conv2d_bn_leaky(x2, 512, (3, 3))
    x2 = max_pool2d(x2, (2, 2), stride=1)
    x2 = conv2d_bn_leaky(x2, 1024, (3, 3))
    x2 = conv2d_bn_leaky(x2, 256, (1, 1))

    y1 = conv2d_bn_leaky(x2, 512, (3, 3))
    y1 = conv2d(y1, num_anchors*(num_classes+5), (1, 1))

    x2 = conv2d_bn_leaky(x2, 128, (1, 1))
    x2 = up_sample(x2)

    y2 = concatenate(x1, x2)
    y2 = conv2d_bn_leaky(y2, 256, (3, 3))
    y2 = conv2d(y2, num_anchors*(num_classes+5), (1, 1))
    return [y1, y2]

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, tf.float32)
    image_shape = tf.cast(image_shape, tf.float32)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]] # default setting
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)

    return boxes_, scores_, classes_

def yolo_head(yolo_output, anchors, num_classes, input_shape, clac_loss=False):
    num_anchors = len(anchors)
    anchor_tensor = tf.reshape(tf.constant(anchors, tf.float32), [1, 1, 1, num_anchors, 2])
    grid_shape = tf.shape(yolo_output)[1:3]

    grid_y = tf.tile(tf.reshape(tf.range(grid_shape[0]), [-1, 1, 1, 1]), multiples=[1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_shape[1]), [1, -1, 1, 1]), multiples=[grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, tf.float32)

    feat = tf.reshape(yolo_output, [-1, grid_shape[0], grid_shape[1], num_anchors, 5+num_classes])
    box_xy = (tf.nn.sigmoid(feat[..., 0:2]) + grid) / tf.cast(grid_shape[::-1], tf.float32)
    box_wh = (tf.exp(feat[..., 2:4]) * anchor_tensor) / tf.cast(input_shape[::-1], tf.float32)
    box_confidence = tf.nn.sigmoid(feat[..., 4:5])
    box_class_prob = tf.nn.sigmoid(feat[..., 5:])
    if clac_loss:
        return grid, feat, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_prob

def box_iou(b1, b2):
    b1 = tf.expand_dims(b1, axis=-2)
    b1_xy = b1[..., 0:2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_min = b1_xy - b1_wh_half
    b1_max = b1_xy + b1_wh_half

    b2 = tf.expand_dims(b2, axis=0)
    b2_xy = b2[..., 0:2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_min = b2_xy - b2_wh_half
    b2_max = b2_xy + b2_wh_half

    intersect_min = tf.maximum(b1_min, b2_min)
    intersect_max = tf.minimum(b1_max, b2_max)
    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)
    intersect_area = intersect_wh[..., 0]*intersect_wh[..., 1]
    b1_area = b1_wh[..., 0]*b1_wh[..., 1]
    b2_area = b2_wh[..., 0]*b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou

def yolo_loss(y_predict, y_true, anchors, num_classes, ignore_thresh=.5):
    '''Return yolo_loss tensor.'''
    num_layers = len(anchors)//3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
    input_shape = tf.cast(tf.shape(y_predict[0])[1:3] * 32, tf.float32)
    grid_shape = [tf.cast(tf.shape(y_predict[l])[1:3], tf.float32) for l in range(num_layers)]
    loss = 0
    m = tf.shape(y_predict[0])[0]
    mf = tf.cast(m, dtype=tf.float32)
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_label = y_true[l][..., 5:]
        grid, pred, box_xy, box_wh = yolo_head(y_predict[l], anchors[anchor_mask[l]], num_classes, input_shape, clac_loss=True)
        pred_box = tf.concat([box_xy, box_wh], axis=-1)
        true_xy = y_true[l][..., 0:2]*grid_shape[l][::-1] - grid
        true_wh = y_true[l][..., 2:4]*input_shape[::-1] / anchors[anchor_mask[l]]
        true_wh = tf.log(tf.where(tf.equal(true_wh, 0), tf.ones_like(true_wh), true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][..., 3:4]

        ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, tf.bool)
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou<ignore_thresh, tf.float32))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

        xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_xy, logits=pred[..., 0:2])
        wh_loss = object_mask * box_loss_scale * 0.5* tf.square(true_wh-pred[..., 2:4])
        confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred[..., 4:5]) + \
                          (1-object_mask)*ignore_mask*tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred[..., 4:5])
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_class_label, logits=pred[..., 5:])

        xy_loss = tf.reduce_sum(xy_loss) / mf
        wh_loss = tf.reduce_sum(wh_loss) / mf
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
    return loss
