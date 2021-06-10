from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../../../detector_in_keras")


import numpy as np
import tensorflow as tf
from models.faster_rcnn.bbox_ops import bbox_overlaps_tf, bbox_transform_tf

class GenerateAnchors(tf.keras.layers.Layer):
    def __init__(self, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        super(GenerateAnchors, self).__init__()
        self.feat_stride = feat_stride
        self.anchors_shift = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))

    def call(self, height, width):
        shift_x = tf.multiply(tf.range(width, name='range_shift_x'), self.feat_stride)
        # height
        shift_y = tf.multiply(tf.range(height,name='range_shift_y'), self.feat_stride)
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y, name="meshgrid_x_y")
        sx = tf.reshape(shift_x, shape=(-1,), name='reshape_sx')
        sy = tf.reshape(shift_y, shape=(-1,), name='reshape_sy')
        xyxy = tf.stack([sx, sy, sx, sy], name='stack_xyxy')
        shifts = tf.transpose(xyxy, name='transpose_shifts')
        K = tf.multiply(width, height, name='multi_w_h')
        shifts_reshape = tf.reshape(shifts, shape=[1, K, 4], name='shifts_reshape')
        shifts = tf.transpose(shifts_reshape, perm=(1, 0, 2))

        # anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
        A = self.anchors_shift.shape[0]
        anchor_constant = tf.reshape(self.anchors_shift, (1, A, 4), name='anchor_constant')
        anchor_constant = tf.cast(anchor_constant, dtype=tf.int32, name='anchor_constant_cast')
        # anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

        length = tf.multiply(K, A, name='length')
        anchors_add_shifts = tf.add(anchor_constant, shifts, name='anchors_add_shifts')
        anchors_tf = tf.reshape(anchors_add_shifts, shape=(length, 4), name='anchors_tf')
        anchors_tf_cast = tf.cast(anchors_tf, dtype=tf.float32, name='anchors_tf_cast')

        # return shift_x, shift_x
        return anchors_tf_cast, length

def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ 预生成图片检测边框

    :param height:
    :param width:
    :param feat_stride:
    :param anchor_scales:
    :param anchor_ratios:
    :return:
    """

    # width
    shift_x = tf.range(width, name='range_shift_x') * feat_stride
    # height
    shift_y = tf.range(height,name='range_shift_y') * feat_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y, name="meshgrid_x_y")
    sx = tf.reshape(shift_x, shape=(-1,), name='reshape_sx')
    sy = tf.reshape(shift_y, shape=(-1,), name='reshape_sy')
    xyxy = tf.stack([sx, sy, sx, sy], name='stack_xyxy')
    shifts = tf.transpose(xyxy, name='transpose_shifts')
    K = tf.multiply(width, height, name='multi_w_h')
    shifts_reshape = tf.reshape(shifts, shape=[1, K, 4], name='shifts_reshape')
    shifts = tf.transpose(shifts_reshape, perm=(1, 0, 2))

    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    anchor_constant = tf.reshape(anchors, (1, A, 4), name='anchor_constant')
    anchor_constant = tf.cast(anchor_constant, dtype=tf.int32, name='anchor_constant_cast')
    # anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

    length = tf.multiply(K, A, name='length')
    anchors_add_shifts = tf.add(anchor_constant, shifts, name='anchors_add_shifts')
    anchors_tf = tf.reshape(anchors_add_shifts, shape=(length, 4), name='anchors_tf')
    anchors_tf_cast = tf.cast(anchors_tf, dtype=tf.float32, name='anchors_tf_cast')

    # return shift_x, shift_x
    return anchors_tf_cast, length


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16,
                     ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

class AnchorTargetLayer(tf.keras.layers.Layer):
    def __init__(self, num_anchors=9, rpn_negative_overlap=0.3, rpn_positive_overlap=0.7, rpn_fg_fraction=0.5,
                 rpn_batchsize=256, rpn_bbox_inside_weights=(1.0, 1.0, 1.0, 1.0), rpn_positive_weight=-1):
        super(AnchorTargetLayer, self).__init__()
        self.num_anchors = num_anchors
        self.rpn_negative_overlap = rpn_negative_overlap
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_fg_fraction = rpn_fg_fraction
        self.rpn_batchsize = rpn_batchsize
        self.rpn_bbox_inside_weights = rpn_bbox_inside_weights
        self.rpn_positive_weight = rpn_positive_weight

    def random_disable_labels(self, labels_input, inds, disable_nums):
        shuffle_fg_inds = tf.random.shuffle(inds)
        disable_inds = shuffle_fg_inds[:disable_nums]
        disable_inds_expand_dim = tf.expand_dims(disable_inds, axis=1)
        neg_ones = tf.ones_like(disable_inds, dtype=tf.float32) * -1.
        return tf.tensor_scatter_nd_update(labels_input, disable_inds_expand_dim, neg_ones)

    def unmap(self, data, count, inds, fill, type):
        """ Unmap a subset of item (data) back to the original set of items (of
        size count) """
        if type == 'labels':
            ret = tf.zeros((count,), dtype=tf.float32, name="unmap_" + type)
            ret += fill
            inds_expand = tf.expand_dims(inds, axis=1)
            return tf.tensor_scatter_nd_update(ret, inds_expand, data)
        else:
            ret = tf.zeros(tf.concat([[count, ], tf.shape(data)[1:]], axis=0), dtype=tf.float32, name="unmap_" + type)
            ret += fill
            inds_expand = tf.expand_dims(inds, axis=1)
            return tf.tensor_scatter_nd_update(ret, inds_expand, data)

    def call(self, rpn_cls_score, gt_boxes, im_info, all_anchors):
        """Same as the anchor target layer in original Fast/er RCNN """
        A = self.num_anchors
        total_anchors = tf.shape(all_anchors)[0]
        # allow boxes to sit over the edge by a small amount
        _allowed_border = 0

        # map of shape (..., H, W, C)
        height = tf.shape(rpn_cls_score)[1]
        width = tf.shape(rpn_cls_score)[2]

        # only keep anchors inside the image
        inds_inside = tf.reshape(tf.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < (im_info[1] + _allowed_border)) &  # width
            (all_anchors[:, 3] < (im_info[0] + _allowed_border))  # height
        ), shape=(-1,))
        # keep only inside anchors
        anchors = tf.gather(all_anchors, inds_inside)
        # label: 1 is positive, 0 is negative, -1 is don't care
        labels = tf.zeros_like(inds_inside, dtype=tf.float32)
        labels -= 1.
        ones = tf.ones_like(inds_inside, dtype=tf.float32)
        zeros = tf.zeros_like(inds_inside, dtype=tf.float32)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps_tf(anchors, gt_boxes[:, :4])
        # 获取每个anchor跟那个gt_box的overlap最大
        argmax_overlaps = tf.cast(tf.argmax(overlaps, axis=1), dtype=tf.int32)
        # print(tf.range(tf.shape(overlaps)), tf.shape(argmax_overlaps))
        argmax_gather_nd_inds = tf.stack([tf.range(tf.shape(overlaps)[0]), argmax_overlaps], axis=1)
        max_overlaps = tf.gather_nd(overlaps, argmax_gather_nd_inds)

        # 获取每个gt_box跟哪个anchor的overlap最大，这里直接将该anchor当成了gt_box
        gt_argmax_overlaps = tf.cast(tf.argmax(overlaps, axis=0), dtype=tf.int32)
        max_overlaps_gather_nd_inds = tf.stack([gt_argmax_overlaps, tf.range(tf.shape(overlaps)[1])], axis=1)
        gt_max_overlaps = tf.gather_nd(overlaps, max_overlaps_gather_nd_inds)
        gt_argmax_overlaps = tf.where(overlaps == gt_max_overlaps)[:, 0]

        labels = tf.where(max_overlaps < self.rpn_negative_overlap, zeros, labels)

        # fg label: for each gt, anchor with highest overlap
        unique_gt_argmax_overlaps = tf.unique(gt_argmax_overlaps)[0]
        highest_fg_label = tf.gather(labels, unique_gt_argmax_overlaps) * -1.
        highest_gt_row_ids_expand_dim = tf.expand_dims(unique_gt_argmax_overlaps, axis=1)
        labels = tf.tensor_scatter_nd_update(labels, highest_gt_row_ids_expand_dim, highest_fg_label)

        # fg label: above threshold IOU
        labels = tf.where(max_overlaps >= self.rpn_positive_overlap, ones, labels)

        # subsample positive labels if we have too many
        num_fg = int(self.rpn_fg_fraction * self.rpn_batchsize)
        fg_inds = tf.reshape(tf.where(labels == 1), shape=(-1,))
        fg_inds_num = tf.shape(fg_inds)[0]

        fg_flag = tf.cast(fg_inds_num > num_fg, dtype=tf.float32)
        labels = fg_flag * self.random_disable_labels(labels, fg_inds, fg_inds_num - num_fg) + \
                 (1.0 - fg_flag) * labels

        # subsample negative labels if we have too many
        num_bg = self.rpn_batchsize - tf.shape(tf.where(labels == 1))[0]
        # bg_inds = np.where(labels == 0)[0]
        bg_inds = tf.reshape(tf.where(labels == 0), shape=(-1,))
        bg_inds_num = tf.shape(bg_inds)[0]
        bg_flag = tf.cast(bg_inds_num > num_bg, dtype=tf.float32)
        labels = bg_flag * self.random_disable_labels(labels, bg_inds, bg_inds_num - num_bg) + \
                 (1.0 - bg_flag) * labels

        # 此处将每个anchor与gt_box对准，gt_box与anchor的dx,dy,dw,dh，用来与模型预测的box计算损失
        bbox_targets = bbox_transform_tf(anchors, tf.gather(gt_boxes, argmax_overlaps, axis=0)[:, :4])

        # bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights = tf.zeros((tf.shape(inds_inside)[0], 4), dtype=tf.float32, name='bbox_inside_weights')
        # only the positive ones have regression targets
        bbox_inside_inds = tf.reshape(tf.where(labels == 1), shape=[-1, ])
        bbox_inside_inds_weights = tf.gather(bbox_inside_weights, bbox_inside_inds) + self.rpn_bbox_inside_weights
        bbox_inside_inds_expand = tf.expand_dims(bbox_inside_inds, axis=1)
        bbox_inside_weights = tf.tensor_scatter_nd_update(bbox_inside_weights,
                                                          bbox_inside_inds_expand,
                                                          bbox_inside_inds_weights)

        # bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_outside_weights = tf.zeros((tf.shape(inds_inside)[0], 4), dtype=tf.float32, name='bbox_outside_weights')
        if self.rpn_positive_weight < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = tf.reduce_sum(tf.cast(labels >= 0, dtype=tf.float32))
            positive_weights = tf.ones((1, 4), dtype=tf.float32) / num_examples
            negative_weights = tf.ones((1, 4), dtype=tf.float32) / num_examples

        else:
            assert ((self.rpn_positive_weight > 0) & (self.rpn_positive_weight < 1))
            positive_weights = self.rpn_positive_weight / tf.reduce_sum(tf.cast(labels == 1, dtype=tf.float32))
            negative_weights = (1.0 - self.rpn_positive_weight) / tf.reduce_sum(tf.cast(labels == 0, dtype=tf.float32))

        bbox_outside_positive_inds = bbox_inside_inds
        bbox_outside_negative_inds = tf.reshape(tf.where(labels == 0), shape=[-1, ])
        bbox_outside_positive_inds_weights = tf.gather(bbox_outside_weights, bbox_outside_positive_inds) + positive_weights
        bbox_outside_negative_inds_weights = tf.gather(bbox_outside_weights, bbox_outside_negative_inds) + negative_weights
        bbox_outside_positive_inds_expand = tf.expand_dims(bbox_outside_positive_inds, axis=1)
        bbox_outside_negative_inds_expand = tf.expand_dims(bbox_outside_negative_inds, axis=1)
        bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                           bbox_outside_positive_inds_expand,
                                                           bbox_outside_positive_inds_weights)
        bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                           bbox_outside_negative_inds_expand,
                                                           bbox_outside_negative_inds_weights)

        # 这里把上面处理完的目标anchors,labels,boxes,weights的size处理成一开始传进来的大小
        labels = self.unmap(labels, total_anchors, inds_inside, fill=-1, type='labels')
        bbox_targets = self.unmap(bbox_targets, total_anchors, inds_inside, fill=0, type='bbox_targets')
        bbox_inside_weights = self.unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0, type='bbox_inside_weights')
        bbox_outside_weights = self.unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0, type='bbox_outside_weights')

        # labels, 这里reshape成anchor的数目, anchor的总数等于(原图宽/16 * 原图高/16 * 9)
        rpn_labels = tf.reshape(labels, (1, height, width, A))
        rpn_bbox_targets = tf.reshape(bbox_targets, (1, height, width, A * 4), name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.reshape(bbox_inside_weights, (1, height, width, A * 4), name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.reshape(bbox_outside_weights, (1, height, width, A * 4),
                                              name='rpn_bbox_outside_weights')
        rpn_labels = tf.cast(rpn_labels, dtype=tf.int32)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, all_anchors, num_anchors,
                        rpn_negative_overlap=0.3, rpn_positive_overlap=0.7, rpn_fg_fraction=0.5,
                        rpn_batchsize=256, rpn_bbox_inside_weights=(1.0, 1.0, 1.0, 1.0),
                        rpn_positive_weight=-1):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    total_anchors = tf.shape(all_anchors)[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W, C)
    height = tf.shape(rpn_cls_score)[1]
    width = tf.shape(rpn_cls_score)[2]

    # only keep anchors inside the image
    inds_inside = tf.reshape(tf.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < (im_info[1] + _allowed_border)) &  # width
        (all_anchors[:, 3] < (im_info[0] + _allowed_border))  # height
    ), shape=(-1,))
    # keep only inside anchors
    anchors = tf.gather(all_anchors, inds_inside)

    # label: 1 is positive, 0 is negative, -1 is don't care
    labels = tf.zeros_like(inds_inside, dtype=tf.float32)
    labels -= 1.
    ones = tf.ones_like(inds_inside, dtype=tf.float32)
    zeros = tf.zeros_like(inds_inside, dtype=tf.float32)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps_tf(anchors, gt_boxes[:, :4])

    # 获取每个anchor跟那个gt_box的overlap最大
    argmax_overlaps = tf.cast(tf.argmax(overlaps, axis=1), dtype=tf.int32)
    argmax_gather_nd_inds = tf.stack([tf.range(tf.shape(overlaps)[0]), argmax_overlaps], axis=1)
    max_overlaps = tf.gather_nd(overlaps, argmax_gather_nd_inds)

    # 获取每个gt_box跟哪个anchor的overlap最大，这里直接将该anchor当成了gt_box
    gt_argmax_overlaps = tf.cast(tf.argmax(overlaps, axis=0), dtype=tf.int32)
    max_overlaps_gather_nd_inds = tf.stack([gt_argmax_overlaps, tf.range(tf.shape(overlaps)[1])], axis=1)
    gt_max_overlaps = tf.gather_nd(overlaps, max_overlaps_gather_nd_inds)
    gt_argmax_overlaps = tf.where(overlaps == gt_max_overlaps)[:, 0]

    labels = tf.where(max_overlaps < rpn_negative_overlap, zeros, labels)

    # fg label: for each gt, anchor with highest overlap
    unique_gt_argmax_overlaps = tf.unique(gt_argmax_overlaps)[0]
    highest_fg_label = tf.gather(labels, unique_gt_argmax_overlaps) * -1.
    highest_gt_row_ids_expand_dim = tf.expand_dims(unique_gt_argmax_overlaps, axis=1)
    labels = tf.tensor_scatter_nd_update(labels, highest_gt_row_ids_expand_dim, highest_fg_label)

    # fg label: above threshold IOU
    labels = tf.where(max_overlaps >= rpn_positive_overlap, ones, labels)

    # subsample positive labels if we have too many
    num_fg = int(rpn_fg_fraction * rpn_batchsize)
    fg_inds = tf.reshape(tf.where(labels == 1), shape=(-1,))
    fg_inds_num = tf.shape(fg_inds)[0]

    def random_disable_labels(labels_input, inds, disable_nums):
        shuffle_fg_inds = tf.random.shuffle(inds)
        disable_inds = shuffle_fg_inds[:disable_nums]
        disable_inds_expand_dim = tf.expand_dims(disable_inds, axis=1)
        neg_ones = tf.ones_like(disable_inds, dtype=tf.float32) * -1.
        return tf.tensor_scatter_nd_update(labels_input, disable_inds_expand_dim, neg_ones)

    fg_flag = tf.cast(fg_inds_num > num_fg, dtype=tf.float32)
    labels = fg_flag * random_disable_labels(labels, fg_inds, fg_inds_num - num_fg) + \
             (1.0 - fg_flag) * labels

    # subsample negative labels if we have too many
    num_bg = rpn_batchsize - tf.shape(tf.where(labels == 1))[0]
    # bg_inds = np.where(labels == 0)[0]
    bg_inds = tf.reshape(tf.where(labels == 0), shape=(-1,))
    bg_inds_num = tf.shape(bg_inds)[0]
    bg_flag = tf.cast(bg_inds_num > num_bg, dtype=tf.float32)
    labels = bg_flag * random_disable_labels(labels, bg_inds, bg_inds_num - num_bg) + \
             (1.0 - bg_flag) * labels

    # 此处将每个anchor与gt_box对准，gt_box与anchor的dx,dy,dw,dh，用来与模型预测的box计算损失
    bbox_targets = bbox_transform_tf(anchors, tf.gather(gt_boxes, argmax_overlaps, axis=0)[:, :4])

    # bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights = tf.zeros((tf.shape(inds_inside)[0], 4), dtype=tf.float32, name='bbox_inside_weights')
    # only the positive ones have regression targets
    bbox_inside_inds = tf.reshape(tf.where(labels == 1), shape=[-1, ])
    bbox_inside_inds_weights = tf.gather(bbox_inside_weights, bbox_inside_inds) + rpn_bbox_inside_weights
    bbox_inside_inds_expand = tf.expand_dims(bbox_inside_inds, axis=1)
    bbox_inside_weights = tf.tensor_scatter_nd_update(bbox_inside_weights,
                                                      bbox_inside_inds_expand,
                                                      bbox_inside_inds_weights)

    # bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_outside_weights = tf.zeros((tf.shape(inds_inside)[0], 4), dtype=tf.float32, name='bbox_outside_weights')
    if rpn_positive_weight < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = tf.reduce_sum(tf.cast(labels >= 0, dtype=tf.float32))
        positive_weights = tf.ones((1, 4), dtype=tf.float32) / num_examples
        negative_weights = tf.ones((1, 4), dtype=tf.float32) / num_examples

    else:
        assert ((rpn_positive_weight > 0) & (rpn_positive_weight < 1))
        positive_weights = rpn_positive_weight / tf.reduce_sum(tf.cast(labels == 1, dtype=tf.float32))
        negative_weights = (1.0 - rpn_positive_weight) / tf.reduce_sum(tf.cast(labels == 0, dtype=tf.float32))

    bbox_outside_positive_inds = bbox_inside_inds
    bbox_outside_negative_inds = tf.reshape(tf.where(labels == 0), shape=[-1, ])
    bbox_outside_positive_inds_weights = tf.gather(bbox_outside_weights, bbox_outside_positive_inds) + positive_weights
    bbox_outside_negative_inds_weights = tf.gather(bbox_outside_weights, bbox_outside_negative_inds) + negative_weights
    bbox_outside_positive_inds_expand = tf.expand_dims(bbox_outside_positive_inds, axis=1)
    bbox_outside_negative_inds_expand = tf.expand_dims(bbox_outside_negative_inds, axis=1)
    bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                       bbox_outside_positive_inds_expand,
                                                       bbox_outside_positive_inds_weights)
    bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                       bbox_outside_negative_inds_expand,
                                                       bbox_outside_negative_inds_weights)

    # 这里把上面处理完的目标anchors,labels,boxes,weights的size处理成一开始传进来的大小
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1, type='labels')
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0, type='bbox_targets')
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0, type='bbox_inside_weights')
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0, type='bbox_outside_weights')

    # labels, 这里reshape成anchor的数目, anchor的总数等于(原图宽/16 * 原图高/16 * 9)
    rpn_labels = tf.reshape(labels, (1, height, width, A))
    rpn_bbox_targets = tf.reshape(bbox_targets, (1, height, width, A * 4), name='rpn_bbox_targets')
    rpn_bbox_inside_weights = tf.reshape(bbox_inside_weights, (1, height, width, A * 4), name='rpn_bbox_inside_weights')
    rpn_bbox_outside_weights = tf.reshape(bbox_outside_weights, (1, height, width, A * 4),
                                          name='rpn_bbox_outside_weights')

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    # return labels, labels, labels, im_info

def _random_disable_labels(labels_input, inds, disable_nums):
    shuffle_fg_inds = tf.random.shuffle(inds)
    disable_inds = shuffle_fg_inds[:disable_nums]
    disable_inds_expand_dim = tf.expand_dims(disable_inds, axis=1)
    neg_ones = tf.ones_like(disable_inds, dtype=tf.float32) * -1.
    return tf.tensor_scatter_nd_update(labels_input, disable_inds_expand_dim, neg_ones)

def _unmap(data, count, inds, fill, type):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if type == 'labels':
        ret = tf.zeros((count,), dtype=tf.float32, name="unmap_" + type)
        ret += fill
        inds_expand = tf.expand_dims(inds, axis=1)
        return tf.tensor_scatter_nd_update(ret, inds_expand, data)
    else:
        ret = tf.zeros(tf.concat([[count, ], tf.shape(data)[1:]], axis=0), dtype=tf.float32, name="unmap_" + type)
        ret += fill
        inds_expand = tf.expand_dims(inds, axis=1)
        return tf.tensor_scatter_nd_update(ret, inds_expand, data)


if __name__ == '__main__':
    import cv2
    # from moviepy.editor import ImageSequenceClip
    # from data.visual_ops import draw_bounding_box, draw_point

    # im_file = "../../data/tmp/Cats_Test49.jpg"
    # im_file = "../../data/detect_data/JPEGImages/Cats_Test1.jpg"
    # im_file = "../../data/car_data/JPEGImages/2009_004567.jpg"
    # im = cv2.imread(im_file)
    # h, w, c = np.shape(im)
    # print(h, w, c)
    #
    # shift_x = tf.range(w/16) * 16
    # # height
    # shift_y = tf.range(h/16) * 16
    # shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    # sx = tf.reshape(shift_x, shape=(-1,))
    # sy = tf.reshape(shift_y, shape=(-1,))
    # points = tf.transpose(tf.stack([sx,sy]))
    # print(points)
    # im = draw_point(im, points=points, color=(0,0,255), size=2)
    # # cv2.imshow("anchor base point", im)
    # cv2.imwrite("../../data/tmp/anchor_base_points.jpg", im)

    # anchors, length = generate_anchors_pre_tf(int(88 / 16.), int(400 / 16.))
    # for i in anchors:
    #     print(i)

    anchors, length = GenerateAnchors()(height=int(88 / 16.), width=int(400 / 16.))
    for i in anchors:
        print(i)
    # overlaps = tf.reshape(bbox_overlaps_tf(anchors, np.array([[128,22.,240,222.]], dtype=np.float32)), shape=(-1,))
    # overlaps = bbox_overlaps_tf(anchors, np.array([[ 7., 23.,103., 84.],
    #                                                [363., 3.,379., 34.]], dtype=np.float32))
    # print(tf.where(overlaps > 0.7))
    # print(anchors, length)
    # print(overlaps)
    # max_overlaps = overlaps.numpy().max(axis=1)
    #
    # fg_inds = np.where(max_overlaps >= 0.5)[0]
    # # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    # bg_inds = np.where((max_overlaps < 0.5) &
    #                    (max_overlaps >= 0.1))[0]
    # # print(tf.where(overlaps > 0.5))
    # print(len(fg_inds), len(bg_inds))
    # print(tf.sort(max_overlaps, direction="DESCENDING")[:100])
    # target_anchors = list(filter(lambda x: x[0] > 0 and x[1] > 0 and x[2] < w and x[3] < h, anchors[10000:100000]))
    # img_names = []
    # for i in range(len(target_anchors)):
    #     if i % 10 == 0:
    #         box = target_anchors[i]
    #         im = draw_bounding_box(im.copy(),"","",box[0],box[1],box[2],box[3],thickness=1)
    #         img_name = "../../data/tmp/box_gif/{}.jpg".format(i)
    #         img_names.append(img_name)
    #         cv2.imwrite(img_name,im)
    # clip = ImageSequenceClip(img_names,fps=24)
    # clip.write_gif('../../data/tmp/box_add.gif')
    # cv2.imshow("1", im)
    # cv2.waitKey(0)
