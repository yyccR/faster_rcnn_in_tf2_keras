from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class ProposalTargetLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, use_gt=False, train_batch_size=256,
                 fg_fraction=0.5, train_fg_thresh=0.5, train_bg_thresh_hi=0.5, train_bg_thresh_lo=0.1,
                 train_bbox_normalize_targets_precomputed=True,
                 train_bbox_normalize_means=(0.0, 0.0, 0.0, 0.0),
                 train_bbox_normalize_stds=(0.1, 0.1, 0.2, 0.2), bbox_inside_weight=(1.0, 1.0, 1.0, 1.0)):
        super(ProposalTargetLayer, self).__init__()
        self.num_classes = num_classes
        self.use_gt = use_gt
        self.train_batch_size = train_batch_size
        self.fg_fraction = fg_fraction
        self.train_fg_thresh = train_fg_thresh
        self.train_bg_thresh_hi = train_bg_thresh_hi
        self.train_bg_thresh_lo = train_bg_thresh_lo
        self.train_bbox_normalize_targets_precomputed = train_bbox_normalize_targets_precomputed
        self.train_bbox_normalize_means = train_bbox_normalize_means
        self.train_bbox_normalize_stds = train_bbox_normalize_stds
        self.bbox_inside_weight = bbox_inside_weight

    def _get_bbox_regression_labels(self, bbox_target_data):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """

        clss = bbox_target_data[:, 0]
        bbox_targets = tf.zeros((tf.shape(clss)[0], 4 * self.num_classes), dtype=tf.float32, name="regression_bbox_targets")
        bbox_inside_weights = tf.zeros_like(bbox_targets, dtype=tf.float32)

        # inds = np.where(clss > 0)[0]
        inds = tf.cast(tf.reshape(tf.where(clss > 0), shape=(-1,)), dtype=tf.int32)

        cols = tf.cast(tf.gather(clss, inds),dtype=tf.int32)
        starts = tf.expand_dims(cols * 4, axis=1)
        starts_1 = starts + 1
        starts_2 = starts + 2
        starts_3 = starts + 3
        col_inds = tf.reshape(tf.concat([starts, starts_1, starts_2, starts_3], axis=1), (-1,))
        row_inds = tf.reshape(tf.tile(tf.expand_dims(inds, axis=1), [1, 4]), (-1,))
        row_col_inds = tf.concat([tf.expand_dims(row_inds, 1), tf.expand_dims(col_inds, 1)], axis=1)
        updates_target_data = tf.reshape(tf.gather(bbox_target_data[:, 1:], inds), (-1,))
        updates_inside_weight = tf.reshape(tf.tile(self.bbox_inside_weight, [tf.shape(inds)[0]]), (-1,))

        bbox_targets = tf.tensor_scatter_nd_update(bbox_targets, row_col_inds, updates_target_data)
        bbox_inside_weights = tf.tensor_scatter_nd_update(bbox_inside_weights, row_col_inds, updates_inside_weight)

        return bbox_targets, bbox_inside_weights

    def _compute_targets(self, ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""

        targets = bbox_transform_tf(ex_rois, gt_rois)
        if self.train_bbox_normalize_targets_precomputed:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = (targets - self.train_bbox_normalize_means) / self.train_bbox_normalize_stds
        labels_expand = tf.expand_dims(labels, axis=1)
        targets_add_labels = tf.concat([labels_expand, targets], axis=1)
        return targets_add_labels


    def _sample_rois(self, all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps_tf(
            tf.cast(all_rois[:, 1:5], dtype=tf.float32),
            tf.cast(gt_boxes[:, :4], dtype=tf.float32))
        gt_assignment = tf.argmax(overlaps, axis=1)
        max_overlaps = tf.keras.backend.max(overlaps, axis=1)
        labels = tf.reshape(tf.gather(gt_boxes, gt_assignment)[:, 4], shape=(-1,))

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = tf.reshape(tf.where(max_overlaps >= self.train_fg_thresh), shape=(-1,))
        bg_inds = tf.reshape(tf.where((max_overlaps < self.train_bg_thresh_hi) &
                                      (max_overlaps >= self.train_bg_thresh_lo)),
                             shape=(-1,))

        # # Small modification to the original version where we ensure a fixed number of regions are sampled
        fg_inds_nums = tf.shape(fg_inds)[0]

        # fg samples
        fg_rois_nums = tf.minimum(fg_rois_per_image, fg_inds_nums)
        fg_inds_selected = tf.random.shuffle(fg_inds)[:fg_rois_nums]
        # bg samples
        bg_rois_nums = rois_per_image - fg_rois_nums
        bg_inds_selected = tf.random.shuffle(bg_inds)[:bg_rois_nums]
        # total samples
        fg_bg_sample_inds = tf.concat([fg_inds_selected, bg_inds_selected], axis=0)
        samples_nums = tf.shape(fg_bg_sample_inds)[0]
        lack_nums = rois_per_image - samples_nums
        lack_samples_inds = tf.reshape(tf.random.categorical(
            logits=tf.expand_dims(tf.zeros(samples_nums, name="lack_samples_inds"), axis=0),
            num_samples=lack_nums), shape=(-1,))
        lack_samples_inds = tf.gather(fg_bg_sample_inds, lack_samples_inds)
        total_samples_inds = tf.concat([fg_bg_sample_inds, lack_samples_inds], axis=0)

        labels = tf.gather(labels, total_samples_inds)
        final_overlaps = tf.gather(overlaps, total_samples_inds)
        final_max_overlaps = tf.keras.backend.max(final_overlaps, axis=1)
        final_bg_inds = tf.reshape(
            tf.where((final_max_overlaps < self.train_bg_thresh_hi) & (final_max_overlaps >= self.train_bg_thresh_lo)),
            shape=(-1,))

        bg_labels = tf.zeros_like(final_bg_inds, dtype=tf.float32)
        labels_bg_inds_expand = tf.expand_dims(final_bg_inds, axis=1)
        labels = tf.tensor_scatter_nd_update(labels, labels_bg_inds_expand, bg_labels)

        rois = tf.gather(all_rois, total_samples_inds)
        roi_scores = tf.gather(all_scores, total_samples_inds)

        gt_boxes_inds = tf.gather(gt_assignment, total_samples_inds)

        bbox_target_data = self._compute_targets(
            ex_rois=rois[:, 1:5],
            gt_rois=tf.gather(gt_boxes, gt_boxes_inds)[:, :4],
            labels=labels)

        bbox_targets, bbox_inside_weights = self._get_bbox_regression_labels(bbox_target_data=bbox_target_data)

        return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

    def call(self, rpn_rois, rpn_scores, gt_boxes):
        """
        Assign object detection proposals to ground-truth targets. Produces proposal
        classification labels and bounding-box regression targets.
        """

        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = rpn_rois
        all_scores = rpn_scores

        # Include ground-truth boxes in the set of candidate rois
        if self.use_gt:
            zeros = tf.zeros((tf.shape(gt_boxes)[0], 1), dtype=tf.float32)
            all_rois = tf.concat([all_rois, tf.concat([zeros, gt_boxes[:, :-1]], axis=1)], axis=0)
            # not sure if it a wise appending, but anyway i am not using it
            all_scores = tf.concat([all_scores, zeros], axis=0)

        num_images = 1
        rois_per_image = int(self.train_batch_size / num_images)
        fg_rois_per_image = int(self.fg_fraction * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, roi_scores, bbox_targets, bbox_inside_weights = self._sample_rois(
            all_rois=all_rois,
            all_scores=all_scores,
            gt_boxes=gt_boxes,
            fg_rois_per_image=fg_rois_per_image,
            rois_per_image=rois_per_image)

        rois = tf.reshape(rois, shape=(-1, 5))
        roi_scores = tf.reshape(roi_scores, shape=(-1,))
        labels = tf.reshape(labels, shape=(-1,))
        bbox_targets = tf.reshape(bbox_targets, shape=(-1, self.num_classes * 4))
        bbox_inside_weights = tf.reshape(bbox_inside_weights, shape=(-1, self.num_classes * 4))
        bbox_outside_weights = tf.cast(bbox_inside_weights > 0, dtype=tf.float32)
        return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def intersection(boxlist1, boxlist2):
    """计算box之间的交叉面积
    :param boxlist1: N*4
    :param boxlist2: M*4
    Returns: N*M
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(
        value=boxlist1, num_or_size_splits=4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(
        value=boxlist2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin + 1.)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin + 1.)
    return intersect_heights * intersect_widths


def area(boxlist):
    """ 计算面积
    :param boxlist1: N*4.
    """
    x_min, y_min, x_max, y_max = tf.split(value=boxlist, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min + 1.) * (x_max - x_min + 1.), [1])


def bbox_overlaps_tf(boxlist1, boxlist2):
    """ 计算iou
    :param boxlist1: N*4
    :param boxlist2: M*4
    Returns: N*M
    """
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def bbox_transform_tf(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = tf.math.log(gt_widths / ex_widths)
    targets_dh = tf.math.log(gt_heights / ex_heights)

    targets = tf.stack([targets_dx, targets_dy, targets_dw, targets_dh], axis=1)
    return targets


def bbox_transform_inv_tf(boxes, deltas):
    boxes = tf.cast(boxes, deltas.dtype)
    widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    ctr_x = tf.add(boxes[:, 0], widths * 0.5)
    ctr_y = tf.add(boxes[:, 1], heights * 0.5)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def multi_bbox_transform_inv(boxes, deltas):

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes_tf(boxes, im_info):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def box_nms(boxes, thresh):
    """ 非极大抑制
    :param boxes: [[x_min, y_min, x_max, y_max, score], [], ...]
    :param thresh: 非极大抑制阈值
    :return: boxes
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    # nms_boxes = boxes[keep]
    return keep

if __name__ == "__main__":
    import numpy as np
    x = np.array([[128,22.,240,222.]], dtype=np.float32)
    y = np.array([[130,24.,220,220.]], dtype=np.float32)
    overlaps = bbox_overlaps_tf(x, y)
    print(overlaps)
