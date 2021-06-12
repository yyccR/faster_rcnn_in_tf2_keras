import sys

sys.path.append("../faster_rcnn_in_tf2_keras")

import gc
import os
import numpy as np
import tensorflow as tf
from bbox_ops import bbox_transform_inv_tf, multi_bbox_transform_inv, \
    clip_boxes_tf, clip_boxes, box_nms, ProposalTargetLayer
from anchors_ops import generate_anchors_pre_tf, GenerateAnchors, AnchorTargetLayer
from vgg16 import Vgg16
from data.generate_voc_data import DataGenerator
from data.visual_ops import draw_bounding_box

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ProposalLayer(tf.keras.layers.Layer):
    def __init__(self, rpn_post_nms_top_n, rpn_nms_threshold, num_anchors):
        super(ProposalLayer, self).__init__()
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_nms_threshold = rpn_nms_threshold
        self.num_anchors = num_anchors

    def call(self, rpn_cls_prob, rpn_bbox_pred, anchors, im_info):
        """ 预测的边框与anchors进行比对, 非极大抑制后输出最终目标边框, 这里边框处理成了[x1,y1,x2,y2] """

        # Get the scores and bounding boxes

        scores = tf.reshape(rpn_cls_prob, (-1, 2))[:,1]
        # scores = rpn_cls_prob[:, :, :, self.num_anchors:]
        # scores = tf.reshape(scores, shape=(-1,))
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

        proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
        proposals = clip_boxes_tf(proposals, im_info)

        # Non-maximal suppression
        indices = tf.image.non_max_suppression(boxes=proposals,
                                               scores=scores,
                                               max_output_size=self.rpn_post_nms_top_n,
                                               iou_threshold=self.rpn_nms_threshold)

        boxes = tf.gather(proposals, indices)
        boxes = tf.cast(boxes, tf.float32)
        scores = tf.gather(scores, indices)
        scores = tf.reshape(scores, shape=(-1, 1))

        # Only support single image as input
        batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
        blob = tf.concat([batch_inds, boxes], 1)

        return blob, scores

class CropPoolLayer(tf.keras.layers.Layer):
    def __init__(self, pool_size_after_rpn):
        self.pool_size_after_rpn = pool_size_after_rpn
        super(CropPoolLayer, self).__init__()

    def call(self, bottom, rois, im_info):
        """ 裁剪层, 对卷积网络层输出的特征, 根据rpn层输出的roi进行裁剪, 且resize到统一的大小

        :return [bbox_nums, pre_pool_size, pre_pool_size, depth]
        """
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1]), [1])
        height = im_info[0]
        width = im_info[1]

        x1 = tf.expand_dims(rois[:, 1] / width, 1)
        y1 = tf.expand_dims(rois[:, 2] / height, 1)
        x2 = tf.expand_dims(rois[:, 3] / width, 1)
        y2 = tf.expand_dims(rois[:, 4] / height, 1)
        # Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.concat([y1, x1, y2, x2], axis=1)
        pre_pool_size = self.pool_size_after_rpn * 2
        # [bbox_nums, pre_pool_size, pre_pool_size, depth]
        crops = tf.image.crop_and_resize(image=bottom,
                                         boxes=bboxes,
                                         box_indices=tf.cast(batch_ids, dtype=tf.int32),
                                         crop_size=[pre_pool_size, pre_pool_size])

        return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='SAME')(crops)


class FasterRCNN:
    def __init__(self,
                 classes,
                 base_model=Vgg16(),
                 im_input_shape=(None, None, 3),
                 gt_box_input_shape=(None, 5),
                 batch_size=1,
                 rpn_channels=512,
                 rpn_post_nms_top_n=2000,
                 rpn_nms_threshold=0.7,
                 feat_stride=[16, ],
                 anchor_scales=(8, 16, 32),
                 anchor_ratios=(0.5, 1, 2),
                 rpn_negative_overlap=0.3,
                 rpn_positive_overlap=0.5,
                 rpn_fg_fraction=0.5,
                 rpn_batchsize=256,
                 rpn_bbox_inside_weights=(1.0, 1.0, 1.0, 1.0),
                 rpn_positive_weight=-1,
                 use_gt=False,
                 train_roi_batch_size=256,
                 fg_fraction=0.5,
                 train_fg_thresh=0.5,
                 train_bg_thresh_hi=0.5,
                 train_bg_thresh_lo=0.,
                 train_bbox_normalize_targets_precomputed=True,
                 train_bbox_normalize_means=(0.0, 0.0, 0.0, 0.0),
                 train_bbox_normalize_stds=(0.1, 0.1, 0.2, 0.2),
                 bbox_inside_weight=(1.0, 1.0, 1.0, 1.0),
                 pool_size_after_rpn=7,
                 smooth_l1_rpn_sigma=3.0,
                 smooth_l1_rcnn_sigma=1.0,
                 pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]])):
        self.gt_boxes = []
        self.im_info = []
        self.anchors = []
        self.anchors_length = 0
        self.base_model = base_model
        self.classes = classes
        self.im_input_shape = im_input_shape
        self.gt_box_input_shape = gt_box_input_shape
        self.batch_size = batch_size
        self.num_classes = len(self.classes)
        self.rpn_channels = rpn_channels
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_nms_threshold = rpn_nms_threshold
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.rpn_negative_overlap = rpn_negative_overlap
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_fg_fraction = rpn_fg_fraction
        self.rpn_batchsize = rpn_batchsize
        self.rpn_bbox_inside_weights = rpn_bbox_inside_weights
        self.rpn_positive_weight = rpn_positive_weight
        self.use_gt = use_gt
        self.train_roi_batch_size = train_roi_batch_size
        self.fg_fraction = fg_fraction
        self.train_fg_thresh = train_fg_thresh
        self.train_bg_thresh_hi = train_bg_thresh_hi
        self.train_bg_thresh_lo = train_bg_thresh_lo
        self.train_bbox_normalize_targets_precomputed = train_bbox_normalize_targets_precomputed
        self.train_bbox_normalize_means = train_bbox_normalize_means
        self.train_bbox_normalize_stds = train_bbox_normalize_stds
        self.bbox_inside_weight = bbox_inside_weight
        self.pool_size_after_rpn = pool_size_after_rpn
        self.smooth_l1_rpn_sigma = smooth_l1_rpn_sigma
        self.smooth_l1_rcnn_sigma = smooth_l1_rcnn_sigma
        self.pixel_mean = pixel_mean

    def _region_proposal_network(self, conv_net, anchors, gt_boxes, im_info, is_training):
        """ rpn网络, 对上个卷积网络输出的特征层做 类别预测和边框预测 """
        anchor_targets = {}
        predictions = {}
        proposal_targets = {}

        # 共享层卷积
        rpn = tf.keras.layers.Conv2D(filters=self.rpn_channels,
                                     kernel_size=(3, 3),
                                     padding='SAME',
                                     kernel_regularizer='l2')(conv_net)

        # 类别预测
        rpn_cls_score = tf.keras.layers.Conv2D(filters=self.num_anchors * 2,
                                               kernel_size=(1, 1),
                                               padding='VALID',
                                               kernel_regularizer='l2')(rpn)
        # reshape成2个通道
        rpn_cls_score_reshape = tf.reshape(rpn_cls_score, (-1, 2))
        # 对通道层做归一化, 保证类别预测的概率之和为1
        rpn_cls_prob_reshape = tf.keras.layers.Softmax()(rpn_cls_score_reshape)
        # 取最终的类别和概率
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        # 转换回[1, h, w, 9*2]
        rpn_cls_prob = tf.reshape(rpn_cls_prob_reshape, tf.shape(rpn_cls_score))

        # 边框预测
        rpn_bbox_pred = tf.keras.layers.Conv2D(filters=self.num_anchors * 4,
                                               kernel_size=(1, 1),
                                               padding='VALID',
                                               kernel_regularizer='l2')(rpn)

        if is_training:
            # 预测的边框与anchors进行比对, 非极大抑制后输出最终目标边框[[0, x1, y1, x2, y2],...]及其分值
            rois, roi_scores = ProposalLayer(rpn_post_nms_top_n=self.rpn_post_nms_top_n,
                                             rpn_nms_threshold=self.rpn_nms_threshold,
                                             num_anchors=self.num_anchors)(
                rpn_cls_prob=rpn_cls_prob,
                rpn_bbox_pred=rpn_bbox_pred,
                anchors=anchors,
                im_info=im_info
            )

            # 生成的anchor与gt_box比对, 输出前景anchor和背景anchor的label
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                AnchorTargetLayer(
                    num_anchors=self.num_anchors,
                    rpn_negative_overlap=self.rpn_negative_overlap,
                    rpn_positive_overlap=self.rpn_positive_overlap,
                    rpn_fg_fraction=self.rpn_fg_fraction,
                    rpn_batchsize=self.rpn_batchsize,
                    rpn_bbox_inside_weights=self.rpn_bbox_inside_weights,
                    rpn_positive_weight=self.rpn_positive_weight
                )(
                    rpn_cls_score=rpn_cls_score,
                    all_anchors=anchors,
                    gt_boxes=gt_boxes[0],
                    im_info=im_info
                )

            # roi采样, 再基于roi[0,x1,y1,x2,y2]计算bbox_targets[dx,dy,dw,dh]
            with tf.control_dependencies([rpn_labels]):
                rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = ProposalTargetLayer(
                    num_classes=self.num_classes,
                    use_gt=self.use_gt,
                    train_batch_size=self.train_roi_batch_size,
                    fg_fraction=self.fg_fraction,
                    train_fg_thresh=self.train_fg_thresh,
                    train_bg_thresh_hi=self.train_bg_thresh_hi,
                    train_bg_thresh_lo=self.train_bg_thresh_lo,
                    train_bbox_normalize_targets_precomputed=self.train_bbox_normalize_targets_precomputed,
                    train_bbox_normalize_means=self.train_bbox_normalize_means,
                    train_bbox_normalize_stds=self.train_bbox_normalize_stds,
                    bbox_inside_weight=self.bbox_inside_weight
                )(
                    rpn_rois=rois,
                    rpn_scores=roi_scores,
                    gt_boxes=gt_boxes[0],
                )

            anchor_targets['rpn_labels'] = rpn_labels
            # [1,height,width, 9*4]
            anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            # [1,height,width, 9*4]
            anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            # [1,height,width, 9*4]
            anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            # [256, 1]
            proposal_targets['labels'] = labels
            # [256, 4 * num_classes]
            proposal_targets['bbox_targets'] = bbox_targets
            # [256, 4 * num_classes]
            proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            # [256, 4 * num_classes]
            proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            # [1, h, w, 9*2]
            predictions["rpn_cls_score"] = rpn_cls_score
            # [1, h*9, w, 2]
            predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            predictions["rpn_cls_prob_reshape"] = rpn_cls_prob_reshape
            # [1, h, w, 9*2]
            predictions["rpn_cls_prob"] = rpn_cls_prob
            # [h*w*9, 1]
            predictions["rpn_cls_pred"] = rpn_cls_pred
            # [1, h, w, 9*4]
            predictions["rpn_bbox_pred"] = rpn_bbox_pred
            # [256, 5]
            predictions["rois"] = rois

            return rois, roi_scores, labels, anchor_targets, proposal_targets, predictions

        else:
            # 预测的边框与anchors进行比对, 非极大抑制后输出最终目标边框[[idx, x1, y1, x2, y2],...]及其分值
            rois, roi_scores = ProposalLayer(rpn_post_nms_top_n=self.rpn_post_nms_top_n,
                                             rpn_nms_threshold=self.rpn_nms_threshold,
                                             num_anchors=self.num_anchors)(
                rpn_cls_prob=rpn_cls_prob,
                rpn_bbox_pred=rpn_bbox_pred,
                anchors=anchors,
                im_info=im_info
            )
            return rois, roi_scores

    def _region_classification(self, fc7):
        """ 预测最终每个roi的类别概率, 边框bbox

        :param fc7:
        """
        cls_score = tf.keras.layers.Dense(units=self.num_classes, kernel_regularizer='l2')(fc7)
        cls_prob = tf.keras.layers.Softmax(name='cls_prob')(cls_score)
        # cls_pred = tf.argmax(cls_score, axis=1)
        bbox_pred = tf.keras.layers.Dense(units=self.num_classes * 4, kernel_regularizer='l2')(fc7)

        return cls_score, cls_prob, bbox_pred

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma, dim):
        """ 计算bbox损失, fast-rcnn论文有详细说明
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf"""

        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        # print(tf.shape(bbox_pred),tf.shape(bbox_targets),tf.shape(bbox_inside_weights),tf.shape(bbox_outside_weights))
        abs_in_box_diff = tf.abs(in_box_diff)
        # smooth_l1 = 0.5 * x² if |x| < 1
        # smooth_l1 = |x| - 0.5 if |x| ≥ 1
        smoothL1_sign = tf.cast(tf.less(abs_in_box_diff, 1. / sigma_2), dtype=tf.float32)
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        # 每个batch计算sum损失和, 再对不同batch平均损失
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
        return loss_box

    def compute_losses(self, anchor_targets, proposal_targets, predictions):

        # 以下class-loss, bbox-loss为第一次预测损失, 即rpn网络的预测结果
        # RPN, class loss
        rpn_cls_score = tf.reshape(predictions['rpn_cls_score_reshape'], (-1, 2))
        rpn_label = tf.reshape(anchor_targets['rpn_labels'], (-1,))
        rpn_select = tf.where(rpn_label != -1)
        # 获取label不为-1的rpn, 只计算这部分的损失, 这部分不是前景就是背景
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), (-1, 2))
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), (-1,))
        # 这里修改原实现, 当groud true目标太小, rpn label=1的样本会很少, 这里平衡一下0、1样本
        rpn_fg = tf.where(rpn_label == 1)
        # rpn_nums = tf.shape(rpn_fg)[0]
        rpn_nums = tf.cast(tf.shape(rpn_fg)[0], dtype=tf.float32) * 1.5
        rpn_nums = tf.cast(tf.math.floor(rpn_nums), dtype=tf.int32)
        rpn_fg_label = tf.gather(rpn_label, rpn_fg)
        rpn_bg = tf.random.shuffle(tf.where(rpn_label == 0))[:rpn_nums]
        rpn_bg_label = tf.gather(rpn_label, rpn_bg)
        rpn_idx = tf.concat([rpn_fg, rpn_bg], axis=0)
        rpn_label = tf.concat([rpn_fg_label, rpn_bg_label], axis=0)
        rpn_cls_score = tf.gather(rpn_cls_score, rpn_idx)

        rpn_cross_entropy = 0.
        if tf.shape(rpn_label)[0] > 0:
            # 计算分类损失
            rpn_cross_entropy = tf.reduce_mean(
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=rpn_label, y_pred=rpn_cls_score))

        # RPN, bbox loss
        rpn_bbox_pred = predictions['rpn_bbox_pred']
        rpn_bbox_targets = anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = anchor_targets['rpn_bbox_outside_weights']
        rpn_loss_box = self._smooth_l1_loss(bbox_pred=rpn_bbox_pred,
                                            bbox_targets=rpn_bbox_targets,
                                            bbox_inside_weights=rpn_bbox_inside_weights,
                                            bbox_outside_weights=rpn_bbox_outside_weights,
                                            sigma=self.smooth_l1_rpn_sigma,
                                            dim=[1, 2, 3])

        # 以下class-loss, bbox-loss为第二次预测损失, 即rpn后两层fc的输出, 可以看成RCNN的输出
        # RCNN, class loss
        cls_score = predictions["cls_score"]
        label = tf.reshape(proposal_targets["labels"], [-1, ])
        # 同样这里修改原实现, 也是为了处理目标太小时样本的均衡问题
        fg = tf.where(label != 0)
        nums = tf.cast(tf.shape(fg)[0], dtype=tf.float32) * 0.5
        nums = tf.cast(tf.math.floor(nums), dtype=tf.int32)
        fg_label = tf.gather(label, fg)
        bg = tf.random.shuffle(tf.where(label == 0))[:nums]
        bg_label = tf.gather(label, bg)
        idx = tf.concat([fg, bg], axis=0)
        label = tf.concat([fg_label, bg_label], axis=0)
        cls_score = tf.gather(cls_score, idx)

        cross_entropy = 0.
        if tf.shape(label)[0] > 0:
            cross_entropy = tf.reduce_mean(
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=label, y_pred=cls_score))

        # rcnn, box loss
        bbox_pred = predictions['bbox_pred']
        bbox_targets = proposal_targets['bbox_targets']
        bbox_inside_weights = proposal_targets['bbox_inside_weights']
        bbox_outside_weights = proposal_targets['bbox_outside_weights']
        # RCNN, bbox loss
        loss_box = self._smooth_l1_loss(bbox_pred=bbox_pred,
                                        bbox_targets=bbox_targets,
                                        bbox_inside_weights=bbox_inside_weights,
                                        bbox_outside_weights=bbox_outside_weights,
                                        sigma=self.smooth_l1_rcnn_sigma,
                                        dim=1)

        # 这里调整了box损失权重
        cross_entropy = cross_entropy * 0.1
        rpn_cross_entropy = rpn_cross_entropy * 0.5
        loss_box = loss_box
        rpn_loss_box = rpn_loss_box
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        # loss = cross_entropy  + rpn_cross_entropy

        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box

    def build_graph(self, is_training):
        """faster-rcnn 构图

           -----------------------------------------                   | -> bbox 第二次回归
          |                                         | -> 后池化全连接 -> |
          |                      | -> bbox 回归 -----                   | -> class 第二次分类
        前卷积 -> 预生成anchors -> |
                                 | -> class 分类
        """
        # [batch=1, h, w, c]
        im_inputs = tf.keras.layers.Input(shape=self.im_input_shape, batch_size=self.batch_size)
        # [batch=1, box_nums, 5] => [[xmin, ymin, xmax, ymax, cls], ... ]
        gt_boxes = tf.keras.layers.Input(shape=self.gt_box_input_shape, batch_size=self.batch_size)

        # 更新类属性
        im_info = tf.cast([tf.shape(im_inputs)[1], tf.shape(im_inputs)[2]], dtype=tf.float32, name="im_info_cast")

        # 前卷积
        feature_map = self.base_model.image_to_head(im_inputs)
        # 生成anchors
        feature_map_height = tf.shape(feature_map)[1]
        feature_map_width = tf.shape(feature_map)[2]
        anchors, anchors_length = GenerateAnchors(feat_stride=self.feat_stride,
                                                  anchor_scales=self.anchor_scales,
                                                  anchor_ratios=self.anchor_ratios)(height=feature_map_height,
                                                                                    width=feature_map_width)
        # x = tf.range(tf.shape(feature_map)[1])
        if is_training:
            # rpn, 第一次bbox回归, class分类
            rois, roi_scores, labels, anchor_targets, proposal_targets, predictions = \
                self._region_proposal_network(conv_net=feature_map,
                                              anchors=anchors,
                                              gt_boxes=gt_boxes,
                                              im_info=im_info,
                                              is_training=is_training)
            #
            # # 后卷积池化全连接, 第二次bbox回归, class分类
            pool5 = CropPoolLayer(pool_size_after_rpn=self.pool_size_after_rpn)(
                bottom=feature_map,
                rois=rois,
                im_info=im_info)
            fc7 = self.base_model.head_to_tail(pool5)
            cls_score, cls_prob, bbox_pred = self._region_classification(fc7)
            predictions['cls_score'] = cls_score
            predictions['cls_prob'] = cls_prob
            predictions['bbox_pred'] = bbox_pred

            model = tf.keras.models.Model(inputs=[im_inputs, gt_boxes],
                                          outputs=[anchor_targets, proposal_targets, predictions, cls_prob, bbox_pred])

            return model

        else:
            rois, roi_scores = \
                self._region_proposal_network(conv_net=feature_map,
                                              anchors=anchors,
                                              gt_boxes=gt_boxes,
                                              im_info=im_info,
                                              is_training=is_training)
            # 后卷积池化全连接, 第二次bbox回归, class分类
            pool5 = CropPoolLayer(pool_size_after_rpn=self.pool_size_after_rpn)(
                bottom=feature_map,
                rois=rois,
                im_info=im_info)
            fc7 = self.base_model.head_to_tail(pool5)
            _, cls_prob, bbox_pred = self._region_classification(fc7)
            model = tf.keras.models.Model(inputs=[im_inputs],
                                          outputs=[rois, cls_prob, bbox_pred])
            return model

    def train(self, epochs, data_root_path, log_dir, save_path):

        faster_rcnn_model = self.build_graph(is_training=True)
        faster_rcnn_model.summary()
        optimizer = tf.keras.optimizers.Adam(1e-5)
        # optimizer = tf.keras.optimizers.Nadam(1e-4)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
        # optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.1)
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        # optimizer = tf.keras.optimizers.Adagrad()

        train_data_generator = DataGenerator(voc_data_path=data_root_path,
                                             classes=self.classes,
                                             batch_size=1,
                                             feat_stride=self.feat_stride[0],
                                             train_fg_thresh=self.train_fg_thresh,
                                             train_bg_thresh_hi=self.train_bg_thresh_hi,
                                             train_bg_thresh_lo=0.1)

        # test_data_generator = DataGenerator(voc_data_path=data_root_path,
        #                                     classes=self.classes,
        #                                     batch_size=1,
        #                                     is_training=False,
        #                                     feat_stride=self.feat_stride[0],
        #                                     train_fg_thresh=self.train_fg_thresh,
        #                                     train_bg_thresh_hi=self.train_bg_thresh_hi,
        #                                     train_bg_thresh_lo=0.)
        summary_writer = tf.summary.create_file_writer(log_dir)

        for epoch in range(epochs):
            for batch in range(train_data_generator.total_batch_size):
                print("epcho: {} batch: {}".format(epoch, batch))
                train_imgs, train_gt_boxes = train_data_generator.next_batch()
                # anchor_targets, proposal_targets, predictions, cls_prob, bbox_pred = \
                #     faster_rcnn_model([train_imgs, train_gt_boxes])

                with tf.GradientTape() as tape:
                    anchor_targets, proposal_targets, predictions, cls_prob, bbox_pred = \
                        faster_rcnn_model([train_imgs, train_gt_boxes], training=True)

                    loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box = \
                        self.compute_losses(anchor_targets, proposal_targets, predictions)
                    if loss > 0 and cross_entropy > 0 and rpn_cross_entropy > 0:

                        # 梯度更新
                        grad = tape.gradient(loss, faster_rcnn_model.trainable_variables)
                        optimizer.apply_gradients(zip(grad, faster_rcnn_model.trainable_variables))

                        # tensorboard loss日志
                        with summary_writer.as_default():
                            tf.summary.scalar('loss/loss', loss,
                                              step=epoch * train_data_generator.total_batch_size + batch)
                            tf.summary.scalar('loss/cross_entropy', cross_entropy,
                                              step=epoch * train_data_generator.total_batch_size + batch)
                            tf.summary.scalar('loss/loss_box', loss_box,
                                              step=epoch * train_data_generator.total_batch_size + batch)
                            tf.summary.scalar('loss/rpn_cross_entropy', rpn_cross_entropy,
                                              step=epoch * train_data_generator.total_batch_size + batch)
                            tf.summary.scalar('loss/rpn_loss_box', rpn_loss_box,
                                              step=epoch * train_data_generator.total_batch_size + batch)

                            # tensorboard image效果
                            if batch % 1 == 0:
                                rois = predictions['rois']
                                cls_prob = np.array(cls_prob)
                                bbox_pred = np.array(bbox_pred)
                                bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
                                bbox_pred = bbox_pred * \
                                            np.array([self.train_bbox_normalize_stds * len(self.classes)], dtype=np.float32) + \
                                            np.array([self.train_bbox_normalize_means * len(self.classes)], dtype=np.float32)
                                pred_boxes = multi_bbox_transform_inv(rois[:, 1:5], bbox_pred)
                                img_shape = np.shape(train_imgs)
                                pred_boxes = clip_boxes(pred_boxes, [img_shape[1], img_shape[2]])

                                im = train_imgs[0] + self.pixel_mean
                                im_pred = im.copy()
                                im_gt = im.copy()
                                for j in range(1, len(self.classes)):
                                    # 非极大抑制
                                    inds = np.where(cls_prob[:, j] > 0.5)[0]
                                    # print("class {} > 0.5 nums: {}".format(self.classes[j], len(inds)))
                                    cls_scores = cls_prob[inds, j]
                                    cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
                                    box_and_score = np.concatenate([cls_boxes, np.expand_dims(cls_scores,1)], axis=1)
                                    keep = box_nms(box_and_score, 0.3)
                                    cls_dets = cls_scores[keep]
                                    box_dets = cls_boxes[keep, :]
                                    for k in range(len(box_dets)):
                                        im_pred = draw_bounding_box(im=im_pred,
                                                                    cls=self.classes[j],
                                                                    scores=cls_dets[k],
                                                                    x_min=box_dets[k][0],
                                                                    y_min=box_dets[k][1],
                                                                    x_max=box_dets[k][2],
                                                                    y_max=box_dets[k][3])

                                if np.sum(np.where(cls_prob[:, 1:] > 0.5)) > 0:
                                    for j in range(np.shape(train_gt_boxes)[1]):
                                        im_gt = draw_bounding_box(im=im_gt,
                                                                  cls=self.classes[int(train_gt_boxes[0][j][4])],
                                                                  scores=1.0,
                                                                  x_min=train_gt_boxes[0][j][0],
                                                                  y_min=train_gt_boxes[0][j][1],
                                                                  x_max=train_gt_boxes[0][j][2],
                                                                  y_max=train_gt_boxes[0][j][3])

                                    concat_imgs = tf.concat([im_gt[:, :, ::-1], im_pred[:, :, ::-1]],
                                                            axis=1)
                                    summ_imgs = tf.expand_dims(concat_imgs, 0)
                                    summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                                    tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs,
                                                     step=batch)

            if epoch % 50 == 0:
                faster_rcnn_model.save_weights("./frcnn-epoch-{}-loss-{}.h5".format(epoch, np.round(loss, 2)))

        faster_rcnn_model.save_weights("./frcnn.h5")

    def predict(self, im, model_path, prod_threshold=0.85, nms_iou_threshold=0.3, nms_max_output_size=200):
        """
        :param im: BGR格式
        :param prod_threshold:
        :param nms_iou_threshold:
        :param nms_max_output_size:
        :return: cls:[...], box:[[x_min,y_min,x_max_y_max]]
        """

        frcnn = self.build_graph(is_training=False)
        frcnn.load_weights(model_path)
        if im is not None:
            h, w, _ = np.shape(im)
            inputs_imgs = np.array([im], dtype=np.float32)
            gt_boxes = np.array([[[0., 0., 0., 0.]]])
            rois, cls_prob, bbox_pred = frcnn.predict([inputs_imgs, gt_boxes])

            # [dx,dy,dw,dh] 转 [x_min,y_min,x_max,y_max]
            cls_prob = np.array(cls_prob)
            bbox_pred = np.array(bbox_pred)
            bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
            bbox_pred = bbox_pred * \
                        np.array([self.train_bbox_normalize_stds * len(self.classes)], dtype=np.float32) + \
                        np.array([self.train_bbox_normalize_means * len(self.classes)], dtype=np.float32)
            pred_boxes = multi_bbox_transform_inv(rois[:, 1:5], bbox_pred)
            pred_boxes = clip_boxes(pred_boxes, [h, w])

            cls_dets = []
            boxes_dets = []
            for j in range(1, len(self.classes)):
                # 非极大抑制
                inds = np.where(cls_prob[:, j] > prod_threshold)[0]
                cls_scores = cls_prob[inds, j]
                cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
                keep = tf.image.non_max_suppression(boxes=cls_boxes,
                                                    scores=cls_scores,
                                                    max_output_size=nms_max_output_size,
                                                    iou_threshold=nms_iou_threshold).numpy()
                cls_dets = cls_scores[keep]
                boxes_dets = cls_boxes[keep, :]

            return cls_dets, boxes_dets

    def test(self):
        frcnn = self.build_graph(is_training=False)
        frcnn.load_weights("./frcnn.h5")
        data_generator = DataGenerator(voc_data_path='../../data/detect_data/',
                                       classes=['__background__', "cat", "dog"],
                                       batch_size=1)
        train_imgs, train_gt_boxes = data_generator.next_batch()
        img_shape = tf.shape(train_imgs)
        rois, cls_prob, bbox_pred = frcnn.predict([train_imgs, train_gt_boxes])

        cls_prob = np.array(cls_prob)
        bbox_pred = np.array(bbox_pred)

        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        bbox_pred[:, 8:] = bbox_pred[:, 8:] * np.array([[0.1, 0.1, 0.2, 0.2]])
        pred_boxes = multi_bbox_transform_inv(rois[:, 1:5], bbox_pred)
        # print(pred_boxes)
        pred_boxes = clip_boxes(pred_boxes, [img_shape[1], img_shape[2]])
        # print(pred_boxes)
        # all_boxes = np.array([[[] for _ in range(1)]
        #                       for _ in range(self.num_classes)])
        max_per_image = 10
        import cv2
        from data.visual_ops import draw_bounding_box
        im = train_imgs[0] * 255.

        for j in range(2, self.num_classes):
            inds = np.where(cls_prob[:, j] > 0.85)[0]
            cls_scores = cls_prob[inds, j]
            cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
            for i in cls_boxes:
                print(i)

            keep = tf.image.non_max_suppression(boxes=cls_boxes,
                                                scores=cls_scores,
                                                max_output_size=200,
                                                iou_threshold=0.3)
            cls_dets = np.array(cls_boxes[keep, :], dtype=np.float32)
            for i in cls_dets:
                im = draw_bounding_box(im, "", "", i[0], i[1], i[2], i[3])
        cv2.imwrite("test.jpg", im)


if __name__ == "__main__":
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    #
    # np.set_printoptions(suppress=True)
    # frcnn = FasterRCNN(rpn_positive_overlap=0.7, classes=['__background__', 'cat', 'dog'])
    # frcnn.train(epochs=100, data_root_path='../../data/detect_data', log_dir='./logs', save_path='./')
    # frcnn = FasterRCNN(rpn_positive_overlap=0.7,
    #                    classes=['__background__','bird', 'cat', 'cow', 'dog', 'horse', 'sheep','aeroplane',
    #                             'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
    #                             'diningtable', 'pottedplant', 'sofa', 'tvmonitor','person'])
    # frcnn.train(epochs=100, data_root_path='../../data/voc_data', log_dir='./logs', save_path='./')
    # frcnn = FasterRCNN(rpn_positive_overlap=0.7,
    #                        classes=['__background__','car'])
    # frcnn.train(epochs=100, data_root_path='../../data/bd100k', log_dir='./logs', save_path='./')
    frcnn = FasterRCNN(rpn_positive_overlap=0.5,
                           classes=['__background__','Y'])
    frcnn.train(epochs=100, data_root_path='../../data/car_logo_data', log_dir='./logs', save_path='./')