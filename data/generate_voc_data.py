
import os
import math
import numpy as np
import cv2
import tensorflow as tf
from data.xml_ops import xml2dict
from anchors_ops import generate_anchors_pre_tf
from bbox_ops import bbox_overlaps_tf, clip_boxes_tf


class DataGenerator:
    def __init__(self,
                 voc_data_path,
                 classes,
                 im_size=600,
                 data_max_size=15000,
                 data_max_size_per_class=350,
                 max_box_fraction=0.5,
                 batch_size=1,
                 is_training=True,
                 feat_stride=16,
                 train_fg_thresh=0.5,
                 train_bg_thresh_hi=0.5,
                 train_bg_thresh_lo=0.1,
                 pixel_mean=np.array([[[102.9801, 115.9465, 122.7717]]]),
                 is_voc_2012=False,
                 ):
        self.annotation_files_root_path = os.path.join(voc_data_path, "Annotations")
        self.img_files_root_path = os.path.join(voc_data_path, "JPEGImages")
        self.imgset_root_path = os.path.join(voc_data_path, "ImageSets", "Main")
        self.max_box_fraction = max_box_fraction
        self.batch_size = batch_size
        self.is_training = is_training
        self.feat_stride = feat_stride
        self.train_fg_thresh = train_fg_thresh
        self.train_bg_thresh_hi = train_bg_thresh_hi
        self.train_bg_thresh_lo = train_bg_thresh_lo
        self.pixel_mean = pixel_mean
        self.data_max_size = data_max_size
        self.data_max_size_per_class = data_max_size_per_class
        self.im_size = im_size
        # classes = ["__background__", ...]
        self.classes = classes
        assert self.classes[0] == "__background__", "classes index 0 need to be __background__"
        self.cls_to_inds = dict(list(zip(self.classes, list(range(len(self.classes))))))
        # 初始化
        self.total_batch_size = 0
        self.img_files = []
        self.annotation_files = []
        self.current_batch_index = 0
        self.file_indices = []
        # color map

        # 重新赋值
        self.__load_files()
        # self._on_epoch_end()
        # 过滤只包含小目标的样本数据
        self.__filter_small_objs()
        # 过滤那些太大目标的样本数据
        self.__filter_big_objs()
        # 平衡每个类别样本数
        self.__balance_class_data()

    def __load_files(self):
        if self.is_training:
            file = os.path.join(self.imgset_root_path, "trainval.txt")
        else:
            file = os.path.join(self.imgset_root_path, 'test.txt')

        img_files = []
        annotation_files = []
        with open(file, encoding='utf-8', mode='r') as f:
            data = f.readlines()
            for file_name in data:
                file_name = file_name.strip()
                img_file_jpeg = os.path.join(self.img_files_root_path, "{}.jpeg".format(file_name))
                img_file_jpg = os.path.join(self.img_files_root_path, "{}.jpg".format(file_name))
                annotation_file = os.path.join(self.annotation_files_root_path, "{}.xml".format(file_name))
                if os.path.isfile(annotation_file):
                    if os.path.isfile(img_file_jpeg):
                        img_files.append(img_file_jpeg)
                        annotation_files.append(annotation_file)
                    elif os.path.isfile(img_file_jpg):
                        img_files.append(img_file_jpg)
                        annotation_files.append(annotation_file)

            self.img_files = img_files[:self.data_max_size]
            self.annotation_files = annotation_files[:self.data_max_size]

        self.total_batch_size = int(math.floor(len(self.annotation_files) / self.batch_size))
        self.file_indices = np.arange(len(self.annotation_files))
        # np.random.shuffle(self.file_indices)

    def __filter_big_objs(self):
        """ 过滤目标太大的样本数据 """
        filter_annotation_files = []
        filter_img_files = []

        for i in range(len(self.annotation_files)):
            # for i in self.file_indices:
            annotation = xml2dict(self.annotation_files[i])
            img_width = int(annotation['annotation']['size']['width'])
            img_height = int(annotation['annotation']['size']['height'])
            objs = annotation['annotation']['object']

            area = img_height * img_width
            keep = True
            if type(objs) == list:
                for box in objs:
                    xmin = int(float(box['bndbox']['xmin']))
                    ymin = int(float(box['bndbox']['ymin']))
                    xmax = int(float(box['bndbox']['xmax']))
                    ymax = int(float(box['bndbox']['ymax']))
                    if (ymax - ymin) * (xmax - xmin) / area > self.max_box_fraction:
                        keep = False
            else:
                xmin = int(float(objs['bndbox']['xmin']))
                ymin = int(float(objs['bndbox']['ymin']))
                xmax = int(float(objs['bndbox']['xmax']))
                ymax = int(float(objs['bndbox']['ymax']))
                if (ymax - ymin) * (xmax - xmin) / area > self.max_box_fraction:
                    keep = False

            if keep:
                filter_annotation_files.append(self.annotation_files[i])
                filter_img_files.append(self.img_files[i])
            else:
                print("filter big obj file: {}, {}".format(self.annotation_files[i], self.img_files[i]))

        remove_file_nums = len(self.annotation_files) - len(filter_annotation_files)
        self.annotation_files = filter_annotation_files
        self.img_files = filter_img_files

        self.total_batch_size = int(math.floor(len(self.annotation_files) / self.batch_size))
        # self.total_batch_size = int(math.floor(len(not_filter_file_indices) / self.batch_size))
        self.file_indices = np.arange(len(self.annotation_files))
        # self.file_indices = not_filter_file_indices
        print("after filter big obj, total file nums: {}, remove {} files".format(len(self.annotation_files),
                                                                                  remove_file_nums))

    def __filter_small_objs(self):
        """ 过滤只包含小目标的样本数据 """
        filter_annotation_files = []
        filter_img_files = []

        for i in range(len(self.annotation_files)):
            # for i in self.file_indices:
            annotation = xml2dict(self.annotation_files[i])
            print(self.annotation_files[i])
            print(annotation)

            # 预生成的anchors
            img_width = int(annotation['annotation']['size']['width'])
            img_height = int(annotation['annotation']['size']['height'])
            anchors, _ = generate_anchors_pre_tf(height=int(img_height / self.feat_stride),
                                                 width=int(img_width / self.feat_stride),
                                                 feat_stride=self.feat_stride)
            inds_inside = tf.reshape(tf.where(
                (anchors[:, 0] >= -0) &
                (anchors[:, 1] >= -0) &
                (anchors[:, 2] < (img_width + 0)) &  # width
                (anchors[:, 3] < (img_height + 0))  # height
            ), shape=(-1,))

            clip_anchors = clip_boxes_tf(anchors, [img_height, img_width])

            # gt_boxes
            objs = annotation['annotation']['object']
            boxes = []

            if type(objs) == list:
                for box in objs:
                    # cls_inds = self.cls_to_inds[box['name']]
                    xmin = int(float(box['bndbox']['xmin']))
                    ymin = int(float(box['bndbox']['ymin']))
                    xmax = int(float(box['bndbox']['xmax']))
                    ymax = int(float(box['bndbox']['ymax']))
                    boxes.append([xmin, ymin, xmax, ymax])
            else:
                # cls_inds = self.cls_to_inds[objs['name']]
                xmin = int(float(objs['bndbox']['xmin']))
                ymin = int(float(objs['bndbox']['ymin']))
                xmax = int(float(objs['bndbox']['xmax']))
                ymax = int(float(objs['bndbox']['ymax']))
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = tf.cast(boxes, dtype=tf.float32)
            overlaps = bbox_overlaps_tf(clip_anchors, boxes).numpy()
            max_overlaps = overlaps.max(axis=1)

            fg_inds = np.where(max_overlaps >= self.train_fg_thresh)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((max_overlaps < self.train_bg_thresh_hi) &
                               (max_overlaps >= self.train_bg_thresh_lo))[0]

            # image is only valid if such boxes exist
            if (len(fg_inds) > 0 or len(bg_inds) > 0) and tf.shape(inds_inside)[0] > 0:
                # not_filter_file_indices.append(i)
                filter_annotation_files.append(self.annotation_files[i])
                filter_img_files.append(self.img_files[i])
            else:
                print("filter small obj file: {}, {}".format(self.annotation_files[i], self.img_files[i]))

        remove_file_nums = len(self.annotation_files) - len(filter_annotation_files)
        self.annotation_files = filter_annotation_files
        self.img_files = filter_img_files

        self.total_batch_size = int(math.floor(len(self.annotation_files) / self.batch_size))
        # self.total_batch_size = int(math.floor(len(not_filter_file_indices) / self.batch_size))
        self.file_indices = np.arange(len(self.annotation_files))
        # self.file_indices = not_filter_file_indices
        print("after filter small obj, total file nums: {}, remove {} files".format(len(self.annotation_files),
                                                                                    remove_file_nums))

    def __balance_class_data(self):
        """ 平衡每个类别样本数 """
        # balance_annotation_files = []
        # balance_img_files = []
        balance_file_indices = []
        per_class_nums = dict(zip(self.classes, [0] * len(self.classes)))

        for i in self.file_indices:
            annotation = xml2dict(self.annotation_files[i])
            objs = annotation['annotation']['object']

            all_classes = []
            if type(objs) == list:
                for obj in objs:
                    all_classes.append(obj['name'])
            else:
                all_classes.append(objs['name'])

            keep = False
            # if 'person' not in all_classes:
            for cls in set(all_classes):
                if per_class_nums[cls] <= self.data_max_size_per_class:
                    keep = True
                    per_class_nums[cls] += 1
            if keep:
                balance_file_indices.append(i)
                # balance_annotation_files.append(self.annotation_files[i])
                # balance_img_files.append(self.img_files[i])

        remove_file_nums = len(self.annotation_files) - len(balance_file_indices)
        # remove_file_nums = len(self.file_indices) - len(balance_file_indices)
        # self.annotation_files = balance_annotation_files
        # self.img_files = balance_img_files

        # self.total_batch_size = int(math.floor(len(self.annotation_files) / self.batch_size))
        self.total_batch_size = int(math.floor(len(balance_file_indices) / self.batch_size))
        # self.file_indices = np.arange(len(self.annotation_files))
        self.file_indices = balance_file_indices
        print("after balance total file nums: {}, remove {} files".format(len(balance_file_indices), remove_file_nums))
        print("every class nums: {}".format(per_class_nums))

    def next_batch(self):
        if self.current_batch_index >= self.total_batch_size:
            self.current_batch_index = 0
            self._on_epoch_end()

        indices = self.file_indices[self.current_batch_index * self.batch_size:
                                    (self.current_batch_index + 1) * self.batch_size]
        annotation_file = [self.annotation_files[k] for k in indices]
        print(annotation_file)
        # annotation_file = ["../../data/car_data/Annotations/2011_001100.xml"]
        # annotation_file = ["../../data/voc_data/Annotations/2008_003374.xml"]
        img_file = [self.img_files[k] for k in indices]
        print(img_file)
        # img_file = ["../../data/car_data/JPEGImages/2011_001100.jpg"]
        # img_file = ["../../data/voc_data/JPEGImages/2008_003374.jpg"]
        imgs, gt_boxes = self._data_generation(annotation_files=annotation_file,
                                               img_files=img_file)
        self.current_batch_index += 1
        print(gt_boxes)
        return imgs, gt_boxes

    def _on_epoch_end(self):
        self.file_indices = np.arange(len(self.annotation_files))
        np.random.shuffle(self.file_indices)
        self.__balance_class_data()

    def _resize_im(self, im, box):
        """ 图片统一处理到一样的大小
        :param im:
        :param box:
        :return:
        """
        im_shape = im.shape
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.im_size) / float(im_size_max)
        im_resize = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        im_resize_shape = im_resize.shape
        blob = np.zeros((self.im_size, self.im_size, 3), dtype=np.float32)
        blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize

        box[:, :4] = box[:, :4] * im_scale
        # print(im_shape, im_resize_shape, np.shape(blob))
        return blob, box, im_scale

    def _data_generation(self, annotation_files, img_files):
        """
        :param annotation_files:
        :param img_files:
        :return:
        """
        gt_boxes = []
        imgs = []
        for i in range(len(annotation_files)):
            img = cv2.imread(img_files[i])
            annotation = xml2dict(annotation_files[i])
            objs = annotation['annotation']['object']
            boxes = []
            if type(objs) == list:
                for box in objs:
                    cls_inds = self.cls_to_inds[box['name']]
                    xmin = int(float(box['bndbox']['xmin']))
                    ymin = int(float(box['bndbox']['ymin']))
                    xmax = int(float(box['bndbox']['xmax']))
                    ymax = int(float(box['bndbox']['ymax']))
                    boxes.append([xmin, ymin, xmax, ymax, cls_inds])
            else:
                cls_inds = self.cls_to_inds[objs['name']]
                xmin = int(float(objs['bndbox']['xmin']))
                ymin = int(float(objs['bndbox']['ymin']))
                xmax = int(float(objs['bndbox']['xmax']))
                ymax = int(float(objs['bndbox']['ymax']))
                boxes.append([xmin, ymin, xmax, ymax, cls_inds])

            if img is not None:
                img, boxes, _ = self._resize_im(img, np.array(boxes, dtype=np.float32))
                imgs.append(img)
                gt_boxes.append(boxes)
        return np.array(imgs, dtype=np.float32) - self.pixel_mean, np.array(gt_boxes, dtype=np.float32)


# class VOC2012DataGenerator(DataGenerator):
class VOC2012DataGenerator:
    def __init__(self):
        # super(VOC2012DataGenerator, self).__init__()
        self.color_map = None
        self.colormap2label = []

        # 初始化一些变量
        self.__init_voc_variables()

    def __init_voc_variables(self):
        self.color_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                          [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                          [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        self.classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant',
                        'sheep', 'sofa', 'train', 'tv/monitor']
        self.colormap2label = np.zeros(256 ** 3)
        for i, colormap in enumerate(self.color_map):
            self.colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


    def generate_voc_2012_mask(self, mask_files):
        mask_img = cv2.imread(mask_files)
        idx = ((mask_img[:, :, 0] * 256 + mask_img[:, :, 1]) * 256 + mask_img[:, :, 2])
        return self.colormap2label[idx]


if __name__ == "__main__":
    from data.visual_ops import draw_bounding_box

    # img = cv2.imread("./tmp/test_mask.png")
    # # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # # print(gray)
    # object_filter = np.dstack((1, 1, 1))
    # filtered = np.array(np.multiply(object_filter, img[:, :, :3]), dtype=np.float32)
    # print(filtered)
    # print(np.shape(filtered))
    # gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
    # max_gray = np.max(gray)
    # print(gray)
    # print(max_gray)

    mask_file = "../data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png"
    vdg = VOC2012DataGenerator()
    data = vdg.generate_voc_2012_mask(mask_file)
    for i in range(np.shape(data)[0]):
        print(data[i,:])
    cv2.imshow("m",data)
    cv2.waitKey(0)

    # d = DataGenerator(voc_data_path='./car_logo_data', classes=['__background__', "Y"], is_training=True)
    # d = DataGenerator(voc_data_path='./voc_data/',
    #                   classes=['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
    #                            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    #                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
    #                            'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'],
    #                   is_training=True)
    # imgs, gt_boxes = d.next_batch()

    # print(np.shape(imgs), np.shape(gt_boxes), d.total_batch_size, d.current_batch_index)
    # im = imgs[0, :, :, :]
    #
    # im_resize = cv2.resize(im.copy(), None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # print(im)
    # im_gt = im.copy()
    # for j in range(np.shape(gt_boxes)[1]):
    #     im_gt = draw_bounding_box(im=im_gt,
    #                               cls="",
    #                               scores="",
    #                               x_min=gt_boxes[0][j][0],
    #                               y_min=gt_boxes[0][j][1],
    #                               x_max=gt_boxes[0][j][2],
    #                               y_max=gt_boxes[0][j][3])
    # cv2.imshow("m", np.array(im_gt, dtype=np.uint8))
    # cv2.imshow("m1", np.array(im_resize, dtype=np.uint8))
    #
    # cv2.waitKey(0)
