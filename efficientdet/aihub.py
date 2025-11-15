"""
Create: 2025.11.15
Author: SG.SUH
Python: 3.8.5
"""

import os
import json
import glob
import torch
import torch.utils.data as data
import cv2
import numpy as np

_debug_ = False

class_id_num_map = {
    "Traffic_light":0,
    "Pedestrian":1,
    "Car":2,
    "Cyclist":3
}

class_colors = (
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (0,255,255)
)

def parse_json(json_file):
    '''
    @brief parse annotation file
    '''
    ls = []
    ld = open(json_file, 'r').readlines()
    for l in ld:
        ls.append(json.loads(l.strip()))
    return ls

def trans_annot_format(label_file):
    annots = parse_json(label_file)[0]
    annots = annots['annotations']

    bboxes = []

    for annot in annots:
        class_id = class_id_num_map[annot['name']]

        if annot['ignored']:
            class_id = -1

        bbox = annot['bbox']

        bboxes.append(np.array([bbox[0], bbox[1], bbox[2], bbox[3], class_id]))

    if len(bboxes) != 0:
        obj_len = len(bboxes)
        bboxes = np.array(bboxes).reshape((obj_len, 5))
    else:
        return None

    return bboxes


class aihub_LQ_dataset(data.Dataset):
    def __init__(self, data_path, transform=None, urban_rate=2, tl_area_thr=8, p_area_thr = 12,obj_area_thr=24, is_training=True):
        image_path = os.path.join(data_path, 'images')
        label_path = os.path.join(data_path, 'labels')

        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.urban_rate = urban_rate

        self.tl_area_thr = tl_area_thr
        self.p_area_thr = p_area_thr
        self.obj_area_thr = obj_area_thr

        self.is_training = is_training

        self.data = []

        data_list = glob.glob(os.path.join(label_path, '*'))

        for data_item in data_list:
            data_name = os.path.basename(data_item)
            if '1M' in data_name:
                continue

            if not ('urban' in data_name or 'highway' in data_name):
                continue

            if 'urban' in data_name:
                is_urban = True
            else:
                is_urban = False

            scene_list = glob.glob(os.path.join(data_item, '*'))

            for scene in scene_list:
                seq_list = glob.glob(os.path.join(scene, '*_annotations_v001_1'))
                for seq in seq_list:
                    label_file_list = glob.glob(os.path.join(seq, '*.json'))

                    for label_file in label_file_list:
                        label = parse_json(label_file)[0]

                        image_file = os.path.join(image_path, label['image'])

                        if os.path.exists(image_file):

                            label = trans_annot_format(label_file)
                            if label is None:
                                continue

                            if is_urban:
                                for i in range(urban_rate):
                                    self.data.append(image_file, label)
                            else:
                                self.data.append(image_file, label)
                        else:
                            print('A FILE NOT FOUND:')
                            print(label_file)
                            print(image_file)

        self.image_size_wh = (512, 256)
        self.image_size_hw = (256, 512)

        print('[aihub LQ dataloader]: %d images found' % len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item[0]
        annots = item[1]

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        if self.is_training and np.random.rand() < 0.5:
            image, annots = self.crop(image, annots)

        if self.is_training and np.random.rand() < 0.5:
            image, annots = self.random_scale(image, annots)

        image, annots = self.resize_to_net_input(image, annots)

        obj_len = annots.shape[0]

        for i in range(obj_len):
            annot = annots[i]
            area = (annot[2] - annot[0]) * (annot[3] - annot[1])

            if annot[4] == 0: # tL
                if area < self.tl_area_thr:
                    annots[i,4] = -1
            elif annot[4] == 1 or annot[4] == 3: # pedestrian
                if area < self.p_area_thr:
                    annots[i,4] = -1
            elif annot[4] != -1:
                if area < self.obj_area_thr:
                    annots[i,4] = -1

        sample = {'img': image, 'annot': annots}

        if self.transform is not None:
            sample = self.transform(sample)

        if (not _debug_) and isinstance(sample['img'], np.ndarray):
            sample = {'img': torch.from_numpy(sample['img']).float(),
                      'annot': torch.from_numpy(sample['annot']).float()}

        sample['scale'] = 1.0
        return sample

    def crop(self, image, annots):
        crop_size = np.array(self.image_size_hw)

        xmax = -1
        xmin = image.shape[1] - 1
        ymax = -1
        ymin = image.shape[0] - 1
        box_len = annots.shape[0]

        for i in range(box_len):
            if (annots[i, 4] == 0):
                if annots[i, 0] < xmin:
                    xmin = annots[i, 0]
                if annots[i, 1] < ymin:
                    ymin = annots[i, 1]
                if annots[i, 2] > xmax:
                    xmax = annots[i, 2]
                if annots[i, 3] > ymax:
                    ymax = annots[i, 3]

        crop_size = (crop_size * (np.random.rand() + 1.0)).astype(int)

        def expand_points(pt1, pt2, image_size, expand_size, random=False):
            if random:
                exp_size1 = int(np.random.rand() * expand_size)
                exp_size2 = expand_size - exp_size1
                pt1 -= exp_size1
                pt2 += exp_size2
            else:
                each_size = expand_size // 2
                pt1 -= each_size
                pt2 += expand_size - each_size

            if pt1 < 0:
                pt2 -= pt1
                pt1 = 0

            if pt2 > (image_size - 1):
                pt1 -= pt2 - (image_size - 1)
                if (pt1 < 0):
                    pt1 -= 1
                pt2 = image_size - 1

            return pt1, pt2

        def cut_box(annots, x1, y1, x2, y2):
            annots[0] = max(0, annots[0] - x1)
            annots[1] = max(0, annots[1] - y1)
            annots[2] = min(x2 - x1, annots[2] - x1)
            annots[3] = min(y2 - y1, annots[3] - y1)

            return annots

        def calc_area(annots):
            return (annots[2] - annots[0]) * (annots[3] - annots[1])

        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0

        if xmax < 0:
            return image, annots

        else:
            if (xmax - xmin) < crop_size[1]:
                pad_width = crop_size[1] - (xmax - xmin)
                x1, x2 = expand_points(xmin, xmax, image.shape[1], pad_width,True)

            if (ymin - ymax) < crop_size[0]:
                pad_height = crop_size[0] - (ymax - ymin)
                y1, y2 = expand_points(ymin, ymax, image.shape[0], pad_height,True)

            aspect_ratio = self.image_size_wh[0] / self.image_size_wh[1]

            new_width = x2 - x1
            new_height = y2 - y1

            if (new_width / new_height) < aspect_ratio:
                pad_width = int(aspect_ratio * new_height - new_width)
                x1, x2 = expand_points(x1, x2, image.shape[1], pad_width)
            elif (new_width / new_height) > aspect_ratio:
                pad_height = int(new_width / aspect_ratio - new_height)
                y1, y2 = expand_points(y1, y2, image.shape[0], pad_height)

            if (x1 < 0) or (y1 < 0) or (x2 > image.shape[1]) or (y2 > image.shape[0]):
                return image, annots

        box_len = annots.shape[0]
        new_annots = []
        for i in range(box_len):
            if annots[i, 0] > x2 or annots[i, 1] > y2 or annots[i, 2] < x1 or annots[i, 3] < y1:
                continue
            old_area = calc_area(annots[i])
            cutted = cut_box(annots[i], x1, y1, x2, y2)
            new_area = calc_area(cutted)
            if new_area * 2 < old_area:
                cutted[4] = -1
            new_annots.append(cutted)

        new_box_len = len(new_annots)
        new_annots = np.array(new_annots).reshape((new_box_len, 5))

        cropped = image[y1:y2, x1:x2, :]

        return cropped, new_annots

    def resize_to_net_input(self,image, annots):
        image_size = np.array(image.shape[:-1])
        net_size = np.array(self.image_size_hw)

        scale = np.amin(net_size / image_size)
        if (scale > 1.0):
            pad_size = net_size - image_size

            image = cv2.copyMakeBorder(image, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT)
        else:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            pad_size = net_size - np.array(image.shape[:-1])
            image = cv2.copyMakeBorder(image, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT)

            annots = annots.astype(float)
            annots[:, :-1] *= scale
            annots = annots.astype(int)

        return image, annots

    def random_scale(self,image, annots):

        scale = np.random.rand() * 0.5 + 0.5

        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        annots = annots.astype(float)
        annots[:, :-1] *= scale
        annots = annots.astype(int)

        return image, annots
