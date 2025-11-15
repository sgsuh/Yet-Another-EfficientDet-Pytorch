"""
Modify: 2021.09.15
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import torch
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import json
import pickle
import struct

def parse_json(json_file):
    # Parse Annotation File
    ls = []
    ld = open(json_file, 'r').readlines()

    for l in ld:
        ls.append(json.loads(l.strip()))

    return ls

class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

class ADDataset(Dataset):
    def __init__(self, dataset_path, transform = None, tl_area_th = 40, p_area_th = 40, obj_area_th = 160, valid_crop = False, dump_file = None, resize_w = 640, resize_h = 384):
        self.dataset_path = dataset_path
        self.transform = transform
        self.tl_area_th = tl_area_th
        self.p_area_th = p_area_th
        self.obj_area_th = obj_area_th
        self.valid_crop = valid_crop
        self.data = []

        if dump_file is None or (not os.path.isfile(dump_file)):
            for fold_path in self.dataset_path:
                img_path = os.path.join(fold_path, 'image')
                annot_file = os.path.join(fold_path, 'annots')
                json_infos = parse_json(annot_file)

                if (isinstance(json_infos, list) and len(json_infos) == 1):
                    json_infos = json_infos[0]

                for json_info in json_infos:
                    img_file = os.path.join(img_path, json_info['image'])
                    img = cv2.imread(img_file)

                    print(img_file)

                    if img is None:
                        continue

                    labels = []
                    boxes = json_info['box']
                    classes = json_info['class']
                    scores = json_info['score']
                    obj_num = len(boxes)

                    for idx, box in enumerate(boxes):
                        box = [int(x) for x in box]

                        if scores[idx] > 0.1:
                            box.append(classes[idx])
                        else:
                            box.append(-1)

                        box = np.array(box)
                        
                        labels.append(box)

                    if len(labels) != 0:
                        labels = np.array(labels).reshape((obj_num, 5))

                        self.data.append([img_file, labels])

            if dump_file is not None:
                with open(dump_file, 'wb') as f:
                    pickle.dump(self.data, f)

                print('Save Dump Data {}'.format(dump_file))
        else:
            with open(dump_file, 'rb') as f:
                self.data = pickle.load(f) 

            print('Load Dump Data {}'.format(dump_file))

        self.resize_wh = (resize_w, resize_h)
        self.resize_hw = (resize_h, resize_w)

        print('Data Loader {} Images Files Found.'.format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_file = item[0]
        annots = item[1]
        img = cv2.imread(img_file, 1)
        
        h, w, c = img.shape
        
        img, annots = self.resize_to_net_input(img, annots)

        for i in range(annots.shape[0]):
            area = (annots[i, 2] - annots[i, 0]) * (annots[i, 3] - annots[i, 1])

            if annots[i, 4] == 0:                               # TL
                if area < self.tl_area_th:
                    annots[i, 4] = -1
            elif annots[i, 4] == 1 or annots[i, 4] == 3:        # Pedestrian, Cyclist
                if area < self.p_area_th:
                    annots[i, 4] = -1
            elif annots[i, 4] != -1:                            # Others
                if area < self.obj_area_th:
                    annots[i, 4] = -1
        
        sample = {'img': img, 'annot': annots}

        if self.transform is not None:
            sample = self.transform(sample)

        if isinstance(sample['img'], np.ndarray):
            sample = {'img': torch.from_numpy(sample['img']).float(), 'annot': torch.from_numpy(sample['annot']).float()}

        sample['scale'] = 1.0

        output = {'img': sample['img'], 'annot': sample['annot'], 'scale': sample['scale']}

        return output

    def crop_img(self, img, annots):
        idx = 12
        step = 4
        img_x = []
        img_y = []

        if self.valid_crop:
            byte_data = img.tobytes()

            for _ in range(3):
                img_x.append(float(struct.unpack('f', byte_data[idx:idx + step])[0]))
                idx += step
                img_y.append(float(struct.unpack('f', byte_data[idx:idx + step])[0]))
                idx += step

            roi_x = int(img_x[0])
            roi_y = int(img_y[0])
            roi_width = int(img_x[1])
            roi_height = int(img_y[1])
            roi_offset_x = int(img_x[2])
            roi_offset_y = int(img_y[2])
        else:
            roi_x = 480
            roi_y = 0
            roi_width = 960
            roi_height = 540
            roi_offset_x = 0
            roi_offset_y = 0

        crop_x = roi_x + roi_offset_x
        crop_y = roi_y + roi_offset_y

        if roi_width < self.resize_wh[0]:
            gap_w = self.resize_wh[0] - roi_width

            if crop_x + roi_width // 2 < img.shape[1] // 2:
                crop_x = max(0, crop_x - gap_w // 2)
            else:
                crop_x = min(img.shape[1] - 1, crop_x + roi_width + gap_w // 2) - roi_width - gap_w
            
            roi_width = self.resize_wh[0]

        if roi_height < self.resize_wh[1]:
            gap_h = self.resize_wh[1] - roi_height

            if crop_y + roi_height // 2 < img.shape[0] // 2:
                crop_y = max(0, crop_y - gap_h // 2)
            else:
                crop_y = min(img.shape[0] - 1, crop_y + roi_height + gap_h // 2) - roi_height - gap_h
            
            roi_height = self.resize_wh[1]

        crop = img[crop_y:crop_y + roi_height, crop_x:crop_x + roi_width, :].copy()

        if crop.shape[0] == 0:
            crop_x = 480
            crop_y = 0
            roi_width = 960
            roi_height = 540
            crop = img[crop_y:crop_y + roi_height, crop_x:crop_x + roi_width, :].copy()

        crop_annots = []

        for i in range(annots.shape[0]):
            x1 = annots[i][0]
            y1 = annots[i][1]
            x2 = annots[i][2]
            y2 = annots[i][3]
            
            if x1 >= crop_x + roi_width or x2 <= crop_x or y1 >= crop_y + roi_height or y2 <= crop_y:
                continue

            crop_x1 = max(0, x1 - crop_x)
            crop_y1 = max(0, y1 - crop_y)
            crop_x2 = min(roi_width - 1, x2 - crop_x)
            crop_y2 = min(roi_height - 1, y2 - crop_y)
            area = (x2 - x1) * (y2 - y1)
            crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)


            if area == crop_area:
                crop_annots.append([crop_x1, crop_y1, crop_x2, crop_y2, annots[i][4]])
            else:
                crop_annots.append([crop_x1, crop_y1, crop_x2, crop_y2, -1])

        crop_annots = np.array(crop_annots).reshape((len(crop_annots), 5))

        return crop, crop_annots

    def resize_to_net_input(self, img, annots):
        img_size = np.array(img.shape[:-1])
        net_size = np.array(self.resize_hw)
        scale = np.amin(net_size / img_size)

        if scale > 1.0:
            pad_size = net_size - img_size
            img = cv2.copyMakeBorder(img, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT)
        else:
            img = cv2.resize(img, (0, 0), fx = scale, fy = scale)
            pad_size = net_size - np.array(img.shape[:-1])
            img = cv2.copyMakeBorder(img, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT)
            annots = annots.astype(float)
            annots[:, :-1] *= scale

        return img, annots
    
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
    
class NormalizerDyn(object):
    def __init__(self, eps = 1e-5):
        self.eps = eps

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = image.astype(np.float32) / 255.
        mean = image.mean(axis = (0, 1))
        std = image.std(axis = (0, 1))

        return {'img': ((image.astype(np.float32) - mean) / (std + self.eps)), 'annot': annots}

class ToTensor(object):
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': torch.from_numpy(image).float(), 'annot': torch.from_numpy(annots).float()}