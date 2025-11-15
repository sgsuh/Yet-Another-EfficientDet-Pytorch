"""
Create: 2025.11.15
Author: SG.SUH
Python: 3.8.5
"""

import os
import json

import torch.utils.data as data
import cv2
import numpy as np
import argparse 

bdd100k_ids = {
    'traffic light':0,
    'person':1,
    'car':2,
    'bus':2,
    'truck':2,
    'rider':3,
    'bike':3,
    'motor':3
}

bdd100k_id_text = (
    'traffic light',
    'pedestrian',
    'car',
    'cyclist'
)

bdd100k_id_colors = (
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (0,128,128)
)

class BDD100kDataset(data.Dataset):
    def __init__(self, image_path, label_file, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.label_file = label_file

        self.image = []
        self.label = []


        with open(self.label_file) as fd:
            json_data = json.load(fd)

        for json_item in json_data:
            json_labels = json_item['labels']

            bboxes = []

            for json_label in json_labels:
                category = json_label['category']
                bbox = json_label.get('box2d')
                if category in bdd100k_ids  and bbox is not None:

                    bboxes.append(
                        np.array([
                            int(bbox['x1']),
                            int(bbox['y1']),
                            int(bbox['x2']),
                            int(bbox['y2']),
                            bdd100k_ids[category]
                        ])
                    )

            bbox_len = len(bboxes)

            if bbox_len != 0:
                self.image.append(os.path.join(image_path, json_item['name']))
                self.label.append(np.array(bboxes).reshape((bbox_len, 5)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_file = self.image[idx]
        bboxes = self.label[idx]

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        if self.transform is not None:
            image, bboxes = self.transform(image, bboxes)

        return image, bboxes


def draw_bboxes(image, bboxes):
    box_len = bboxes.shape[0]

    for i in range(box_len):
        box = bboxes[i,:]
        class_id = box[4]
        box_color = bdd100k_id_colors[class_id] if class_id >= 0 else (50,50,50)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]) , box_color, 2)

    return image

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path",
                        default="bdd100k/images/100k/train")
    parser.add_argument("--label_file",
                        default="bdd100k/labels/bdd100k_labels_images_train.json")
    
    return parser.parse_args()

def main():
    args = make_parser()
    image_path = args.image_path
    label_file = args.label_file

    loader = BDD100kDataset(image_path,
                            label_file)

    for image, bboxes in loader:
        image = draw_bboxes(image, bboxes)

        cv2.imshow('', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()