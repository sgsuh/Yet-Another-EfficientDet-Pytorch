"""
Create: 2021.09.23
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import argparse
from efficientdet.dataset import parse_json
import struct
import numpy as np
import glob
import cv2
import os
import torch

from efficientdet.infer_engine import InferEngine

obj_list = ['TRAFFIC_LIGHT', 'PEDESTRIAN', 'CAR', 'CYCLIST']
obj_color = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255))

def get_args():
    parser = argparse.ArgumentParser('EfficientDet PyTorch')

    parser.add_argument('--data_fold', type = str, default = 'data')
    parser.add_argument('--model_path', type = str, default = 'weight/effdet_d1.pth')
    parser.add_argument('--valid_crop', action = 'store_true', default = True)
    parser.add_argument('--resize', type = list, default = [455, 256])
    parser.add_argument('--threshold', type = float, default = 0.4)
    parser.add_argument('--iou_threshold', type = float, default = 0.3)
    parser.add_argument('--tl_threshold', type = float, default = 0.3)

    args = parser.parse_args()

    return args

def crop(img, valid_crop):
    idx = 12
    step = 4

    if valid_crop:
        img_x = []
        img_y = []
        byte_data = img.tobytes()

        for _ in range(3):
            img_x.append(float(struct.unpack('f', byte_data[idx:idx + step])[0]))
            
            idx += step

            img_y.append(float(struct.unpack('f', byte_data[idx:idx + step])[0]))

            idx += step

        roi_x = int(img_x[0]) if not np.isnan(img_x[0]) else 0
        roi_y = int(img_y[0]) if not np.isnan(img_y[0]) else 0
        roi_width = int(img_x[1]) if not np.isnan(img_x[1]) else 0
        roi_height = int(img_y[1]) if not np.isnan(img_y[1]) else 0
        roi_offset_x = int(img_x[2]) if not np.isnan(img_x[2]) else 0
        roi_offset_y = int(img_y[2]) if not np.isnan(img_y[2]) else 0
    else:
        roi_x = 480
        roi_y = 0
        roi_width = 960
        roi_height = 540
        roi_offset_x = 0
        roi_offset_y = 0
    
    crop_x = roi_x + roi_offset_x
    crop_y = roi_y + roi_offset_y
    crop = img[crop_y:crop_y + roi_height, crop_x:crop_x + roi_width, :].copy()

    if crop.shape[0] == 0:
        crop_x = 480
        crop_y = 0
        roi_width = 960
        roi_height = 540
        crop = img[crop_y:crop_y + roi_height, crop_x:crop_x + roi_width, :].copy()

    return crop, crop_x, crop_y, roi_width, roi_height

def test(opt):
    file_list = glob.glob(opt.data_fold + '/*.png')
    
    save_fold = 'tl_bg'

    if not os.path.isdir(save_fold):
        os.makedirs(save_fold)

    file_list.sort()

    infer = InferEngine(opt.model_path, batch_size = 2)

    for file_path in file_list:
        img = cv2.imread(file_path)
        save_img = img.copy()
        file_name = os.path.basename(file_path)
        crop_img, offset_x, offset_y, roi_width, roi_height = crop(img, opt.valid_crop)
        im_size = [img.shape[1], img.shape[0]]
        crop_size = [crop_img.shape[1], crop_img.shape[0]]
        batch = [img, crop_img]
        scale = [[im_size[0] / opt.resize[0], im_size[1] / opt.resize[1]], [crop_size[0] / opt.resize[0], crop_size[1] / opt.resize[1]]]
        offset = [[0, 0], [offset_x, offset_y]]
        input = infer.preprocess(batch, net_resize = True, dyn_std = True)
        input = infer.to_tensor(input)

        with torch.no_grad():
            features, regression, classification, anchors = infer.model(input)

        out = infer.postprocess(input, scale, offset, anchors, regression, classification, opt.threshold, opt.iou_threshold, opt.tl_threshold)

        if len(out['box']) != 0:
            for i in range(len(out['box'])):
                x1 = int(out['box'][i][0])
                y1 = int(out['box'][i][1])
                x2 = int(out['box'][i][2])
                y2 = int(out['box'][i][3])

                cv2.rectangle(img, (x1, y1), (x2, y2), obj_color[out['class'][i]], 2)

                obj_str = obj_list[out['class'][i]]
                score = float(out['score'][i])

                cv2.putText(img, '{:.2f}'.format(score), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, obj_color[out['class'][i]], 1)
                    
        cv2.rectangle(img, (offset_x, offset_y), (offset_x + roi_width, offset_y + roi_height), (255, 255, 255), 2)
        cv2.imshow(file_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = get_args()

    test(opt)
