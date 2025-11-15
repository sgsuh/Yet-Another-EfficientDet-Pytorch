"""
Create: 2021.08.19
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import glob
import os
import json
import cv2
import numpy as np
import struct
import argparse 

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_fold",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    image_fold = args.image_fold

    dst_fold = image_fold + '/dst'

    if not os.path.isdir(dst_fold):
        os.makedirs(dst_fold)

    image_list = glob.glob(image_fold + '/*.png')
    image_list.sort()

    for image_path in image_list:
        image_name = image_path.split('.')[0]
        json_path = image_name + '.json'

        print(image_path)
        file_name = image_path.split('/')[-1].split('.')[0]
        dst_path = dst_fold + '/' + file_name + '.png'

        img = cv2.imread(image_path)
        img_height, img_width, img_ch = img.shape

        roi_x = 480
        roi_y = 0
        roi_width = 960
        roi_height = 540
        roi_offset_x = 0
        roi_offset_y = 0

        with open(json_path, 'r') as f:
            data = json.load(f)

            for bbox in data['shapes']:
                class_id = bbox['label']

                is_roi = False

                if class_id == 'ROI':
                    is_roi = True

                    x1 = min(bbox['points'][0][0], bbox['points'][1][0])
                    x2 = max(bbox['points'][0][0], bbox['points'][1][0])
                    y1 = min(bbox['points'][0][1], bbox['points'][1][1])
                    y2 = max(bbox['points'][0][1], bbox['points'][1][1])

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    width = int(x2 - x1)
                    height = int(y2 - y1)

                    if width > img_width // 3 or height > img_height // 3:
                        roi_width = 960
                        roi_height = 540
                    else:
                        roi_width = 640
                        roi_height = 360

                    if cx < img_width // 2:
                        roi_x1 = max(0, cx - roi_width // 2)
                        roi_x2 = roi_x1 + roi_width
                    else:
                        roi_x2 = min(cx + roi_width // 2, img_width - 1)
                        roi_x1 = roi_x2 - roi_width

                    roi_x = int(roi_x1)

                    if cy < img_height // 2:
                        roi_y1 = max(0, cy - roi_height // 2)
                        roi_y2 = roi_y1 + roi_height
                    else:
                        roi_y2 = min(img_height - 1, cy + roi_height // 2)
                        roi_y1 = roi_y2 - roi_height

                    roi_y = int(roi_y1)

                    break
            
            idx = 12
            step = 4

            dst_img = img.copy()
            dst_img = dst_img.flatten()

            byte_data = bytes(struct.pack('f', roi_x))
            val = np.array(struct.unpack('BBBB', byte_data), dtype = np.uint8)

            dst_img[idx:idx + step] = val
            idx += step

            byte_data = bytes(struct.pack('f', roi_y))
            val = np.array(struct.unpack('BBBB', byte_data), dtype = np.uint8)

            dst_img[idx:idx + step] = val
            idx += step

            byte_data = bytes(struct.pack('f', roi_width))
            val = np.array(struct.unpack('BBBB', byte_data), dtype = np.uint8)

            dst_img[idx:idx + step] = val
            idx += step

            byte_data = bytes(struct.pack('f', roi_height))
            val = np.array(struct.unpack('BBBB', byte_data), dtype = np.uint8)

            dst_img[idx:idx + step] = val
            idx += step

            byte_data = bytes(struct.pack('f', roi_offset_x))
            val = np.array(struct.unpack('BBBB', byte_data), dtype = np.uint8)

            dst_img[idx:idx + step] = val
            idx += step

            byte_data = bytes(struct.pack('f', roi_offset_y))
            val = np.array(struct.unpack('BBBB', byte_data), dtype = np.uint8)

            dst_img[idx:idx + step] = val

            dst_img = dst_img.reshape(img_height, img_width, img_ch)

            cv2.imwrite(dst_path, dst_img)
            


