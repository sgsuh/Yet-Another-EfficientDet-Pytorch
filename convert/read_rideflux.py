"""
Create: 2021.12.07
Author: SG.SUH
Python: 3.7
"""

import glob
import os
import json
import cv2
import argparse 

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_fold",
                        default="rideflux/train")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    root_fold = args.root_fold

    fold_list = glob.glob(root_fold + '/*')
    fold_list.sort()

    for fold_path in fold_list:
        image_fold = fold_path + '/image0'
        json_fold = fold_path + '/json0'

        image_list = glob.glob(image_fold + '/*.jpg')
        image_list.sort()

        for image_path in image_list:
            filename = os.path.splitext(os.path.basename(image_path))[0]

            print(image_path)
            img = cv2.imread(image_path)

            json_path = json_fold + '/' + filename + '.json'

            with open(json_path, 'r') as f:
                annot_info = json.load(f)['annotations']

                for annot in annot_info:
                    box = annot['bbox']
                    label = annot['class']

                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)


            resize = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

            cv2.imshow('rideflux', resize)
            cv2.waitKey(0)