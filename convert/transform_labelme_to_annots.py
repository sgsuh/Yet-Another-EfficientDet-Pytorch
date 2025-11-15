"""
Create: 2021.08.05
Author: SG.SUH
Python: 3.7
"""

import glob
import os
import json
import argparse 

from shutil import copyfile

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_fold",
                        default="label")
    parser.add_argument("--dst_root_fold",
                        default="od")
    
    return parser.parse_args()

if __name__ == "__main__":    
    args = make_parser()

    root_fold = args.root_fold
    dst_root_fold = args.dst_root_fold

    fold_list = glob.glob(root_fold + '/*')
    fold_list.sort()

    class_ids_v1 = {
    
    }

    class_ids_v2 = {
        
    }

    class_ids = class_ids_v1

    for fold_path in fold_list:
        if not os.path.isdir(fold_path):
            continue

        parse = fold_path.split('/')[-1].split('_')
        car_num = parse[0]
        day = parse[1]

        car_fold_path = dst_root_fold + '/{}'.format(car_num)

        if not os.path.isdir(car_fold_path):
            os.makedirs(car_fold_path)

        dst_fold_path = car_fold_path + '/' + day

        if not os.path.isdir(dst_fold_path):
            os.makedirs(dst_fold_path)

        file_list = glob.glob(fold_path + '/*.png')
        file_list.sort()

        dst_img_fold = dst_fold_path + '/image'

        if not os.path.isdir(dst_img_fold):
            os.makedirs(dst_img_fold)

        annots = []

        for img_file_path in file_list:
            print(img_file_path)

            json_file_path = img_file_path.split('.')[0] + '.json'
            img_file_name = img_file_path.split('/')[-1].split('.')[0] + '.png'

            dst_img_file_path = dst_img_fold + '/' + img_file_name

            copyfile(img_file_path, dst_img_file_path)

            boxes = []
            classes = []
            scores = []

            with open(json_file_path, 'r') as f:
                data = json.load(f)

                for bbox in data['shapes']:
                    class_id = class_ids[bbox['label']]

                    if class_id == -2:
                        continue

                    x1 = min(bbox['points'][0][0], bbox['points'][1][0])
                    x2 = max(bbox['points'][0][0], bbox['points'][1][0])
                    y1 = min(bbox['points'][0][1], bbox['points'][1][1])
                    y2 = max(bbox['points'][0][1], bbox['points'][1][1])

                    classes.append(class_id)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(1.0)

            annots.append({'image': img_file_name, 'box': boxes, 'class': classes, 'score': scores})

        save_label_path = dst_fold_path + '/annots'

        with open(save_label_path, 'w') as f:
            json.dump(annots, f)