"""
Create: 2021.07.26
Author: SG.SUH
Python: 3.7
"""

import json
import os
import argparse 

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_path",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    class_ids = {

    }

    fold_path = args.fold_path
    json_path = fold_path + '/annots.json'

    def parse_json(json_file):
        ls = []
        ld = open(json_file, 'r').readlines()

        for l in ld:
            ls.append(json.loads(l.strip()))
            
        return ls

    label_infos = parse_json(json_path)

    annots = []

    for label_info in label_infos[0]:
        boxes = []
        classes = []
        scores = []

        file_name = label_info['image']

        img_file_path = fold_path + '/image/' + file_name

        print(img_file_path)

        if not os.path.isfile(img_file_path):
            raise ValueError('Not File {}'.format(img_file_path))

        bbox_list = label_info['annotations']

        for bbox in bbox_list:
            class_id = class_ids[bbox['label']]

            cx = float(bbox['coordinates']['x'])
            cy = float(bbox['coordinates']['y'])
            w = float(bbox['coordinates']['width'])
            h = float(bbox['coordinates']['height'])

            x1 = cx - w / 2.
            y1 = cy - h / 2.
            x2 = cx + w / 2.
            y2 = cy + w / 2.

            classes.append(class_id)
            boxes.append([x1, y1, x2, y2])
            scores.append(1.0)

        annots.append({'image': file_name, 'box': boxes, 'class': classes, 'score': scores})

    save_file_path = fold_path + '/annots'

    with open(save_file_path, 'w') as f:
        json.dump(annots, f)