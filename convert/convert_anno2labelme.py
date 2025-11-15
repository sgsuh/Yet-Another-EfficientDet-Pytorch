"""
Create: 2021.12.08
Author: SG.SUH
Python: 3.7
"""

import os
import glob
import json
import base64
import argparse

from shutil import copyfile

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    parser.add_argument("--dst_fold",
                        default="")
    parser.add_argument("--sampling",
                        default=30)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    root_path = args.root_path
    image_fold = root_path + '/image'
    annot_path = root_path + '/annots'

    dst_fold = args.dst_fold

    sampling = args.sampling

    if not os.path.isdir(dst_fold):
        os.makedirs(dst_fold)

    with open(annot_path, 'r') as f:
        contents = json.load(f)

    contents.sort(key = lambda x : x['image'])

    image_list = glob.glob(image_fold + '/*.png')
    image_list.sort()

    for i, image_path in enumerate(image_list):
        if i % sampling != 0:
            continue

        file_name = os.path.splitext(os.path.basename(image_path))[0]

        dst_json_path = dst_fold + '/' + file_name + '.json'
        dst_image_path = dst_fold + '/' + file_name + '.png'

        content = contents[i]

        content['version'] = '4.5.7'
        content['flags'] = {}

        for _ in range(len(content['class'])):
            content['shapes'] = [0 for _ in range(len(content['class']))]

        for j in range(len(content['class'])):
            content['shapes'][j] = {'label': {}, 'points': {}, 'group_id': None, 'shape_type': 'rectangle', 'flags': {}}

            if content['class'][j] == 0:
                content['shapes'][j]['label'] = 'TL'
            elif content['class'][j] == 1:
                content['shapes'][j]['label'] = 'Pedestrian'
            elif content['class'][j] == 2:
                content['shapes'][j]['label'] = 'Car'
            elif content['class'][j] == 3:
                content['shapes'][j]['label'] = 'Cyclist'
            else:
                content['shapes'][j]['label'] = 'Dontcare'

            content['shapes'][j]['points'] = [[content['box'][j][0], content['box'][j][1]], [content['box'][j][2], content['box'][j][3]]]

        content['imageData'] = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
        content['imagePath'] = content['image']
        content['imageHeight'] = 1080
        content['imageWidth'] = 1920

        del content['score']
        del content['box']
        del content['class']
        del content['image']

        copyfile(image_path, dst_image_path)

        with open(dst_json_path, 'w') as f:
            json.dump(content, f, indent = 2)