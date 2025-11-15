"""
Create: 2021.07.26
Author: SG.SUH
Python: 3.7
"""

import os
import glob
import xml.etree.ElementTree as ET
import json
import argparse 

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_fold",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    class_ids = {

    }

    root_fold = args.root_fold

    fold_list = glob.glob(root_fold + '/*')
    fold_list.sort()

    for fold_path in fold_list:
        xml_fold_path = fold_path + '/gt'

        xml_file_list = glob.glob(xml_fold_path + '/*.xml')
        xml_file_list.sort()

        annots = []

        for xml_file_path in xml_file_list:
            boxes = []
            classes = []
            scores = []

            print(xml_file_path)

            file_name = xml_file_path.split('/')[-1].split('.xml')[0]
            file_name = file_name + '.jpg'

            xml_tree = ET.parse(xml_file_path)
            xml_root = xml_tree.getroot()

            img_file_path = fold_path + '/image/' + file_name

            if not os.path.isfile(img_file_path):
                raise ValueError('Not File {}'.format(img_file_path))

            for item in xml_root:
                if item.tag == 'object':
                    class_name = item[0].text

                    if class_name in class_ids:
                        class_id = class_ids[class_name]

                        classes.append(class_id)

                        x1 = float(item[1][0].text)
                        y1 = float(item[1][1].text)
                        x2 = float(item[1][2].text)
                        y2 = float(item[1][3].text)

                        boxes.append([x1, y1, x2, y2])

                        scores.append(1.0)

            annots.append({'image': file_name, 'box': boxes, 'class': classes, 'score': scores})

        save_file_path = fold_path + '/annots'

        with open(save_file_path, 'w') as f:
            json.dump(annots, f)