"""
Create: 2021.07.16
Author: SG.SUH
Python: 3.7
"""

import glob
import os
import shutil
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_root",
                        default="")
    parser.add_argument("--img_root",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    src_root = args.src_root
    img_root = args.img_root

    fold_list = glob.glob(src_root + '/*')
    fold_list.sort()

    for fold_path in fold_list:
        fold_name = os.path.basename(fold_path)

        img_fold_path = os.path.join(img_root, fold_name + '/image')

        save_fold_path = os.path.join(fold_path, 'raw')

        if not os.path.isdir(save_fold_path):
            os.makedirs(save_fold_path)

        file_list = glob.glob(fold_path + '/*.jpg')
        file_list.sort()

        for file_path in file_list:
            print(file_path)

            file_name = os.path.splitext(os.path.basename(file_path))[0]
            img_file_path = os.path.join(img_fold_path, file_name + '.png')
            json_file_path = os.path.join(img_fold_path, file_name + '.json')

            save_file_path = os.path.join(save_fold_path, os.path.basename(img_file_path))
            save_json_path = os.path.join(save_fold_path, os.path.basename(json_file_path))

            shutil.copyfile(img_file_path, save_file_path)
            shutil.copyfile(json_file_path, save_json_path)
