"""
Create: 2025.11.15
Author: SG.SUH
Python: 3.8.5
"""

import json
import os
import glob
import numpy as np
import cv2
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_fold",
                        default="")
    parser.add_argument("--save_fold",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    root_fold = args.root_fold
    save_fold = args.save_fold

    if not os.path.isdir(save_fold):
        os.makedirs(save_fold)

    fold_list = glob.glob(root_fold + '/*')
    fold_list.sort()

    for path_dir in fold_list:
        
        if not os.path.isdir(path_dir):
            continue

        file_list = os.listdir(path_dir)

        dir = save_fold + '/crop_box' 

        if not os.path.isdir(dir):
            os.makedirs(dir)

        for i in file_list:
            name, ext = os.path.splitext(i)
            
            if(ext == '.png'):
                j = open(path_dir+"/"+name+".json", encoding="UTF-8")
                json_data = json.loads(j.read())
                img = cv2.imread(path_dir + '/' + i)
                count = 0

                for shape in range(0,len(json_data["shapes"])):
                    if(str(json_data["shapes"][shape]["label"]).startswith("TL") or str(json_data["shapes"][shape]["label"]).startswith("PL") or str(json_data["shapes"][shape]["label"]).startswith("VTL")):
                        box = json_data["shapes"][shape]["points"]
                        
                        box[0][0] = min(max(box[0][0], 0), 1919)
                        box[0][1] = min(max(box[0][1], 0), 1079)
                        box[1][0] = min(max(box[1][0], 0), 1919)
                        box[1][1] = min(max(box[1][1], 0), 1079)

                        left = int(min(box[0][0], box[1][0]))
                        right = int(max(box[0][0], box[1][0]))
                        top = int(min(box[0][1], box[1][1]))
                        bottom = int(max(box[0][1], box[1][1]))

                        coord_left = left - left
                        coord_right = right - left
                        coord_top = top - top
                        coord_bottom = bottom - top

                        width = right - left
                        height = bottom - top

                        margin = 0.25

                        padding = height - width if height > width else width - height
                        patch_size = height if height > width else width

                        p_left = int(left - max(0, height - width) // 2 - patch_size * margin)
                        p_right = int(right + max(0, height - width) // 2 + patch_size * margin)
                        p_top = int(top - max(0, width - height) // 2 - patch_size * margin)
                        p_bottom = int(bottom + max(0, width - height) // 2 + patch_size * margin)

                        coord_left = coord_left + int(max(0, height - width) // 2 + patch_size * margin)
                        coord_right = coord_right + int(max(0, height - width) // 2 + patch_size * margin)
                        coord_top = coord_top + int(max(0, width - height) // 2 + patch_size * margin)
                        coord_bottom = coord_bottom + int(max(0, width - height) // 2 + patch_size * margin)

                        p_width = p_right - p_left
                        p_height = p_bottom - p_top

                        total_size = p_width if p_width > p_height else p_height

                        sample = np.zeros((total_size, total_size, 3), dtype = np.uint8)

                        dst_l = 0 if p_left >= 0 else -p_left
                        dst_r = p_right - p_left if p_right < 1920 else (p_right - p_left - (p_right - 1919))
                        dst_t = 0 if p_top >= 0 else -p_top
                        dst_b = p_bottom - p_top if p_bottom < 1080 else (p_bottom - p_top - (p_bottom - 1079))

                        src_l = max(0, p_left)
                        src_r = min(1919, p_right)
                        src_t = max(0, p_top)
                        src_b = min(1079, p_bottom)

                        sample[dst_t:dst_b, dst_l:dst_r, :] = img[src_t:src_b, src_l:src_r, :]

                        img_fold = dir + '/' + json_data["shapes"][shape]["label"]

                        if not os.path.isdir(img_fold):
                            os.makedirs(img_fold)

                        save_path = img_fold + '/' + name + '_{:04d}'.format(count) + '_(' + str(coord_left) + ',' + str(coord_top) + ',' + str(coord_right) + ',' + str(coord_bottom) + ')' + ext

                        print(save_path)

                        count += 1

                        cv2.imwrite(save_path, sample)