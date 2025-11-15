"""
Create: 2021.11.15
Author: SG.SUH
Python: 3.7
"""

import cv2
import glob
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_fold",
                        type=str,
                        default="export")
    parser.add_argument("--start_idx",
                        type=int,
                        default=1)
    parser.add_argument("--end_idx",
                        type=int,
                        default=10)
    parser.add_argument("--fps",
                        type=int,
                        default=10)
    parser.add_argument("--width",
                        type=int,
                        default=1920)
    parser.add_argument("--height",
                        type=int,
                        default=1080)
    parser.add_argument("--video_path",
                        type=str,
                        default="out.avi")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    video_fold = args.video_fold

    start_idx = args.start_idx
    end_idx = args.end_idx

    fps = args.fps
    width = args.width
    height = args.height
    fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    out = cv2.VideoWriter(args.video_path, fcc, fps, (width, height))

    fold_list = glob.glob(video_fold + '/*')
    fold_list.sort()

    for fold_path in fold_list:
        file_list = glob.glob(fold_path + '/*.png')
        file_list.sort()

        for file_path in file_list:
            print(file_path)

            img = cv2.imread(file_path)

            out.write(img)

    out.release()