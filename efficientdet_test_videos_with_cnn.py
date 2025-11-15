"""
Create: 2021.09.23
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import argparse
import os
import glob
import torch
import cv2
import numpy as np
import onnxruntime as ort

from efficientdet.infer_engine import InferEngine

from log_loader import LogLoader

from convert.convert_det_to_json import convert_det_cnn_to_json

obj_list = ['TRAFFIC_LIGHT', 'PEDESTRIAN', 'CAR', 'CYCLIST', 'BG','TL_Go','TL_GoLeft','TL_GoWarn','TL_Off','TL_Stop','TL_StopLeft','TL_StopWarn','TL_UN','TL_Warn','PL_Go','PL_Stop','VL_OFF','VL_UN','VTL_Go','VTL_Stop','VTL_Warn']
obj_color = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 0), (125, 0, 0), (0, 125, 0), (0, 0, 125), (125, 125, 0), (125, 0, 125), (0, 125, 125), (125, 125, 125), (255, 255, 255), (75, 0, 0), (0, 75, 0), (0, 0, 75), (75, 75, 0), (75, 0, 75), (75, 75, 75))

def get_args():
    parser = argparse.ArgumentParser('EfficientDet PyTorch')

    parser.add_argument('--fold_name', type = str, default = 'logging')
    parser.add_argument('--model_path', type = str, default = 'weight/effdet_d1.pth')
    parser.add_argument('--HL_model', type = str, default = 'weight/hl.onnx')
    parser.add_argument('--VL_model', type = str, default = 'weight/vl.onnx')
    parser.add_argument('--threshold', type = float, default = 0.4)
    parser.add_argument('--iou_threshold', type = float, default = 0.3)
    parser.add_argument('--tl_threshold', type = float, default = 0.3)
    parser.add_argument('--resize', type = list, default = [455, 256])
    parser.add_argument('--save_root', type = str, default = 'result')

    args = parser.parse_args()

    return args

def test_video(opt):
    model_name = os.path.splitext(os.path.basename(opt.model_path))[0]
    hl_model = ort.InferenceSession(opt.HL_model)
    vl_model = ort.InferenceSession(opt.VL_model)

    infer = InferEngine(opt.model_path, batch_size = 2)
    file_list = glob.glob(opt.fold_name + '/*.dat')

    file_list.sort()

    if not os.path.isdir(opt.save_root):
        os.makedirs(opt.save_root)

    for file_path in file_list:
        dataset = LogLoader(file_path)
        log_name = os.path.splitext(os.path.basename(file_path))[0]
        save_log_fold = os.path.join(opt.save_root, log_name)

        if not os.path.isdir(save_log_fold):
            os.makedirs(save_log_fold)

        save_det_fold = os.path.join(save_log_fold, model_name)

        if not os.path.isdir(save_det_fold):
            os.makedirs(save_det_fold)

        save_img_fold = os.path.join(save_log_fold, 'image')

        if not os.path.isdir(save_img_fold):
            os.makedirs(save_img_fold)

        for im, crop in dataset:
            try:
                im_size = [im.shape[1], im.shape[0]]
                crop_size = [crop.shape[1], crop.shape[0]]

                if crop.shape[0] == 0:
                    continue
            except:
                print('End of Decoding')

                break

            result = im.copy()
            batch = [im, crop]
            scale = [[im_size[0] / opt.resize[0], im_size[1] / opt.resize[1]], [crop_size[0] / opt.resize[0], crop_size[1] / opt.resize[1]]]
            offset = [[0, 0], [dataset.crop_offset_x, dataset.crop_offset_y]]
            input = infer.preprocess(batch, net_resize = True, dyn_std = True)
            input = infer.to_tensor(input)

            with torch.no_grad():
                features, regression, classification, anchors = infer.model(input)

            out = infer.postprocess(input, scale, offset, anchors, regression, classification, opt.threshold, opt.iou_threshold, opt.tl_threshold)

            for i in range(len(obj_list)):
                top = 20 * i + 20
                left = 20

                for j in range(20):
                    cv2.line(result, (left, top + j), (left + 20, top + j), obj_color[i], 1)

                cv2.putText(result, obj_list[i], (left + 30, top + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, obj_color[i], 1)
            tl_list=[]
            if len(out['box']) != 0:
                for j in range(len(out['box'])):
                    x1 = max(0,int(out['box'][j][0]))
                    y1 = max(0,int(out['box'][j][1]))
                    x2 = min(1919,int(out['box'][j][2]))
                    y2 = min(1079,int(out['box'][j][3]))

                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    maxSize = max(width,height)
                    margin = int(maxSize * 0.1)
                    fetchSize = maxSize + margin * 2
                    cropMat = np.zeros((fetchSize,fetchSize,3),np.uint8)
                    cropMat[:,:] = (160,150,130)
                    str_score = '{:.2f}'.format(out['score'][j])

                    if out['class'][j] == 0:
                        if width > height:
                            gap = int((width-height)/2)
                            cropMat[gap+margin:gap+height+margin,margin:width+margin]=im[y1:y2,x1:x2]
                            cropMat = np.float32(cropMat)
                            cropMat =cv2.resize(cropMat,(32,32))
                            cropMat = cv2.cvtColor(cropMat, cv2.COLOR_BGR2RGB)[np.newaxis,:,:,:]
                            ort_input = {hl_model.get_inputs()[0].name:cropMat}
                            res=hl_model.run(None,ort_input)
                            cv2.rectangle(result, (x1, y1), (x2, y2), obj_color[np.argmax(res[0]) + 4], 2)
                            cv2.putText(result, str_score, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, obj_color[np.argmax(res[0]) + 4], 1)
                            tl_list.append(obj_list[np.argmax(res[0]) + 4])
                        else:
                            gap = int((height-width)/2)
                            cropMat[margin:height+margin,gap+margin:gap+width+margin]=im[y1:y2,x1:x2]
                            cropMat = np.float32(cropMat)
                            cropMat =cv2.resize(cropMat,(32,32))
                            cropMat = cv2.cvtColor(cropMat, cv2.COLOR_BGR2RGB)[np.newaxis,:,:,:]
                            ort_input = {vl_model.get_inputs()[0].name:cropMat}
                            res=vl_model.run(None,ort_input)
                            if np.argmax(res[0]) != 0:
                                cv2.rectangle(result, (x1, y1), (x2, y2), obj_color[np.argmax(res[0]) + 13], 2)
                                cv2.putText(result, str_score, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, obj_color[np.argmax(res[0]) +13], 1)
                                tl_list.append(obj_list[np.argmax(res[0]) + 13])
                            else:
                                cv2.rectangle(result, (x1, y1), (x2, y2), obj_color[np.argmax(res[0]) + 4], 2)
                                cv2.putText(result, str_score, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, obj_color[np.argmax(res[0]) + 4], 1)
                                tl_list.append(obj_list[np.argmax(res[0]) + 4])
                    
                    else:    
                        cv2.rectangle(result, (x1, y1), (x2, y2), obj_color[out['class'][j]], 2)
                        cv2.putText(result, str_score, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, obj_color[out['class'][j]], 1)

            cv2.rectangle(result, (dataset.crop_offset_x, dataset.crop_offset_y), (dataset.crop_offset_x + dataset.roi_width, dataset.crop_offset_y + dataset.roi_height), (255, 255, 255), 2)
            cv2.imshow('Det', result)

            save_det_path = os.path.join(save_det_fold, log_name + '_{:06d}.jpg'.format(dataset.frame_num))
            save_img_path = os.path.join(save_img_fold, log_name + '_{:06d}.png'.format(dataset.frame_num))

            print(save_img_path)

            cv2.imwrite(save_img_path, im)
            cv2.imwrite(save_det_path, result)

            save_json_path = os.path.splitext(save_img_path)[0] + '.json'

            convert_det_cnn_to_json(save_img_path, out, tl_list, save_json_path)

            cv2.waitKey(1)

if __name__ == '__main__':
    opt = get_args()

    test_video(opt)