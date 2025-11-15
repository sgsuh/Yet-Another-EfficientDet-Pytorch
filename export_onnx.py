"""
Create: 2021.09.23
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import argparse
import torch
import numpy as np
import cv2
import onnx

from torch import nn
from torch.autograd import Variable

from backbone import EfficientDetBackbone
from efficientdet.utils import generate_anchors
from efficientdet.utils import BBoxTransform
from efficientdet.utils import ClipBoxes_fix

obj_list = ['traffic_light', 'pedestrian', 'car', 'cyclist']

class OD(nn.Module):
    def __init__(self, opt, net_size_hw):
        super(OD, self).__init__()

        self.model = EfficientDetBackbone(compound_coef = opt.compound_coef, num_classes = len(obj_list), onnx_export = True).cuda()
        self.anchors = torch.from_numpy(generate_anchors(image_size = net_size_hw, anchor_scale = 4)).float().cuda()
        self.net_only = opt.net_only

        if not self.net_only:
            self.regress_boxes = BBoxTransform()
            
            self.clip_boxes = ClipBoxes_fix(net_size_hw[1], net_size_hw[0])

        self.mean = Variable(torch.Tensor((0.406, 0.456, 0.485))).cuda()
        self.std = Variable(torch.Tensor((0.225, 0.224, 0.229))).cuda()

    def postprocess(self, reg, cls):
        transformed_anchors = self.regress_boxes(self.anchors, reg)
        transformed_anchors = self.clip_boxes(transformed_anchors)

        return cls, transformed_anchors

    def forward(self, input):
        x = input.permute(0, 3, 1, 2)

        with torch.no_grad():
            regression, classification = self.model(x)

        if not self.net_only:
            regression, classification = self.postprocess(regression, classification)

        return regression, classification

def get_args():
    parser = argparse.ArgumentParser('EfficientDet PyTorch')

    parser.add_argument('--model_file', type = str, default = 'weight/effdet_d1.pth')
    parser.add_argument('--net_only', action = 'store_true', default = False)
    parser.add_argument('--export_file_name', type = str, default = 'onnx_file/effdet_b1.onnx')
    parser.add_argument('--opset', type = int, default = 9)
    parser.add_argument('--in_size', type = int, default = 456)
    parser.add_argument('--compound_coef', type = int, default = 1)

    args = parser.parse_args()

    return args

def export(opt):
    dummy_input = np.zeros((1080, 1920, 3), dtype = np.uint8)
    dummy_input2 = np.zeros((1080, 1920, 3), dtype = np.uint8)

    if opt.in_size == 512:
        net_size_hw = (256, 512)
        dummy_input = cv2.resize(dummy_input, (512, 288))
        dummy_input = dummy_input[:-32, :, :]
        dummy_input2 = cv2.resize(dummy_input2, (512, 288))
        dummy_input2 = dummy_input2[:-32, :, :]
    elif opt.in_size == 896:
        net_size_hw = (512, 896)
        dummy_input = cv2.resize(dummy_input, (896, 504))
        dummy_input = cv2.copyMakeBorder(dummy_input, 0, 8, 0, 0, cv2.BORDER_CONSTANT)
        dummy_input2 = cv2.resize(dummy_input2, (896, 504))
        dummy_input2 = cv2.copyMakeBorder(dummy_input2, 0, 8, 0, 0, cv2.BORDER_CONSTANT)
    elif opt.in_size == 640:
        net_size_hw = (384, 640)
        dummy_input = cv2.resize(dummy_input, (640, 360))
        dummy_input = cv2.copyMakeBorder(dummy_input, 0, 24, 0, 0, cv2.BORDER_CONSTANT)
        dummy_input2 = cv2.resize(dummy_input2, (640, 360))
        dummy_input2 = cv2.copyMakeBorder(dummy_input2, 0, 24, 0, 0, cv2.BORDER_CONSTANT)
    elif opt.in_size == 456:
        net_size_hw = (256, 512)
        dummy_input = cv2.resize(dummy_input, (455, 256))
        dummy_input = cv2.copyMakeBorder(dummy_input, 0, 0, 0, 57, cv2.BORDER_CONSTANT)
        dummy_input2 = cv2.resize(dummy_input2, (455, 256))
        dummy_input2 = cv2.copyMakeBorder(dummy_input2, 0, 0, 0, 57, cv2.BORDER_CONSTANT)
    else:
        raise ValueError('Error In Size')

    det = OD(opt, net_size_hw)

    det.model.load_state_dict(torch.load(opt.model_file))
    det.model.eval()

    onnx_input = np.expand_dims(dummy_input, axis = 0)
    onnx_input = torch.from_numpy(onnx_input).float().cuda()

    torch.onnx.export(det, onnx_input, opt.export_file_name, export_params = True, opset_version = opt.opset, input_names = ['onnx_input'], output_names = ['regression', 'classification'], verbose = True)

    original_model = onnx.load(opt.export_file_name)

    onnx.checker.check_model(original_model)

if __name__ == '__main__':
    opt = get_args()

    export(opt) 