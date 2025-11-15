"""
Create: 2021.09.17
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch
import cv2
import numpy as np

from torchvision.ops import nms

from backbone import EfficientDetBackbone

from efficientdet.utils import BBoxTransform
from efficientdet.utils import ClipBoxes

class InferEngine:
    def __init__(self, checkpoint_file, batch_size = 1, compound_coef = 1):
        obj_list = ['traffic_light', 'pedestrian', 'car', 'cyclist']
        model = EfficientDetBackbone(num_classes = len(obj_list), compound_coef = compound_coef)
        
        model.load_state_dict(torch.load(checkpoint_file))
        model.requires_grad_(False)
        model.eval()

        self.model = model.cuda()
        self.regress_boxes = BBoxTransform()
        self.clip_boxes = ClipBoxes()
        self.batch_size = batch_size

    def preprocess(self, input, net_resize = False, dyn_std = False):
        s_455 = True
        s_480 = False
        s_640 = False
        s_896 = False

        if not dyn_std:
            mean = (0.406, 0.456, 0.485)
            std = (0.225, 0.224, 0.229)

        out = []

        for i in range(len(input)):
            if net_resize:
                if s_455:
                    image = cv2.resize(input[i], (455, 256))
                    image = cv2.copyMakeBorder(image, 0, 0, 0, 57, cv2.BORDER_CONSTANT)
                elif s_480:
                    img_size = np.array(input[i].shape[:-1])
                    net_size = np.array((270, 480))
                    scale = np.amin(net_size / img_size)
                    image = cv2.resize(input[i], (0, 0), fx = scale, fy = scale)
                    pad_size = net_size - np.array(image.shape[:-1])
                    image = cv2.copyMakeBorder(image, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT)
                elif s_640:
                    image = cv2.resize(input[i], (640, 360))
                    image = cv2.copyMakeBorder(image, 0, 24, 0, 0, cv2.BORDER_CONSTANT)
                elif s_896:
                    image = cv2.resize(input[i], (896, 504))
                    image = cv2.copyMakeBorder(image, 0, 8, 0, 0, cv2.BORDER_CONSTANT)
                else:
                    image = cv2.resize(input[i], (910, 512))
                    image = cv2.copyMakeBorder(image, 0, 0, 0, 114, cv2.BORDER_CONSTANT)
            else:
                image = cv2.copyMakeBorder(input[i], 0, 8, 0, 0, cv2.BORDER_CONSTANT)

            image = image / 255

            if dyn_std:
                mean = image.mean(axis = (0, 1))
                std = image.std(axis = (0, 1))

                out.append((image - mean) / (std + 1e-5))
            else:
                out.append((image - mean) / std)

        return out

    def to_tensor(self, input):
        input = torch.stack([torch.from_numpy(image).float().cuda() for image in input])
        input = input.permute([0, 3, 1, 2])

        return input

    def postprocess(self, inputs, scales, offsets, anchors, regression, classification, threshold, iou_threshold, tl_threshold):
        transformed_anchors = self.regress_boxes(anchors, regression)
        transformed_anchors = self.clip_boxes(transformed_anchors, inputs)
        tl_class = classification[:, :, 0]
        other_class = classification[:, :, 1:]

        if other_class.shape[0] > 1:
            other_class[1, :, :] = 0

        scores = torch.max(classification, dim = 2, keepdim = True)[0]
        other_scores = torch.max(other_class, dim = 2, keepdim = True)[0]
        tl_scores_over_thresh = (tl_class > tl_threshold)[:, :]
        scores_over_thresh = (other_scores > threshold)[:, :, 0]
        scores_over_thresh = scores_over_thresh + tl_scores_over_thresh
        out = None
        cls_list = []
        anchor_list = []
        score_list = []

        for i in range(self.batch_size):
            if torch.eq(scores_over_thresh.sum(), 0):
                out = {'box': [], 'class': [], 'score': []}

                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...]
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            transformed_anchors_per[:, 0] = transformed_anchors_per[:, 0] * scales[i][0] + offsets[i][0]
            transformed_anchors_per[:, 1] = transformed_anchors_per[:, 1] * scales[i][1] + offsets[i][1]
            transformed_anchors_per[:, 2] = transformed_anchors_per[:, 2] * scales[i][0] + offsets[i][0]
            transformed_anchors_per[:, 3] = transformed_anchors_per[:, 3] * scales[i][1] + offsets[i][1]
            scores_per = scores[i, scores_over_thresh[i, :], ...]

            cls_list.append(classification_per)
            anchor_list.append(transformed_anchors_per)
            score_list.append(scores_per)

        if len(cls_list) == 0:
            out = {'box': [], 'class': [], 'score': []}

            return out

        cls_list = torch.cat(cls_list, dim = 0)
        anchor_list = torch.cat(anchor_list, dim = 0)
        score_list = torch.cat(score_list, dim = 0)
        nms_idx = nms(anchor_list, score_list[:, 0], iou_threshold = iou_threshold)

        if nms_idx.shape[0] != 0:
            score_, cls_ = cls_list[nms_idx, :].max(dim = 1)
            box_ = anchor_list[nms_idx, :]
            out = {'box': box_.cpu().numpy().tolist(), 'class': cls_.cpu().numpy().tolist(), 'score': score_.cpu().numpy().tolist()}
        else:
            out = {'box': [], 'class': [], 'score': []}

        return out
