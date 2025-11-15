"""
Create: 2021.09.13
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch
import argparse

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.ops import nms
from collections import Counter

from backbone import EfficientDetBackbone

from efficientdet.dataset import ADDataset
from efficientdet.dataset import NormalizerDyn
from efficientdet.dataset import ToTensor
from efficientdet.dataset import collater_v2

from efficientdet.utils import BBoxTransform
from efficientdet.utils import ClipBoxes

def boxes_intersect(box_a, box_b):
    if box_a[0] > box_b[2]:
        return False

    if box_b[0] > box_a[2]:
        return False

    if box_a[3] < box_b[1]:
        return False

    if box_a[1] > box_b[3]:
        return False

    return True

def get_intersection_area(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    return (xb - xa + 1) * (yb - ya + 1)

def get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def get_union_areas(box_a, box_b, inter_area = None):
    area_a = get_area(box_a)
    area_b = get_area(box_b)

    if inter_area is None:
        inter_area = get_intersection_area(box_a, box_b)

    return float(area_a + area_b - inter_area)

def calc_iou(box_a, box_b):
    if boxes_intersect(box_a, box_b) is False:
        return 0

    inter_area = get_intersection_area(box_a, box_b)
    union = get_union_areas(box_a, box_b, inter_area = inter_area)
    
    result = inter_area / union

    assert result >= 0

    return result

def mean_average_precision(pred_boxes, true_boxes, iou_threshold = 0.5, num_classes = 4):
    average_precisions = []
    epsilon = 1e-6

    ignore_gts = []

    for true_box in true_boxes:
        if true_box[1] == -1:
            ignore_gts.append(true_box)

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key = lambda x: x[2], reverse = True)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0

            best_gt_idx = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = calc_iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]))

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                TP[detection_idx] = 1
            else:
                best_ignore_iou = 0
                ignore_img = [bbox for bbox in ignore_gts if bbox[0] == detection[0]]

                for idx, ignore in enumerate(ignore_img):
                    iou = calc_iou(torch.tensor(detection[3:]), torch.tensor(ignore[3:]))

                    if iou > best_ignore_iou:
                        best_ignore_iou = iou

                if best_ignore_iou < iou_threshold:
                    FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_data_path",
                        type=list,
                        default=["od/test"])
    parser.add_argument("--tl_area_th",
                        type=int,
                        default=16)
    parser.add_argument("--p_area_th",
                        type=int,
                        default=25)
    parser.add_argument("--obj_area_th",
                        type=int,
                        default=49)
    parser.add_argument("--val_bin_path",
                        type=str,
                        default="val.bin")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--num_workers",
                        type=int,
                        default=1)
    parser.add_argument("--weights_path",
                        type=str,
                        default="weight/effdet_d1.pth")
    
    return parser.parse_args()
    

if __name__ == '__main__':
    args = make_parser()

    val_data_path = args.val_data_path

    tl_area_th = args.tl_area_th
    p_area_th = args.p_area_th
    obj_area_th = args.obj_area_th

    val_bin_path = args.val_bin_path

    batch_size = args.batch_size
    num_workers = args.num_workers

    val_params = {'batch_size': batch_size, 'shuffle': False, 'drop_last': True, 'collate_fn': collater_v2, 'num_workers': num_workers}

    val_dataset = ADDataset(val_data_path, transform = transforms.Compose([NormalizerDyn(), ToTensor()]), tl_area_th = tl_area_th, p_area_th = p_area_th, obj_area_th = obj_area_th, valid_crop = False, dump_file = val_bin_path)
    val_loader = DataLoader(val_dataset, **val_params)

    obj_list = ['traffic_light', 'pedestrian', 'car', 'cyclist']
    compound_coef = 1
    anchors_ratios = [(1.0, 4.0), (1.0, 2.0), (1.0, 1.0), (2.0, 1.0), (4.0, 1.0)]
    anchors_scales = [0.25, 0.5, 2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    model = EfficientDetBackbone(num_classes = len(obj_list), compound_coef = compound_coef, ratio = anchors_ratios, scales = anchors_scales, onnx_export = False)

    weights_path = args.weights_path

    model.load_state_dict(torch.load(weights_path), strict = False)
    model = model.cuda()
    model.eval()

    map_list = []

    step = 0.5

    for i in range(10):
        step = float(format(step, '.2f'))
        map_list.append(step)
        step += 0.05

    regress_boxes = BBoxTransform()
    clip_boxes = ClipBoxes()

    threshold = 0.4
    iou_threshold = 0.3
    tl_threshold = 0.3

    pred_boxes = []
    true_boxes = []

    for iter, data in enumerate(val_loader):
        with torch.no_grad():
            img = data['img']
            annot = data['annot']
            crop_img = data['crop_img']
            crop_annot = data['crop_annot']

            img = img.cuda()
            annot = annot.cuda()
            crop_img = crop_img.cuda()
            crop_annot = crop_annot.cuda()

            idx = 0

            for x, y in [(img, annot), (crop_img, crop_annot)]:
                _, regression, classification, anchors = model(x)

                transformed_anchors = regress_boxes(anchors, regression)
                transformed_anchors = clip_boxes(transformed_anchors, x)

                tl_class = classification[:, :, 0]
                other_class = classification[:, :, 1:]

                scores = torch.max(classification, dim = 2, keepdim = True)[0]
                other_scores = torch.max(other_class, dim = 2, keepdim = True)[0]

                tl_scores_over_thresh = (tl_class > tl_threshold)[:, :]
                scores_over_thresh = (other_scores > threshold)[:, :, 0]
                scores_over_thresh = scores_over_thresh + tl_scores_over_thresh

                classification_per = classification[0, scores_over_thresh[0, :], ...].permute(1, 0)
                transformed_anchors_per = transformed_anchors[0, scores_over_thresh[0, :], ...]
                scores_per = scores[0, scores_over_thresh[0, :], ...]
                anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold = iou_threshold)

                if anchors_nms_idx.shape[0] != 0:
                    scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim = 0)
                    boxes_ = transformed_anchors_per[anchors_nms_idx, :]
                else:
                    scores_ = torch.tensor([]).cuda()
                    classes_ = torch.tensor([]).cuda()
                    boxes_ = torch.tensor([]).cuda()

                frame_idx = iter * 2 + idx

                print(frame_idx)

                for i in range(boxes_.shape[0]):
                    box_info = [frame_idx, int(classes_[i]), float(scores_[i]), float(boxes_[i, 0]), float(boxes_[i, 1]), float(boxes_[i, 2]), float(boxes_[i, 3])]

                    pred_boxes.append(box_info)
                
                box_annot = y[0]

                for i in range(box_annot.shape[0]):
                    box_info = [frame_idx, int(box_annot[i, 4]), 1.0, float(box_annot[i, 0]), float(box_annot[i, 1]), float(box_annot[i, 2]), float(box_annot[i, 3])]

                    true_boxes.append(box_info)

                idx += 1

    map_50 = mean_average_precision(pred_boxes, true_boxes)

    print(map_50)              