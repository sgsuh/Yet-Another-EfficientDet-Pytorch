"""
Create: 2021.09.15
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from utils.utils import boolean_string
from utils.utils import init_weights
from utils.utils import replace_w_sync_bn
from utils.utils import CustomDataParallel

from efficientdet.dataset import collater
from efficientdet.dataset import ADDataset
from efficientdet.dataset import Augmenter
from efficientdet.dataset import NormalizerDyn
from efficientdet.dataset import ToTensor

from backbone import EfficientDetBackbone

from efficientdet.loss import FocalLoss

from utils.sync_batchnorm import patch_replication_callback

obj_list = ['traffic_light', 'pedestrian', 'car', 'cyclist']

def get_args():
    parser = argparse.ArgumentParser('EfficientDet PyTorch')

    parser.add_argument('--num_gpus', type = int, default = 1)
    parser.add_argument('--num_epochs', type = int, default = 500)
    parser.add_argument('--date', type = str, default = '211105')
    parser.add_argument('--train_bin_path', type = str, default = 'train.bin')
    parser.add_argument('--val_bin_path', type = str, default = 'val.bin')
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--num_workers', type = int, default = 1)
    parser.add_argument('--train_list', type = str, default = 'list/train.txt')
    parser.add_argument('--val_list', type = str, default = 'list/val.txt')
    parser.add_argument('--tl_area_th', type = int, default = 16)
    parser.add_argument('--p_area_th', type = int, default = 25)
    parser.add_argument('--obj_area_th', type = int, default = 49)
    parser.add_argument('--compound_coef', type = int, default = 1)
    parser.add_argument('--weights_path', type = str, default = 'weight/effdet_d1.pth')
    parser.add_argument('--head_only', type = boolean_string, default = False)
    parser.add_argument('--optim', type = str, default = 'adamw')
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--val_interval', type = int, default = 1)
    parser.add_argument('--es_min_delta', type = float, default = 0.0)
    parser.add_argument('--es_patience', type = int, default = 0)
    parser.add_argument('--valid_roi', type = boolean_string, default = True)

    args = parser.parse_args()

    return args

class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.criterion = FocalLoss()
        self.model = model
    
    def forward(self, imgs, annotations):
        _, regression, classification, anchors = self.model(imgs)

        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)

        return cls_loss, reg_loss

def train_iteration(opt, train_loader, model, optimizer, scheduler, step, epoch):
    progress_bar = tqdm(train_loader)
    epoch_loss = []
    num_iter_per_epoch = len(train_loader)

    for iter, data in enumerate(progress_bar):
        try:
            imgs = data['img']
            annot = data['annot']

            if opt.num_gpus == 1:
                imgs = imgs.cuda()
                annot = annot.cuda()

            optimizer.zero_grad()

            cls_loss, reg_loss = model(imgs, annot)

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            
            loss = cls_loss + reg_loss

            if loss == 0 or not torch.isfinite(loss):
                continue

            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            progress_bar.set_description('Step: {}, Epoch: {}/{}, Iter: {}/{}, Cls Loss: {:.5f}, Reg Loss: {:.5f}, Total Loss: {:.5f}'.format(step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(), reg_loss.item(), loss.item()))

            current_lr = optimizer.param_groups[0]['lr']

            print('Learning Rate: {}'.format(current_lr))

            step += 1
        except Exception as e:
            print(e)

            continue

    return step

def save_checkpoint(model, save_path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), save_path)
    else:
        torch.save(model.model.state_dict(), save_path)

def train(opt):
    save_path = 'weight/' + opt.date

    os.makedirs(save_path, exist_ok = True)

    if opt.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    log_path = 'logs/' + opt.date

    os.makedirs(log_path, exist_ok = True)

    train_params = {'batch_size': opt.batch_size, 'shuffle': True, 'drop_last': True, 'collate_fn': collater, 'num_workers': opt.num_workers}
    val_params = {'batch_size': opt.batch_size, 'shuffle': False, 'drop_last': True, 'collate_fn': collater, 'num_workers': opt.num_workers}

    train_data_path = []

    with open(opt.train_list, 'r') as f:
        for line in f:
            path = line.strip()

            train_data_path.append(path)

    val_data_path = []

    with open(opt.val_list, 'r') as f:
        for line in f:
            path = line.strip()

            val_data_path.append(path)
    
    train_dataset = ADDataset(train_data_path, transform = transforms.Compose([Augmenter(), NormalizerDyn(), ToTensor()]), tl_area_th = opt.tl_area_th, p_area_th = opt.p_area_th, obj_area_th = opt.obj_area_th, valid_crop = opt.valid_roi, dump_file = opt.train_bin_path)
    val_dataset = ADDataset(val_data_path, transform = transforms.Compose([NormalizerDyn(), ToTensor()]), tl_area_th = opt.tl_area_th, p_area_th = opt.p_area_th, obj_area_th = opt.obj_area_th, valid_crop = opt.valid_roi, dump_file = opt.val_bin_path)

    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **val_params)

    anchors_ratios = [(1.0, 4.0), (1.0, 2.0), (1.0, 1.0), (2.0, 1.0), (4.0, 1.0)]
    anchors_scales = [0.25, 0.5, 2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    model = EfficientDetBackbone(num_classes = len(obj_list), compound_coef = opt.compound_coef, ratio = anchors_ratios, scales = anchors_scales, onnx_export = False)

    if opt.weights_path is not None:
        try:
            ret = model.load_state_dict(torch.load(opt.weights_path), strict = False)
        except RuntimeError as e:
            print('[Warning] Ignoring {}'.format(e))

        print('[Info] Loaded Weights: {}'.format(os.path.basename(opt.weights_path)))
    else:
        print('[Info] Initialize Weight')

        init_weights(model)

    if opt.head_only:
        def freeze_backbone(m):
            class_name = m.__class__.__name__

            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in class_name:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)

        print('[Info] Freeze Backbone')

    use_sync_bn = False

    if opt.num_gpus > 1 and opt.batch_size // opt.num_gpus < 4:
        model.apply(replace_w_sync_bn)

        use_sync_bn = True

    model = ModelWithLoss(model)

    if opt.num_gpus > 0:
        model = model.cuda()

        if opt.num_gpus > 1:
            model = CustomDataParallel(model, opt.num_gpus)

            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum = 0.9, nesterov = True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, verbose = True)

    model.train()

    step = 0
    best_loss = 1e5
    best_epoch = 0

    try:
        for epoch in range(opt.num_epochs):
            step = train_iteration(opt, train_loader, model, optimizer, scheduler, step, epoch)

            save_checkpoint(model, os.path.join(save_path, 'effdet_d{}_{:04d}.pth'.format(opt.compound_coef, epoch)))

            print('Save Checkpoint')

            if epoch % opt.val_interval == 0:
                model.eval()

                loss_reg_ls = []
                loss_cls_ls = []

                for iter, data in enumerate(val_loader):
                    with torch.no_grad():
                        img = data['img']
                        annot = data['annot']

                        if opt.num_gpus == 1:
                            img = img.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(img, annot)

                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss

                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_cls_ls.append(cls_loss.item())
                        loss_reg_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_cls_ls)
                reg_loss = np.mean(loss_reg_ls)
                loss = cls_loss + reg_loss

                print('Val Epoch: {}/{}, Cls Loss: {:.5f}, Reg Loss: {:.5f}, Total Loss: {:.5f}'.format(epoch, opt.num_epochs, cls_loss, reg_loss, loss))

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, os.path.join(save_path, 'effdet_d{}_{:04d}_best.pth'.format(opt.compound_coef, epoch)))

                scheduler.step(loss)
                model.train()

                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop Train at Epoch {}. The Lowest Loss {}'.format(epoch, best_loss))

                    break
    except KeyboardInterrupt:
        save_checkpoint(model, os.path.join(save_path, 'effdet_d{}_{:04d}_{}.pth'.format(opt.compound_coef, epoch, step)))

if __name__ == '__main__':
    opt = get_args()
    
    train(opt)

    print('Complete Train')