"""
Create: 2021.09.17
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import numpy as np
import struct

class LogLoader(object):
    def __init__(self, file_path):
        self.width = 1920
        self.height = 1080
        self.channel = 3
        self.img_size = self.width * self.height * self.channel
        self.f = open(file_path, 'rb')
        self.num_of_tl = 0
        self.tick = 0
        self.save_flag = 0
        self.tl_img_coordi_x = np.zeros((5), dtype = np.float)
        self.tl_img_coordi_y = np.zeros((5), dtype = np.float)
        self.cx = 0
        self.cy = 0
        self.fx = 0
        self.fy = 0
        self.radial_distort = np.zeros((3), dtype = np.float)
        self.tangential_distort = np.zeros((2), dtype = np.float)
        self.dx = 0
        self.dy = 0
        self.step = 4
        self.fps = 30
        self.frame_num = 0
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        self.roi_width = 0
        self.roi_height = 0

    def __getitem__(self, idx):
        try:
            data = self.f.read(self.img_size)
        except:
            self.f.close()

            raise ValueError('Cannot Read Bin Data')

        im = np.frombuffer(data, dtype = np.uint8)

        if im.shape[0] != self.img_size:
            self.f.close()

            return None, None

        idx = 0
        self.num_of_tl = int(struct.unpack('i', data[idx:idx + self.step])[0])
        idx += self.step
        self.tick = int(struct.unpack('i', data[idx:idx + self.step])[0])
        idx += self.step
        self.save_flag = int(struct.unpack('i', data[idx:idx + self.step])[0])
        idx += self.step

        for i in range(5):
            self.tl_img_coordi_x[i] = float(struct.unpack('f', data[idx:idx + self.step])[0])
            idx += self.step
            self.tl_img_coordi_y[i] = float(struct.unpack('f', data[idx:idx + self.step])[0])
            idx += self.step

        self.cx = float(struct.unpack('f', data[idx:idx + self.step])[0])
        idx += self.step
        self.cy = float(struct.unpack('f', data[idx:idx + self.step])[0])
        idx += self.step
        self.fx = float(struct.unpack('f', data[idx:idx + self.step])[0])
        idx += self.step
        self.fy = float(struct.unpack('f', data[idx:idx + self.step])[0])
        idx += self.step

        for i in range(3):
            self.radial_distort[i] = float(struct.unpack('f', data[idx:idx + self.step])[0])
            idx += self.step

        for i in range(2):
            self.tangential_distort[i] = float(struct.unpack('f', data[idx:idx + self.step])[0])
            idx += self.step
        
        self.dx = float(struct.unpack('f', data[idx:idx + self.step])[0])
        idx += self.step
        self.dy = float(struct.unpack('f', data[idx:idx + self.step])[0])
        im = im.reshape(self.height, self.width, self.channel)
        roi_x = int(self.tl_img_coordi_x[0])
        roi_y = int(self.tl_img_coordi_y[0])
        roi_width = int(self.tl_img_coordi_x[1])
        roi_height = int(self.tl_img_coordi_y[1])
        roi_offset_x = int(self.tl_img_coordi_x[2])
        roi_offset_y = int(self.tl_img_coordi_y[2])

        self.crop_offset_x = roi_x + roi_offset_x
        self.crop_offset_y = roi_y + roi_offset_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        crop = im[self.crop_offset_y:self.crop_offset_y + roi_height, self.crop_offset_x:self.crop_offset_x + roi_width, :].copy()
        self.frame_num += 1

        return im, crop
