import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import pandas as pd
from scipy.signal import find_peaks

def load_spectra(file, nvals):
    df = pd.read_csv(file)
    ampl = np.array(df['A'])
    if np.amax(ampl) > 0:
        ampl /= np.amax(ampl)
    assert ampl.shape[0] == nvals
    return ampl

def load_spectra2(file, nvals):
    df = pd.read_csv(file)
    fs = np.array(df['f'])
    ampl = np.array(df['A'])
    peaks, properties = find_peaks(x=ampl, distance=2)
    pps = [(idx, ampl[idx], fs[idx]) for idx in peaks]
    pps = sorted(pps, key=lambda x: -x[1])
    # ol = len(pps)
    while len(pps) < nvals:
        pps.append((0, max(fs), 0))
   #  ml = len(pps)
    pps = pps[:nvals]
    # print(f"{ol} -> {ml} -> {len(pps)} -> \n{pps}\n")


    return np.array([x[2] for x in pps]) / max(fs)

class DatasetSTACK(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSTACK, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        self.is_numpy =  self.paths_H[0].split('.')[-1] == 'npy'

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, 1, self.is_numpy)

        # plt.imshow(img_H)
        # plt.show()
        # print(H_path)
        dn = os.path.dirname(H_path)
        dn2 = os.path.dirname(dn)

        spec_path = os.path.join(dn2, "Spectra_Full", os.path.basename(H_path).split('.')[0] + '.csv')
        # print(spec_path)
        # assert os.path.isfile(spec_path)
        # plt.imshow(img_H)
        # plt.show()

        vec = load_spectra2(spec_path, self.n_channels - 1)

        # vec *= 0
        if not self.is_numpy:
            img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # plt.imshow(img_H)
        # plt.show()

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, 1, self.is_numpy)
            if not self.is_numpy:
                img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # Stack LR-Image with Map
        # print(img_L.shape)


        newL = np.zeros((img_L.shape[0], img_L.shape[1], self.n_channels))
        newL[:, :, 0] = img_L[:, :, 0]
        newL[:, :, 1:] = vec

        img_L = newL
        # if img_L.shape[2] != 65:
        #     print(img_L.shape)
        #     return self.__getitem__(random.randint(0, len(self.paths_H)))


        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)


        if L_path is None:
            L_path = H_path
        # with open('stats_ds.csv', 'a') as f:
        #     f.write(f"{self.paths_H[index]};{img_L.shape[0]};{img_L.shape[1]};{img_L.shape[2]};{img_H.shape[0]};{img_H.shape[1]};{img_H.shape[2]}\n")

        # if img_L.shape[0] != 65 or img_L.shape[1] != 64 or img_L.shape[2] != 64 or img_H.shape[0] != 1 or img_H.shape[1] != 64 or img_H.shape[2] != 64:
        #     print("Err", img_L.shape, img_H.shape, index)
        #     return self.__getitem__(random.randint(0, len(self.paths_H)))

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
