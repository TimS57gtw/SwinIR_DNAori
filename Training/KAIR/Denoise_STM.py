import shutil

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm
import Load_STM_Files
import cv2
from PIL import Image
from ApplySwinIR import find_json
import argparse
import copy

import torch
from data.select_dataset import define_Dataset
from models.model_plain import ModelPlain as M
from torch.utils.data import DataLoader
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option
from torch.utils.data import DataLoader
import logging


def load_file(fn, linespeed=1, tolerance=0.01, scn_rng=None):
    date = datetime.now()

    ext = os.path.basename(fn).split('.')[-1]

    if ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        arr = np.array(Image.open(fn))
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]
        scn_rng = 1, 1
        scn_dir = 'down'
    elif ext == 'sxm':
        sxm = Load_STM_Files.SXM(fn)
        arr = sxm.data
        linespeed = sxm.time_per_line
        date = sxm.date
        scn_rng = sxm.real_size
        scn_dir = sxm.scan_dir
        if sxm.scan_dir == 'up':
            arr = np.flipud(arr)
    elif ext.lower() == 'sm2':
        sm2 = Load_STM_Files.SM2(fn)
        arr = sm2.data
        linespeed = sm2.time_per_line
        date = sm2.date
        scn_rng = sm2.real_size
        scn_dir = sm2.scan_dir

        if sm2.scan_dir == 'up':
            arr = np.flipud(arr)
    elif ext.lower() == 'nid':
        nid = Load_STM_Files.NID(fn)
        arr = nid.data
        linespeed = nid.time_per_line
        date = nid.date
        scn_rng = nid.real_size
        scn_dir = nid.scan_dir

        if nid.scan_dir == 'up':
            arr = np.flipud(arr)

    else:
        raise NotImplementedError(f"Unknown Filetype: {ext}")

    if scn_rng is None:
        scn_rng = 1, 1
    afl = arr.flatten()
    afl = sorted(afl)
    mini = afl[int(tolerance * len(afl))]
    maxi = afl[int((1 - tolerance) * len(afl))]
    arr = arr.astype(float)
    arr -= mini
    arr /= maxi - mini
    arr = np.clip(arr, 0, 1)

    if PLOT:
        plt.imshow(arr, cmap='gray')
        plt.show()

    return arr, linespeed, date, scn_rng, scn_dir


def load_closest_spectrum(fld, date):
    def print_time_format(secs):
        if type(secs) is timedelta:
            secs = timedelta.total_seconds(secs)
        sign = np.sign(secs)
        secs /= sign
        ds = secs // (3600 * 24)
        secs = secs % (3600 * 24)
        hs = secs // 3600
        secs = secs % 3600
        mins = secs // 60
        secs = secs % 60

        ds = int(ds)
        hs = int(hs)
        mins = int(mins)
        secs = int(secs)

        ss = "-" if sign < 0 else "+"
        return f"{ss}{hs}:{mins}:{secs}" if ds == 0 else f"{ss} {ds}d {hs}:{mins}:{secs}"

    mode = 'nearest'
    files = []
    times = []
    diffs = []
    spcs = []

    for file in os.listdir(fld):
        spc = Load_STM_Files.Spectrum_dat(os.path.join(fld, file))
        files.append(file)
        times.append(spc.timestamp)

        timediff = (spc.timestamp - date).total_seconds()
        diffs.append(timediff)
        spcs.append(spc)

    if mode == 'closest_before':
        mind = diffs[0]
        cls = spcs[0]
        time = times[0]
        file = None
        for i in range(len(diffs)):
            if diffs[i] < 0 and diffs[i] > mind:
                mind = diffs[i]
                cls = spcs[i]
                time = times[i]
                file = files[i]


    elif mode == 'nearest':
        absdiff = np.infty
        cls = None
        time = None
        file = None
        for i in range(len(diffs)):
            if abs(diffs[i]) < absdiff:
                absdiff = abs(diffs[i])
                cls = spcs[i]
                time = times[i]
                file = files[i]

    else:
        raise NotImplementedError
    if VERBOSE: print(f"Found closest Spectrum at {time}: {print_time_format(time - date)}\nFile: {file}")

    if PLOT:
        plt.plot(cls.freq, cls.ampl)
        plt.xlabel(cls.f_head)
        plt.ylabel(cls.a_head)
        plt.show()

    return cls


def max_interp(x, oldx, oldy):
    ys = np.zeros_like(x, dtype=float)
    for ox, oy in zip(oldx, oldy):
        p = np.argmin(np.abs(ox - x))
        ys[p] = max(oy, ys[p])
    # plt.plot(x, ys, label='new')
    # plt.plot(oldx, oldy, label='old')
    # plt.legend()
    # plt.xlim(0, max(x))
    # plt.title("Max IP")
    # plt.show()

    return ys


def transform_spectrum(spec, linespeed):
    old_freqs = spec.freq
    if VERBOSE: print("Linespeed: ", linespeed)
    ampls = spec.ampl

    newfreqs = old_freqs / linespeed

    old_freqs = old_freqs[1:]
    newfreqs = newfreqs[1:]
    ampls = ampls[1:]

    if PLOT:
        plt.plot(old_freqs, ampls, label='old')
        plt.plot(newfreqs, ampls, label='new')
        plt.legend()
        plt.show()

    interp_freqs = np.arange(2, 129, 2)

    interp_ampl = max_interp(interp_freqs, newfreqs, ampls)
    if PLOT:
        plt.plot(interp_freqs, interp_ampl)
        plt.show()

    interp_ampl = np.random.random(interp_ampl.shape)

    return (interp_freqs, interp_ampl)


def prepare_model(modelfld):
    json_path = find_json(os.path.join(modelfld, 'options'))
    if VERBOSE: print("Found JSON: ", json_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=False)  # Train True

    opt['dist'] = parser.parse_args().dist
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    border = opt['scale']
    if opt['rank'] == 0:
        option.save(opt)
    opt = option.dict_to_nonedict(opt)
    if opt['rank'] == 0:
        logger_name = 'test'  # train
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    maxst = -1
    for elem in os.listdir(os.path.join(modelfld, 'models')):
        prts = elem.split('_')
        step = int(prts[0])
        if step > maxst:
            maxst = step

    E_pth = os.path.join(modelfld, 'models', f"{maxst}_E.pth")
    G_pth = os.path.join(modelfld, 'models', f"{maxst}_G.pth")
    O_pth = os.path.join(modelfld, 'models', f"{maxst}_optimizerG.pth")

    opt['path']['pretrained_netG'] = G_pth
    opt['path']['pretrained_netE'] = E_pth
    opt['path']['pretrained_optimizerG'] = O_pth
    model = M(opt)
    model.init_train()

    return opt, model


def apply_denoising(arr, spec, opt, model, temp_dir, fn=None, detailed_dir=None, mode='PNG'):
    img_fld = os.path.join(temp_dir, 'Img')
    spc_fld = os.path.join(temp_dir, 'Spectra')
    res_fld = os.path.join(temp_dir, 'res')

    os.makedirs(img_fld, exist_ok=True)
    os.makedirs(spc_fld, exist_ok=True)
    os.makedirs(res_fld, exist_ok=True)

    if mode == "npy":
        np.save(os.path.join(img_fld, 'Image.npy'))
    elif mode.lower() == 'png':
        plt.imsave(os.path.join(img_fld, 'Image.png'), arr, cmap='gray')
    else:
        raise NotImplementedError(f"Unknown Mode {mode}")

    # Save Spectrum
    frq, amp = spec
    df = pd.DataFrame({
        'f': frq,
        'A': amp
    })
    df.to_csv(os.path.join(spc_fld, 'Image.csv'))

    dataset_opt = opt['datasets']['test']
    dataset_opt['dataroot_L'] = os.path.join(temp_dir, 'Img')
    dataset_opt['dataroot_H'] = dataset_opt['dataroot_L']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    assert len(test_loader) == 1
    data = [x for x in test_loader][0]
    model.feed_data(data)
    model.test()

    visuals = model.current_visuals()
    E_img = util.tensor2uint(visuals['E'])
    H_img = util.tensor2uint(visuals['H'])
    L_img = util.tensor2uint(visuals['L'])
    E_img = E_img
    L_img = L_img[:, :, 0]

    E_fl = copy.deepcopy(E_img)
    H_fl = copy.deepcopy(H_img)
    L_fl = copy.deepcopy(L_img)
    E_fl = E_fl.astype(float)
    H_fl = H_fl.astype(float)
    L_fl = L_fl.astype(float)
    diff = E_fl - H_fl

    if detailed_dir is not None:
        dd = os.path.join(detailed_dir, fn)
        os.makedirs(dd, exist_ok=True)
        util.imsave(E_img, os.path.join(dd, "E.png"))
        util.imsave(H_img, os.path.join(dd, "H.png"))
        util.imsave(L_img, os.path.join(dd, "L.png"))
        df.to_csv(os.path.join(dd, 'Spectrum.csv'))

        current_psnr = util.calculate_psnr(E_img, H_img, border=opt['scale'])
        ssim = util.calculate_ssim(E_img, H_img, border=opt['scale'])
        mse = util.calculate_mse(E_img, H_img, border=opt['scale'])

        plt.imshow(diff, cmap='seismic')
        plt.colorbar()
        plt.title(f"PSNR: {current_psnr:.3f}dB")
        plt.savefig(os.path.join(dd, 'residual.png'))
        plt.clf()

        mind = 0
        maxd = 0
        for i in range(3):
            if i == 0:
                ls = L_fl
            elif i == 1:
                ls = E_fl
            else:
                ls = H_fl

            for j in range(3):
                if j == 0:
                    rs = L_fl
                elif j == 1:
                    rs = E_fl
                else:
                    rs = H_fl

            diff = rs - ls
            if np.amin(diff) < mind:
                mind = np.amin(diff)
            if np.amax(diff) > maxd:
                maxd = np.amax(diff)

            mind = min(mind, -maxd)
            maxd = -mind

        fig, axs = plt.subplots(3, 3)

        for i in range(3):
            if i == 0:
                ls = L_fl
            elif i == 1:
                ls = E_fl
            else:
                ls = H_fl

            for j in [2, 1, 0]:
                if j == 0:
                    rs = L_fl
                elif j == 1:
                    rs = E_fl
                else:
                    rs = H_fl

                if i == j:
                    axs[i, j].imshow(ls, cmap='gray', vmin=0, vmax=255)

                else:
                    diff = ls - rs
                    pcm = axs[i, j].imshow(diff, cmap='seismic', vmin=mind, vmax=maxd)
                    # if i == 1 and j == 2:
                    #     fig.colorbar(pcm, ax=axs[i, j], shrink=3.0)
                    # if j == 2 and i != 1:
                    #     fig.colorbar(pcm, ax=axs[i, j], shrink=0.01)

                axs[i, j].axis('off')
                if i == 0:
                    if j == 0:
                        axs[i, j].set_title("L")
                    elif j == 1:
                        axs[i, j].set_title("E")
                    else:
                        axs[i, j].set_title("H")

                if j == 0:
                    if i == 0:
                        axs[i, j].set_xlabel("L")
                    elif i == 1:
                        axs[i, j].set_xlabel("E")
                    else:
                        axs[i, j].set_xlabel("H")

        # fig.colorbar(pcm, ax=[axs[0, 1], axs[0, 2], axs[1, 0], axs[2, 0], axs[2, 1], axs[1, 2]])
        # fig.colorbar(ax=axs)
        fig.colorbar(pcm, ax=axs.ravel().tolist())
        plt.suptitle(f"PSNR: {current_psnr:.3f}dB, MSE: {mse:.3f}, SSIM: {ssim:.3f}")
        plt.savefig(os.path.join(dd, 'comp.png'))
        plt.clf()

    shutil.rmtree(temp_dir)

    return E_fl


def save_sxm(arr, outf, linespeed=1.0, date=None, range=None, scn_dir="Down"):

    Load_STM_Files.My_SXM().write_sxm(outf, data=arr, range=range, date=date, scn_dir=scn_dir, linespeed=linespeed)


def subsample_image(arr, ims_size=256):
    if arr.shape[0] == ims_size:
        return arr

    arr = cv2.resize(arr, (ims_size, ims_size), interpolation=cv2.INTER_NEAREST)
    return arr

def measure(arr, axis=0):
    ret = []
    def m(arr):
        larr = copy.deepcopy(arr)
        larr = sorted(larr)
        lng = len(larr)
        rng = 0.2
        medarr = larr[int((0.5-rng)*lng):int((0.5+rng)*lng)]
        return np.average(medarr)

    if axis==1:
        for i in range(arr.shape[0]):
            ret.append(m(arr[i, :]))
    elif axis==0:
        for i in range(arr.shape[1]):
            ret.append(m(arr[:, i]))

    # print(ret)
    return ret



def preprocess(arr):
    plt.switch_backend('TkAgg')
    if PLOT:
        plt.imshow(arr)
        plt.show()

    lines = measure(arr, axis=1)
    if PLOT:
        plt.plot(lines, label='meas')
        plt.plot(np.median(arr, axis=1), label='med')
        plt.legend()
        plt.show()

    for i in range(arr.shape[0]):
        arr[i, :] -= lines[i]

    lines = measure(arr, axis=1)
    if PLOT:
        plt.plot(lines, label='meas')
        plt.plot(np.median(arr, axis=1), label='med')
        plt.legend()

        plt.imshow(arr)
        plt.title("Line Corr")
        plt.show()

    lines = measure(arr, axis=0)
    if PLOT:
        plt.plot(lines, label='meas')
        plt.plot(np.median(arr, axis=0), label='med')
        plt.legend()
        plt.show()

    for i in range(arr.shape[1]):
        arr[:, i] -= lines[i]

    lines = measure(arr, axis=0)
    if PLOT:
        plt.plot(lines, label='meas')
        plt.plot(np.median(arr, axis=0), label='med')
        plt.legend()
        plt.show()

        plt.imshow(arr)
        plt.title("Flatten")
        plt.show()


    return arr


PLOT=False
VERBOSE=True
if __name__ == "__main__":
    infld = r"D:\Dateien\KI_Speicher\SampleSXM\Test"
    specra_fld = os.path.join(infld, 'spectra')
    modelfld = r"D:\Dateien\KI_Speicher\SwinSTM_Denoise\Runs\swinir_sr_CL_H128_W8_full"
    outfld = os.path.join(infld, 'denoised_RandSpec')
    out_ori_png = os.path.join(outfld, 'Original')
    out_ori_prp = os.path.join(outfld, 'Preproc')
    out_den_png = os.path.join(outfld, 'PNG')
    out_den_sxm = os.path.join(outfld, 'SXM')
    out_den_det = os.path.join(outfld, "Detail")
    os.makedirs(out_den_png, exist_ok=True)
    os.makedirs(out_den_sxm, exist_ok=True)
    os.makedirs(out_ori_png, exist_ok=True)
    os.makedirs(out_ori_prp, exist_ok=True)
    tempdir = os.path.join(outfld, 'temp')
    os.makedirs(tempdir, exist_ok=True)

    os.makedirs(outfld, exist_ok=True)

    opt, model = prepare_model(modelfld)
    files = [x for x in os.listdir(infld)]
    for file in tqdm(files):
        try:
            if VERBOSE: print("File: ", file)
            if not os.path.isfile(os.path.join(infld, file)):
                continue
            arr, linespeed, date, scan_range, scn_dir = load_file(os.path.join(infld, file))

            plt.imsave(os.path.join(out_ori_png, file.split('.')[0] + '.png'), arr, cmap='gray')
            plt.clf()

            arr = preprocess(arr)

            plt.imsave(os.path.join(out_ori_prp, file.split('.')[0] + '.png'), arr, cmap='gray')
            plt.clf()

            spec = load_closest_spectrum(specra_fld, date)

            spec = transform_spectrum(spec, linespeed)
            arr = subsample_image(arr, ims_size=256)

            if PLOT:
                plt.imshow(arr)
                plt.show()



            denoised = apply_denoising(arr=arr, spec=spec, opt=opt, model=model, temp_dir=tempdir, fn=file.split('.')[0],
                                       detailed_dir=out_den_det, mode='PNG')

            plt.imsave(os.path.join(out_den_png, file.split('.')[0] + '.png'), denoised, cmap='gray')
            plt.clf()

            save_sxm(denoised, os.path.join(out_den_sxm, file.split('.')[0] + '.sxm'), linespeed, date, scan_range, scn_dir=scn_dir)

        except Exception as e:
            print(f"Issue for file {file}")
            print(e)
            # raise e