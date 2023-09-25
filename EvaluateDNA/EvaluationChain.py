import os
import random
import shutil
import sys
import time
import warnings

import PIL.Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from SPM_Filetype import SPM
import matplotlib
import QualityMeasurements
from ReadWriteSXM import arr_2_sxm
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import numpy as np
#from PIL import Image
#import copy
from YOLO.Preprocessing import preprocess_yolo_test
#import YOLO.Preprocessing as YoloPreprocess
from Preprocessing import png_folder
import argparse
from YOLO.utils.general import print_args
from YOLO.testSSDNA import main as testSSDNA_main
import Preprocessing
from SemanticSegmentation import evaluate_real, evaluate_real_2
from MyUNet.Pytorch_UNet.unet import UNet
from CompleteEval import extract_gsc_ss
from Evaluate_SS import evaluate_ss
import pandas as pd
import sys
import scipy.stats as stats
from AnalyzeError import measurementError, get_q_errors
#from ErrorsNNFit import Net as importErrorNet
#from matplotlib.colors import LinearSegmentedColormap
import struct
import wandb
SHOW_ALL = False
from ClassImageAnalysis import canny, hough, pretransform_img, npy2img, find_direction_range, scan_origami, find_maxima
import traceback
from QualityMeasurements import *
from ManageTrainData import hoshen_koppelmann
from ManipulateDNA.ModifySPM import denoise_set
#from visualize_pdf import create_pdf_report

def get_folder_files(fldr, extensions=None):
    files = []
    foldernames = []

    for x in os.listdir(fldr):
        f = os.path.join(fldr, x)
        if os.path.isdir(f):
            files2, foldernames2 = get_folder_files(f, extensions=extensions)
            for file in files2:
                if extensions is not None:
                    if file.split(".")[-1] in extensions:

                        files.append(file)
                else:
                    files.append(file)
            for fn in foldernames2:
                foldernames.append(os.path.join(fldr, fn))
        else:
            if extensions is not None:
                if x.split(".")[-1] in extensions:
                    files.append(x)
            else:
                files.append(x)
            foldernames.append(fldr)

    filtered_files = []
    filtererd_foldernames = []
    forbidden = "= .,:;"
    changes = False
    for i in range(len(files)):
        file = files[i]
        file = files[i]
        fn_prplcd = copy.deepcopy(foldernames[i])[3:]
        foldername = foldernames[i]
        for bst in forbidden:
            fn_prplcd = fn_prplcd.replace(bst, "_")

        fn_prplcd = foldername[:3] + fn_prplcd

        if fn_prplcd != foldername:
            if fn_prplcd in foldernames:
                continue
        filtered_files.append(file)
        filtererd_foldernames.append(foldername)

    return filtered_files, filtererd_foldernames


def rename_files(files, foldernames):
    """
    Replace all empty spaces and dots by _
    :param files:
    :param foldernames:
    :return:
    """

    fnsnew = []
    forbidden = "= .,:;"
    for i, fn in tqdm(enumerate(foldernames), desc="Renaming Folders", disable=True):
        fns = str(fn)
        # fns = fns.replace(" ", "_")
        # fns = fns.replace("=", "_")
        # fns = fns.replace(".", "_")
        for i in range(3, len(fns)):  # 3 to account for D:

            if fns[i] in forbidden:
                fnstmp = copy.deepcopy(fns)
                fns = fns[:i] + "_" + fns[i + 1:]
                # print("Changing ", fnstmp, " -> ", fns)
                try:
                    # if os.path.isfile(fns) or os.path.isfile(fns):
                    #     continue
                    os.rename(fnstmp, fns)
                except FileNotFoundError as e:
                    # print(e)
                    pass
                    # input()
        fnsnew.append(fns)

    fsnew = []
    for i, f in tqdm(enumerate(files), desc="Renaming Files"):
        fold = copy.deepcopy(f)
        f = os.path.join(fnsnew[i], f)
        f = str(f).replace(".", "_")
        f = str(f).replace(" ", "_")
        f = str(f).replace("=", "_")
        assert f[-4] == "_", f
        fnew = f[:-4] + "." + f[-3:]
        f = fnew
        # if not os.path.isfile(f):
        os.rename(os.path.join(fnsnew[i], fold), f)

        fsnew.append(f.split("\\")[-1])

    return fsnew, fnsnew

def read_input(file, resultf):
    os.makedirs(resultf, exist_ok=True)
    ext = file.split(".")[-1]
    if ext == "sxm":
        return read_sxm(file, resultf, dontflip=False)
    elif ext == "spm":
        return read_spm(file, resultf)
    else:
        raise Exception("Unknown raw file {}".format(file))

def read_sxm(file, resultf, dontflip=True):
    """
    Gets data from existing SXM file
    :param filename: file
    :param dontflip: flips the matrix as default to match image file
    :return:
    """
    assert os.path.exists(file)
    f = open(file, 'rb')
    l = ''
    key = ''
    header = {}
    while l != b':SCANIT_END:':
        l = f.readline().rstrip()
        if l[:1] == b':':
            key = l.split(b':')[1].decode('ascii')
            header[key] = []
        else:
            if l:  # remove empty lines
                header[key].append(l.decode('ascii').split())
    while f.read(1) != b'\x1a':
        pass
    assert f.read(1) == b'\x04'
    assert header['SCANIT_TYPE'][0][0] in ['FLOAT', 'INT', 'UINT', 'DOUBLE']
    data_offset = f.tell()
    size = dict(pixels={
        'x': int(header['SCAN_PIXELS'][0][0]),
        'y': int(header['SCAN_PIXELS'][0][1])
    }, real={
        'x': float(header['SCAN_RANGE'][0][0]),
        'y': float(header['SCAN_RANGE'][0][1]),
        'unit': 'm'
    })
    im_size = size['pixels']['x'] * size['pixels']['y']
    print(file)
    try:
        data = np.array(struct.unpack('<>'['MSBFIRST' == header['SCANIT_TYPE'][0][1]] + str(im_size) +
                                  {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[header['SCANIT_TYPE'][0][0]],
                                  f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))
    except Exception as e:
        print(e)
    if not dontflip:
        data = np.flipud(data)

    validmin = 1000
    validmax = 0

    for i in range(data.shape[0]):
        if np.isnan(data[i, 0]) or np.isnan(data[i, -1]):
            continue
        if i < validmin:
            validmin = i
        if i > validmax:
            validmax = i
    data = data[validmin:validmax+1, :]

    data = data.astype(float)
    maxi = np.amax(data)
    mini = np.amin(data)
    data = (data - mini) / (maxi - mini)
    # plt.imshow(data)
    # plt.title("Normalized")
    # plt.show()

    data *= 255
    data = data.astype(np.uint8)
    img = Image.fromarray(data, "L")
    img.save(os.path.join(resultf, f"Image.png"))

    # img.show()
    with open(os.path.join(resultf, os.path.join(resultf, f"Image.txt")), "w") as f:
        f.write(f"ImgPx_X: {data.shape[1]}\n")
        f.write(f"ImgPx_Y: {data.shape[0]}\n")
        f.write(f"spmnM_X: {size['real']['x'] * 1e9}\n")
        f.write(f"spmnM_Y: {size['real']['y'] * 1e9}\n")
        f.write(f"spmPx_X: {size['pixels']['x']}\n")
        f.write(f"spmPx_Y: {size['pixels']['y']}\n")
        f.write(f"spmnM/px_X: {size['real']['x']/size['pixels']['x']}\n")
        f.write(f"spmnM/px_Y: {size['real']['y']/size['pixels']['y']}\n")

    return data

def read_spm(file, resultf):
    """
    Read .spm file and save result as grayscale png
    :param file:
    :return:
    """

    # print("ReadSPM ", file, " -> ", resultf)

    chID = 0  # 0: Image, 1: DI/DV, 2: ?
    spm = SPM(file)
    dat = spm.get_data(chID=chID)
    if SHOW_ALL:
        plt.imshow(dat)
        plt.title("ChannelID: " + str(chID))
        plt.show()
        plt.close()
    dat = dat.astype(float)
    maxi = np.amax(dat)
    mini = np.amin(dat)
    dat = (dat - mini) / (maxi - mini)
    if SHOW_ALL:
        plt.imshow(dat)
        plt.title("Normalized")
        plt.show()

    dat *= 255
    dat = dat.astype(np.uint8)
    img = Image.fromarray(dat, "L")
    if SHOW_ALL:
        img.show()

    img.save(os.path.join(resultf, f"Image.png"))
    # print("Saved as ", os.path.join(resultf, f"Image{str(id).zfill(4)}.png"))

    img_px_x = dat.shape[1]
    img_px_y = dat.shape[0]
    spm_nm_x, spm_nm_y = spm.get_size_real()  # um to nm
    spm_px_x, spm_px_y = spm.get_size_pixel()
    spm_nmppx_x, spm_nmppx_y = spm.get_size_per_pixel()  # um to nm
    if spm_nm_x < 50:
        spm_nm_x *= 1000
        spm_nmppx_x *= 1000
    if spm_nm_y < 50:
        spm_nm_y *= 1000
        spm_nmppx_y *= 1000

    with open(os.path.join(resultf, os.path.join(resultf, f"Image.txt")), "w") as f:
        f.write(f"ImgPx_X: {img_px_x}\n")
        f.write(f"ImgPx_Y: {img_px_y}\n")
        f.write(f"spmnM_X: {spm_nm_x}\n")
        f.write(f"spmnM_Y: {spm_nm_y}\n")
        f.write(f"spmPx_X: {spm_px_x}\n")
        f.write(f"spmPx_Y: {spm_px_y}\n")
        f.write(f"spmnM/px_X: {spm_nmppx_x}\n")
        f.write(f"spmnM/px_Y: {spm_nmppx_y}\n")

def apply_sr(model, infld, outfld, scale):
    print("Apply SR", model, infld, outfld, scale)
    denoise_set(infld=infld, outfld=outfld, model=model, do_preprocess=False, fraction=scale, save_sxm=False)

def prepare_for_yolo(filefolders, tempfolder, resultfolder, sr_model, sr_folder, sr_scale, save_pp_of=None, sample_png=None):
    """
        Pretransform png files and turn into YOLO Readable format

    :param filefolders:
    :param tempfolder:
    :param resultfolder:
    :return:
    """
    tempnames = []
    os.makedirs(os.path.join(tempfolder, "images"), exist_ok=True)
    for i, fld in enumerate(filefolders):
        file = os.path.join(fld, "Image.png")
        newname = os.path.join(tempfolder, "images", f"Image{str(i).zfill(4)}.png")
        shutil.copy(file, newname)
        tempnames.append(newname)
    if os.path.isdir(os.path.join(resultfolder, "images")):
        os.rename(resultfolder, resultfolder + "_oldPP" + str(random.randint(0, 1000)))

    show = False
    enhance_contrast = True
    # YoloPreprocess.normalize_soft = YoloPreprocess.gamma_correction
    zfill = 6
    threads = 14
    img_size = None
    overwrite = False
    lines_corr = True
    do_flatten = False
    flip = False
    flatten_line_90 = False
    do_flatten_border = False
    dir_img = os.path.join(tempfolder, "images")
    dir_img_pret = os.path.join(resultfolder, "images")
    os.makedirs(dir_img_pret, exist_ok=True)
    rot90=False
    args = (dir_img, None, None, None, dir_img_pret,
            None, None, None, show,
            enhance_contrast,
            zfill, threads, img_size, overwrite, lines_corr,
            do_flatten, flip, flatten_line_90,
            do_flatten_border, False, False, True, rot90)

    preprocess_yolo_test(tempfolder, resultfolder, args)
    if not os.path.isdir(os.path.join(resultfolder, "images_npy")):
        os.rename(os.path.join(resultfolder, "images"), os.path.join(resultfolder, "images_npy"))
        png_folder(os.path.join(resultfolder, "images_npy"), os.path.join(resultfolder, "images"))

    print("Preparing for YOLO predictions...")

    if sr_model is not None:
        apply_sr(sr_model, os.path.join(resultfolder, "images"), sr_folder, sr_scale)
        shutil.copytree(os.path.join(sr_folder, "PNGs", "mixed"), os.path.join("datasets", "EvalChain", "images"),  dirs_exist_ok=True)
    else:
        shutil.copytree(os.path.join(resultfolder, "images"), os.path.join("datasets", "EvalChain", "images"),
                        dirs_exist_ok=True)
    with open(os.path.join("YOLO", "EvalChain.yaml"), "w") as f:
        f.write("path:   ..\datasets\RealData\ntrain: images\nval: images\ntest:  images\nnames:\n\t0: origami\n")



    tempdict = {}
    for i in range(len(filefolders)):
        tempdict[tempnames[i].split("\\")[-1]] = filefolders[i]

    if save_pp_of is not None:
        assert sample_png is not None
        sample_png = sample_png.split("\\")[-1]
        def get_subpath(path):
            # print("inpt: ", path)
            parts = path.split("\\")
            for i in range(len(parts)):
                if parts[i] == sample_png:
                    # print("Ret: ", "\\".join(parts[i+1:]))
                    parts[i] = save_pp_of.split("\\")[-1]
                    return "\\".join(parts[i+1:]), "\\".join(parts)

        for key in tempdict.keys():
            sbp, newpath = get_subpath(tempdict[key])
            rp = os.path.join(save_pp_of, sbp)
            os.makedirs(rp, exist_ok=True)
            shutil.copy(os.path.join(resultfolder, "images", key), os.path.join(rp, "Image.png"))
            shutil.copy(os.path.join(tempdict[key], "Image.txt"), os.path.join(rp, "Image.txt"))
            tempdict[key] = newpath



    # for key in tempdict.keys():
    #     print(f"{key} --> {tempdict[key]}")


    return os.path.join("YOLO", "EvalChain.yaml"), tempdict, os.path.join("datasets", "EvalChain", "images"), args


def clear_yolo_dataset(dataset_path):
    for img in tqdm(os.listdir(dataset_path), desc="Deleting YOLO Dataset"):
        os.remove(os.path.join(dataset_path, img))


def predict_yolo(model, yaml_file, conf_thrsh=0.7):
    """
    Put prep whole image into yolo network and get predictions
    :param model:
    :param yaml_file:
    :param conf_thrsh:
    :return:
    """
    proj = "EvalChain"
    name = proj
    weights = model
    data = yaml_file
    src = os.path.join("datasets", "EvalChain", "images")
    img_size = 256

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model path(s)')
    parser.add_argument('--source', type=str, default=src, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=data, help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[img_size],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=conf_thrsh, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=False, action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=proj, help='save results to project/name')
    parser.add_argument('--name', default=name, help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    testSSDNA_main(opt)


def read_imgTXT(file):
    img_wPX = 0
    img_hPX = 0
    img_wnM = 0
    img_hnM = 0
    img_nMpPX_x = 0
    img_nMpPX_y = 0
    with open(file, "r") as f:
        for line in f:
            if line.startswith("ImgPx_X"):
                img_wPX = int(line.split(":")[1].strip())
            elif line.startswith("ImgPx_Y"):
                img_hPX = int(line.split(":")[1].strip())
            elif line.startswith("spmnM_X"):
                img_wnM = float(line.split(":")[1].strip())
            elif line.startswith("spmnM_Y"):
                img_hnM = float(line.split(":")[1].strip())
            elif line.startswith("spmnM/px_X"):
                img_nMpPX_x = float(line.split(":")[1].strip())
            elif line.startswith("spmnM/px_Y"):
                img_nMpPX_y = float(line.split(":")[1].strip())
            else:
                pass
    return img_wPX, img_hPX, img_wnM, img_hnM, img_nMpPX_x, img_nMpPX_y


def crop_images(tempdict, yolo_pred, resultf, sample_png, tempfolder2, extend=2.0, square=True, drop_border=False,
                crop_labels=False, prov_labels_folder=None, label_crops=None, sxm_folder=None):
    """
    Crop image according to yolo predictions
    :param tempdict: dictionary for mapping current filenames to original ones
    :param sample_png_norm: Necessary if sample_png is pp version istead of normal to separ
    :return:
    """
    imgs = tempdict.keys()
    tempdict2 = {}
    temp_idx = 0
    conf_dict = {}
    crpsize_dict = {}
    for img in tqdm(imgs, desc="Cropping Images"):
        img_key = img
        crop_file = os.path.join(yolo_pred, "labels", img.split(".")[0] + ".txt")
        #print("CropFile: ", crop_file)

        if crop_labels:
            assert prov_labels_folder is not None
            temp_sample = os.path.join(prov_labels_folder, img)
            # print("Temp Sample: ", temp_sample)
            lbl_img = Image.open(temp_sample)
            # lbl_img.show()

        if not os.path.isfile(crop_file):
            # No detections in Image
            continue

        img_file = os.path.join(tempdict[img], "Image.png")
        dim_file = os.path.join(tempdict[img], "Image.txt")
        #print("imgFile: ", img_file)

        subdir = str(tempdict[img].split(".")[0])
        #print("subdir: ", subdir)

        parts = subdir.split("\\")
        td = None
        for i in range(len(parts)):
            temp = sample_png.split("\\")[-1]
            # print(f"Searching for {temp} in {parts}")
            if parts[i] == temp:
                td = "\\".join(parts[i + 1:])
                break
        img_resdir = os.path.join(resultf, td)
        #print("Resdir", img_resdir)
        os.makedirs(img_resdir, exist_ok=True)

        img_wPX, img_hPX, img_wnM, img_hnM, img_nMpPX_x, img_nMpPX_y = read_imgTXT(dim_file)

        img = Image.open(img_file)
        if False:  # ToDo Show ALL
            img.show()
        # print(img.size)
        assert img.size[0] == img_wPX, f"{img.size[0]} == {img_wPX}"
        assert img.size[1] == img_hPX, F"{img.size[1]} == {img_hPX}"
        # input()

        # Filestructure s. Zettel, crop images with variable boundary
        lims = []
        confs = []
        with open(crop_file, "r") as f:
            for line in f:
                parts = line.split(" ")
                lims.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
                confs.append(float(parts[5]))

        crp_idx = 0



        for i, crp in enumerate(lims):
            conf = confs[i]
            border_pct = 1.0
            x, y, w, h = crp
            w *= extend
            h *= extend
            if square:
                w_px = w * img_wPX
                h_px = h * img_hPX
                w_px = max(w_px, h_px)
                h_px = w_px
                w = w_px / img_wPX
                h = h_px / img_hPX

            # print("W, H: ", w, h)

            xmin = x - w / 2
            xmax = x + w / 2
            ymin = y - h / 2
            ymax = y + h / 2

            w_px = int(np.ceil(w * img_wPX))
            h_px = int(np.ceil(h * img_hPX))
            xmin_px = int(round(img_wPX * xmin))
            xmax_px = xmin_px + w_px
            ymin_px = int(round(img_hPX * ymin))
            ymax_px = ymin_px + h_px
            xmin_px_o = int(round(img_wPX * xmin))
            xmax_px_o = xmin_px + w_px
            ymin_px_o = int(round(img_hPX * ymin))
            ymax_px_o = ymin_px + h_px

            # print("W: ", xmax_px - xmin_px, "H: ", ymax_px - ymin_px)

            if 0 > xmin_px or 0 > ymin_px or img_wPX < xmax_px or img_hPX < ymax_px:
                # print("Border case: ", xmin_px, xmax_px, ymin_px, ymax_px)
                border_pct = ((min(img_wPX, xmax_px) - max(0, xmin_px)) * (min(img_hPX, ymax_px) - max(0, ymin_px))) / (
                        (xmax_px - xmin_px) * (ymax_px - ymin_px))
                # print("Border PCT: ", border_pct)
                if drop_border:
                    continue

                if xmin_px < 0:
                    xmax_px -= (xmin_px)
                    xmin_px = 0
                if xmax_px > img_wPX:
                    xmin_px -= (xmax_px - img_wPX)
                    xmax_px = img_wPX
                if ymin_px < 0:
                    ymax_px -= (ymin_px)
                    ymin_px = 0
                if ymax_px > img_hPX:
                    ymin_px -= (ymax_px - img_hPX)
                    ymax_px = img_hPX

            w_px = xmax_px - xmin_px
            h_px = ymax_px - ymin_px
            if not w_px == h_px:
                print("Dims not matching")
                continue

            crp_w_nM = img_wnM * (w_px / img_wPX)
            crp_h_nM = img_hnM * (h_px / img_hPX)

            if CROP_LABELS:
                labelfld = CROP_LABELFLD
                # print(img_file)
                name = os.path.basename(os.path.dirname(img_file))
                # print(name)
                num = int(name[6:])
                lblfile = os.path.join(labelfld, f"Label_ss{str(num).zfill(6)}.png")
                lbl = Image.open(lblfile)
                lbl = lbl.transpose(Image.FLIP_TOP_BOTTOM)

                # plt.switch_backend('TkAgg')
                # fid, axs = plt.subplots(1, 2)
                # axs[0].imshow(np.array(img))
                # axs[1].imshow(np.array(lbl))
                # axs[0].set_title(os.path.basename(img_file))
                # axs[1].set_title(os.path.basename(lblfile))
                # plt.show()
                # resf = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\SynthCrop\Image'
                # resl = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\SynthCrop\Temp'
#
                # idx = len(os.listdir(resl))
                # plt.savefig(os.path.join(resl, f"Image{str(idx).zfill(6)}.png"))




            crp_img = img.crop((xmin_px, ymin_px, xmax_px, ymax_px))

            if CROP_LABELS:
                crp_lbl = lbl.crop((xmin_px, ymin_px, xmax_px, ymax_px))


            assert crp_img.size[0] == crp_img.size[
                1], f"{crp_img.size}, {img.size}, {xmin_px_o}, {xmax_px_o}, {ymin_px_o}, {ymax_px_o}, --> {xmin_px}," \
                    f" {xmax_px}, {ymin_px}, {ymax_px}"
            if False:  # ToDo: SHOW_ALL
                crp_img.show()

            crop_fld = os.path.join(img_resdir, f"Crop{crp_idx}")
            # print("Crop folder ", crop_fld)
            crp_idx += 1
            os.makedirs(crop_fld, exist_ok=True)

            crp_img.save(os.path.join(crop_fld, "Image.png"))
            if CROP_LABELS:


                resf = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\Set161_EvalYOLO_63_label\bild'
                resl = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\Set161_EvalYOLO_63_label\data\PNG'

                idx = len(os.listdir(resf))
                crp_lbl.save(os.path.join(resl, f"Image{str(idx).zfill(6)}.png"))
                crp_img.save(os.path.join(resf, f"Image{str(idx).zfill(6)}.png"))





            #print("CropFld: ", crop_fld)
            with open(os.path.join(crop_fld, "Image.txt"), "w") as f:
                f.write(f"Pos_Px_X: {x * img_wPX}\n")
                f.write(f"Pos_Px_Y: {y * img_hPX}\n")
                f.write(f"Crop_Px_X: {w_px}\n")
                f.write(f"Crop_Px_Y: {h_px}\n")
                f.write(f"Crop_nM_X: {crp_w_nM}\n")
                f.write(f"Crop_nM_Y: {crp_h_nM}\n")
                f.write(f"Crop_nMpPX_X: {img_nMpPX_x}\n")
                f.write(f"Crop_nMpPX_Y: {img_nMpPX_y}\n")
                f.write(f"Confidence: {conf}\n")
                f.write(f"BorderPCT: {border_pct}\n")
            # print("Savef as ", os.path.join(crop_fld, "Image.txt"))

            tempname = f"Image{str(temp_idx).zfill(4)}.png"
            conf_dict[f"Image{str(temp_idx).zfill(4)}"] = conf
            crpsize_dict[f"Image{str(temp_idx).zfill(4)}"] = w_px
            temppath = os.path.join(tempfolder2, tempname)
            tempdict2[tempname] = crop_fld
            shutil.copy(os.path.join(crop_fld, "Image.png"), temppath)


            if crop_labels:
                crp_lbl = lbl_img.crop((xmin_px, ymin_px, xmax_px, ymax_px))
                crp_lbl.save(os.path.join(label_crops, tempname))

            #print("Temppath: ", temppath)
            temp_idx += 1

    return tempdict2, conf_dict, crpsize_dict


def preprocess_origami(tempfolder, resf_npy, resf_png, threads, img_size=128, skip_all=None):
    """
    Prepare Image for SS Prediction: Rescale to 128 and normalize etc
    :return:
    """
    # Preprocessing.normalize_soft = Preprocessing.normalize_half_sigmoid
    dir_img = tempfolder
    dir_img_pret = resf_npy
    dir_mask = None
    dir_img_test = None
    dir_mask_test = None
    dir_mask_pret = None
    dir_img_test_pret = None
    dir_mask_test_pret = None
    show = False
    enhance_contrast = True
    zfill = 4
    threads = threads
    img_size = img_size
    overwrite = False
    lines_corr = False
    do_flatten = False
    flip = False
    flatten_line_90 = False
    do_flatten_border = True
    use_masks = False
    use_Test = False
    keep_name = True
    skip_all = False if skip_all is None else skip_all
    resize_method = "bilinear" # bilinear
    Preprocessing.pretransform_all(dir_img=dir_img, dir_mask=dir_mask, dir_img_test=dir_img_test,
                                   dir_mask_test=dir_mask_test, dir_img_pret=dir_img_pret, dir_mask_pret=dir_mask_pret,
                                   dir_img_test_pret=dir_img_test_pret, dir_mask_test_pret=dir_mask_test_pret,
                                   show=show, enhance_contrast=enhance_contrast, zfill=zfill,
                                   threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                                   do_flatten=do_flatten,
                                   do_flatten_border=do_flatten_border, flip=flip, flatten_line_90=flatten_line_90,
                                   keep_name=keep_name,
                                   use_Test=use_Test, use_masks=use_masks, resize_method=resize_method, skip_all=skip_all)
    Preprocessing.png_folder(resf_npy, resf_png)
    pp_origami_args = f"dir_img={dir_img}, dir_mask={dir_mask}, dir_img_test={dir_img_test}," \
                      f" dir_mask_test={dir_mask_test}, dir_img_pret={dir_img_pret}, dir_mask_pret={dir_mask_pret}," \
                      f" dir_img_test_pret={dir_img_test_pret}, dir_mask_test_pret={dir_mask_test_pret}, show={show}," \
                      f" enhance_contrast={enhance_contrast}, zfill={zfill},threads={threads}, img_size={img_size}," \
                      f" overwrite={overwrite}, lines_corr={lines_corr}, do_flatten={do_flatten}," \
                      f" do_flatten_border={do_flatten_border}, flip={flip}, flatten_line_90={flatten_line_90}," \
                      f" keep_name={keep_name}, use_Test={use_Test}, use_masks={use_masks}," \
                      f" resize_method={resize_method}"
    return pp_origami_args


def get_semantic_seg(model, folder, resf, threshold=0.3, bilinear=False, oldModel=None):
    """
    Perform Semantic Segementation of markers
    :param model: model for prediction
    :param resf: result
    :return:
    """
    folder_non_pt = folder  # Same folders to trick SS.eval_real to do not Preproc
    folder_pt = folder
    model_path = model
    n_classes = 3
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    net = UNet(n_channels=1, n_classes=n_classes, bilinear=bilinear).to(device=DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = net.to(device=DEVICE)

    evaluate_real(model=model, device=None, folder=folder_non_pt, folder_pt=folder_pt,
                  folder_results=resf, label_folder=None, save_pret=None, line_corr=None,
                  do_flatten=None, do_flatten_border=None,
                  enhance_contrast=None, flip=None, threshold=threshold, flatten_line_90=None,
                  workers=0, old_model=OLD_MODEL if oldModel is None else oldModel)


def get_distances(images, gsc_labels, resf, fit_params, threads=6, export_hists=False, pp_img_folder=None,
                  eval_ss_SHOWALL=False):
    """
    calculate distance form image with provided image and ss labeling
    :param resf: Result Folder
    :return:
    """
    image_folder = images
    label_folder = gsc_labels
    save_folder = resf
    accuracy = None
    transpose = False
    distances, thetas = evaluate_ss(image_folder, label_folder, save_folder, threads, accuracy, show_all=eval_ss_SHOWALL,
                                    verbose=False, transpose=transpose, fit_params_folder=fit_params,
                                    export_hists=export_hists, pp_img_folder=pp_img_folder)
    return distances, thetas


def make_csv(images, distances, thetas=None, resfp=None):
    if thetas is not None:
        assert len(distances) == len(
            thetas), f"Distances and thetas must be same length but have different lengths {len(distances)} and " \
                     f"{len(thetas)}"
        res = {}
        for i, elem in enumerate(os.listdir(images)):
            res[elem.split(".")[0]] = [distances[i], thetas[i]]

        df = pd.DataFrame.from_dict(res)
        df = df.transpose()
        df.to_csv(resfp)

    else:
        res = {}
        for i, elem in enumerate(os.listdir(images)):
            res[elem.split(".")[0]] = [distances[i], 0]

        df = pd.DataFrame.from_dict(res)
        df = df.transpose()
        df.to_csv(resfp)


def rescale_orig_imgs(imgs, resf, img_size=128):
    interpol = Image.BICUBIC
    for file in tqdm(os.listdir(imgs), desc="Rescaling Images"):
        img = Image.open(os.path.join(imgs, file))
        if img_size is not None:
            img = img.resize((img_size, img_size), interpol)
        img.save(os.path.join(resf, file))


def read_crpTXT(file):
    posX, posY, crp_x_px, crp_y_px, crp_x_nm, crp_y_nm, crp_x_nmPpx, crp_y_nmPpx = 0, 0, 0, 0, 0, 0, 0, 0
    dist_px = 0
    dist_nm = 0
    success = False
    with open(file, "r") as f:
        for line in f:
            if line.startswith("Pos_Px_X"):
                posX = float(line.split(":")[1].strip())
            elif line.startswith("Pos_Px_Y"):
                posY = float(line.split(":")[1].strip())
            elif line.startswith("Crop_Px_X"):
                crp_x_px = float(line.split(":")[1].strip())
            elif line.startswith("Crop_Px_Y"):
                crp_y_px = float(line.split(":")[1].strip())
            elif line.startswith("Crop_nM_X"):
                crp_x_nm = float(line.split(":")[1].strip())
            elif line.startswith("Crop_nM_Y"):
                crp_y_nm = float(line.split(":")[1].strip())
            elif line.startswith("Crop_nMpPX_X"):
                crp_x_nmPpx = float(line.split(":")[1].strip())
            elif line.startswith("Crop_nMpPX_Y"):
                crp_y_nmPpx = float(line.split(":")[1].strip())
            elif line.startswith("Dist_px"):
                dist_px = float(line.split(":")[1].strip())
            elif line.startswith("Dist_nm"):
                dist_nm = float(line.split(":")[1].strip())
            elif line.startswith("Success"):
                success = line.split(":")[1].strip() == "True"
            else:
                pass

    return posX, posY, crp_x_px, crp_y_px, crp_x_nm, crp_y_nm, crp_x_nmPpx, crp_y_nmPpx, dist_px, dist_nm, success


def reorganize_files(images, dist_images, dist_csv, tempdict, resfile, img_size=128, filter_cropsize=True,
                     filter_thrsh=3, saveplot=False):
    start = True
    imgs = [os.path.join(images, x) for x in os.listdir(images)]
    dist_imgs = [os.path.join(dist_images, x) for x in os.listdir(dist_images)]
    dist_imgs_bare = [x for x in os.listdir(dist_images)]
    resf = open(resfile, "w")
    resf.write(f"Tempname;Foldername;Distance\n")

    # print(tempdict.keys())
    # for key in tempdict.keys():
    #     print(f"{key}: {tempdict[key]}")

    ret_dict = {}

    def get_cropsizes(crpfolder, saveas=None):
        # Histogram all cropsizes except provided one
        crpfolder_base = os.path.dirname(crpfolder)
        if len(os.listdir(crpfolder_base)) < 7:
            return 0, np.inf, np.inf
        thisone = os.path.basename(crpfolder)

        crpsize = []
        for fld in os.listdir(crpfolder_base):
            dim_file = os.path.join(crpfolder_base, fld, "Image.txt")
            _, _, crp_x_px, _, _, _, _, _, _, _, _ = read_crpTXT(dim_file)

            if fld == thisone:
                thissize = crp_x_px
            else:
                crpsize.append(crp_x_px)

        avg = np.average(crpsize)
        std = np.std(crpsize)
        dst = (abs(thissize - avg) / std)

        if saveas is not None:
            matplotlib.use("Agg")
            fig, axs = plt.subplots(1, 1)
            ax = axs
            ax.hist(crpsize, bins=max(10, int(np.sqrt(len(crpsize)))))
            ax.autoscale(False)
            ax.scatter([thissize], [1], c='red', zorder=1)
            plt.title(f"Avg: {avg:.2f}, Std: {std:.2f}, dist: {dst:.2f} xSTD")
            plt.savefig(saveas)
            plt.close('all')

        # if dst > 3:
        #     matplotlib.use("TkAgg")
        #     fig, axs = plt.subplots(1, 1)
        #     ax = axs
        #     ax.hist(crpsize, bins=max(10, int(np.sqrt(len(crpsize)))))
        #     ax.autoscale(False)
        #     ax.scatter([thissize], [1], c='red', zorder=1)
        #     plt.title(f"Avg: {avg:.2f}, Std: {std:.2f}, dist: {dst:.2f} xSTD")
        #     plt.show()
        return avg, std, dst

    tls = 0
    with open(dist_csv, "r") as f:
        for line in f:
            tls += 1

    with open(dist_csv, "r") as f:
        for i, line in tqdm(enumerate(f), desc="Reorganizing files", total=tls):
            if start:
                start = False
                continue

            img, dist, theta = line.split(",")
            img += ".png"
            dist = float(dist)
            theta = float(theta)

            if img in dist_imgs_bare:
                d_image = os.path.join(dist_images, img)
            else:
                d_image = None
            # print("Line: ", line)
            # print("Images: ", imgs[idx])
            # print("DistImgs: ", d_image)
            # print("dist: ", dist)
            # print("Temp: ", tempdict[img])

            crp_folder = tempdict[img]
            if d_image is not None:
                shutil.copy(d_image, os.path.join(crp_folder))
            dim_file = os.path.join(crp_folder, "Image.txt")
            _, _, crp_x_px, crp_y_px, crp_x_nm, crp_y_nm, _, _, _, _, _ = read_crpTXT(dim_file)

            # assert crp_x_px == crp_y_px, f"{crp_x_px} != {crp_y_px}"
            # assert abs(crp_x_nm - crp_y_nm) / crp_x_nm < 1e-3, f"{crp_x_nm} != {crp_y_nm}"

            avg, std, dst = get_cropsizes(crp_folder,
                                          saveas=os.path.join(crp_folder, "Cropsize_Hist.png") if saveplot else None)

            if not crp_x_px == crp_y_px:
                successful = False
                dist_px_orig = -1
                dist_nm_orig = -1
                print("Image not square: ", dim_file)
            elif filter_cropsize and dst > filter_thrsh:
                successful = False
                dist_px_orig = -1
                dist_nm_orig = -1
                print("Cropsize not OK: ", dim_file)
            else:
                dx = dist * abs(np.sin(theta))
                dy = dist * abs(np.cos(theta))
                if img_size is None:
                    d_px_x_orig = dx
                    d_px_y_orig = dy
                else:
                    d_px_x_orig = dx * crp_x_px / img_size
                    d_px_y_orig = dy * crp_y_px / img_size
                d_nm_x_orig = crp_x_nm * d_px_x_orig / crp_x_px
                d_nm_y_orig = crp_y_nm * d_px_y_orig / crp_y_px
                dist_nm_orig = np.sqrt(d_nm_x_orig ** 2 + d_nm_y_orig ** 2)
                dist_px_orig = 0
                # dist_px_orig = dist * crp_y_px / img_size
                # dist_nm_orig = crp_x_nm * dist_px_orig / crp_y_px
                successful = d_image is not None

            with open(dim_file, "a") as f:
                f.write(f"Dist_px: {dist_px_orig}\n")
                f.write(f"Dist_nm: {dist_nm_orig}\n")
                f.write(f"Success: {successful}\n")
                f.write(f"Theta: {theta}\n")
                f.write(f"Theta: {180 * theta / np.pi} deg\n")

            if SAVE_SXM_CROPS:
                fld = os.path.dirname(dim_file)
                print("Fld: ", fld)
                arr = np.array(Image.open(os.path.join(fld, "Image.png")))
                plt.imshow(arr)
                plt.title("Crop")
                plt.show()

                names = [elem.split(".")[0] for elem in os.listdir(fld)]
                if len(names) == 3:
                    name = sorted(names, key=lambda x: len(x))[-1]
                    print(name, names)
                    num = int(name[5:])
                    print("Save as ", os.path.join(fld, f"Molecule{str(num).zfill(4)}.sxm"))
                    arr_2_sxm(arr, np.array([crp_x_nm*1e-9, crp_y_nm*1e-9]), os.path.join(fld, f"Molecule{str(num).zfill(4)}.sxm"))

            ret_dict[img.split(".")[0]] = (dist_nm_orig, theta)
            resf.write(f"{img.split('.')[0]};{crp_folder};{dist_nm_orig}\n")

    # keys = ret_dict.keys()
    # keys = sorted(keys)
    #
    # for key in keys:
    #     print(f"{key}: {ret_dict[key]}nm")
    resf.close()
    return ret_dict


def reorganize_files_class(images, resfs, dist_csv, tempdict, crop_origami, res_origamis, img_size=128):
    start = True
    imgs = [os.path.join(images, x) for x in os.listdir(images)]
    res_flds = [os.path.join(resfs, x) for x in os.listdir(resfs)]
    res_flds_bare = [x for x in os.listdir(resfs)]

    # print(tempdict.keys())
    # for key in tempdict.keys():
    #     print(f"{key}: {tempdict[key]}")

    with open(dist_csv, "r") as f:
        for i, line in tqdm(enumerate(f), desc="Reorganizing files"):
            if start:
                start = False
                continue

            img, dist = line.split(",")
            dist = float(dist)

            if img in res_flds_bare:
                d_image = os.path.join(resfs, img)
            else:
                d_image = None

            imgtxt_file = os.path.join(tempdict[img + ".png"], "Image.txt")

            if d_image is not None:
                shutil.copytree(d_image, os.path.join(tempdict[img + ".png"], "Results"))

            dim_file = imgtxt_file
            _, _, crp_x_px, crp_y_px, crp_x_nm, crp_y_nm, _, _, _, _, _ = read_crpTXT(dim_file)

            # assert crp_x_px == crp_y_px, f"{crp_x_px} != {crp_y_px}"
            # assert abs(crp_x_nm - crp_y_nm) / crp_x_nm < 1e-3, f"{crp_x_nm} != {crp_y_nm}"
            if not crp_x_px == crp_y_px:
                successful = False,
                dist_px_orig = -1
                dist_nm_orig = -1
                print("Image not square: ", dim_file)
            else:
                dist_nm_orig = dist * (crp_x_nm / crp_x_px)
                successful = d_image is not None

            with open(dim_file, "a") as f:
                f.write(f"Dist_px: {dist}\n")
                f.write(f"Dist_nm: {dist_nm_orig}\n")
                f.write(f"Success: {successful}\n")





def export_distances(crop_origami, resf, crp_fld_prefix="Crop", filter_dist=(30, 150)):
    found_folders = []
    search_folders = []
    for fld in os.listdir(crop_origami):
        search_folders.append(fld)
    while len(search_folders) > 0:
        sf = search_folders.pop(0)
        for x in os.listdir(os.path.join(crop_origami, sf)):
            if x.startswith(crp_fld_prefix):
                found_folders.append(os.path.join(sf))
                break
            if os.path.isdir(os.path.join(crop_origami, sf, x)):
                search_folders.append(os.path.join(sf, x))

    subdirs = []
    subdir_distances = []
    subdir_distances_complete = []  # With not working ones
    total_distances = []
    total_distances_complete = []
    for ff in found_folders:
        dirs = ff.split("\\")
        subdirs.append(dirs)
    longest_chain = len(max(subdirs, key=lambda x: len(x)))

    for sd in subdirs:
        dsts = []
        dsts_complete = []
        fld = os.path.join(crop_origami, "\\".join(sd))
        for crop_fld in os.listdir(fld):
            if not os.path.isdir(os.path.join(fld, crop_fld)):
                continue
            assert crop_fld.startswith("Crop"), f"Invalid crop folder {crop_fld}"
            _, _, _, _, _, _, _, _, _, dist_nm, success = read_crpTXT(os.path.join(fld, crop_fld, "Image.txt"))

            if filter_dist is not None and not filter_dist[0] < dist_nm < filter_dist[1] and dist_nm > 0:
                success = False
            else:
                success = dist_nm > 0

            dsts_complete.append(dist_nm)
            total_distances_complete.append(dist_nm)
            if success:
                dsts.append(dist_nm)
                total_distances.append(dist_nm)
        subdir_distances.append(dsts)
        subdir_distances_complete.append(dsts_complete)

    categories = []
    category_dists = []
    category_dists_complete = []
    tmp_fkt = lambda x, i: "\\".join(x[:i + 1]) if len(x) >= i else None

    for i in range(longest_chain):
        cats = [tmp_fkt(x, i) for x in subdirs]
        cats = np.unique(cats)

        for c in cats:
            c_dists = []
            c_dists_complete = []

            for j, subdir in enumerate(subdirs):
                if len(subdir) >= i and "\\".join(subdir[:i + 1]) == c:
                    [c_dists.append(x) for x in subdir_distances[j]]
                    [c_dists_complete.append(x) for x in subdir_distances_complete[j]]
            categories.append(c)
            category_dists.append(c_dists)
            category_dists_complete.append(c_dists_complete)

    categories.insert(0, "total")
    category_dists.insert(0, total_distances)
    category_dists_complete.insert(0, total_distances_complete)
    category_dists_10pct = [[] for x in category_dists]
    averages = []
    medians = []
    stabws = []
    ok = []
    ims = []

    averages10p = []
    medians10p = []
    stabws10p = []
    ok10p = []
    ims10p = []

    for i in range(len(categories)):
        med = np.median(category_dists[i])
        ims.append(len(category_dists[i]))
        if len(category_dists[i]) > 0:
            averages.append(np.average(category_dists[i]))
            medians.append(med)
            stabws.append(np.std(category_dists[i]))
            ok.append(len(category_dists[i]) / len(category_dists_complete[i]))
        else:
            averages.append(0)
            medians.append(0)
            stabws.append(0)
            ok.append(0)

    med = medians[0]
    for i in range(len(categories)):
        for elem in category_dists[i]:
            if abs(elem - med) / med < 0.1:
                category_dists_10pct[i].append(elem)
        ims10p.append(len(category_dists_10pct[i]))
        if len(category_dists_10pct[i]) > 0:
            averages10p.append(np.average(category_dists_10pct[i]))
            medians10p.append(np.median(category_dists_10pct[i]))
            stabws10p.append(np.std(category_dists_10pct[i]))
            ok10p.append(len(category_dists_10pct[i]) / len(category_dists_complete[i]))
        else:
            averages10p.append(0)
            medians10p.append(0)
            stabws10p.append(0)
            ok10p.append(0)

    longest = len(category_dists[0])
    indices_n = ["Average", "Median", "Stabw", "OK", "Crops", "", ""]
    for i in range(longest):
        indices_n.append(i)

    dict_norm = {}
    for i, cat in enumerate(categories):
        lst = category_dists[i]
        while len(lst) <= longest:
            lst.append(np.nan)
        lst.insert(0, averages[i])
        lst.insert(1, medians[i])
        lst.insert(2, stabws[i])
        lst.insert(3, ok[i])
        lst.insert(4, ims[i])
        lst.insert(5, "")
        lst.insert(6, "")

        dict_norm[cat] = lst

    longest_complete = len(category_dists_complete[0])
    indices_c = ["Average", "Median", "Stabw", "OK", "Crops", "", ""]
    for i in range(longest_complete):
        indices_c.append(i)

    dict_compl = {}
    for i, cat in enumerate(categories):
        lst = category_dists_complete[i]
        while len(lst) <= longest_complete:
            lst.append(np.nan)
        lst.insert(0, averages[i])
        lst.insert(1, medians[i])
        lst.insert(2, stabws[i])
        lst.insert(3, ok[i])
        lst.insert(4, ims[i])
        lst.insert(5, "")
        lst.insert(6, "")

        dict_compl[cat] = lst

    longest_10p = len(category_dists_10pct[0])
    indices_10p = ["Average", "Median", "Stabw", "OK", "Crops", "", ""]
    for i in range(longest_10p):
        indices_10p.append(i)

    dict_10p = {}
    for i, cat in enumerate(categories):
        lst = category_dists_10pct[i]
        while len(lst) <= longest_10p:
            lst.append(np.nan)
        lst.insert(0, averages10p[i])
        lst.insert(1, medians10p[i])
        lst.insert(2, stabws10p[i])
        lst.insert(3, ok10p[i])
        lst.insert(4, ims10p[i])
        lst.insert(5, "")
        lst.insert(6, "")

        dict_10p[cat] = lst

    # norm_cats = dict_norm.keys()
    # for k in norm_cats:
    #     print(f"A-{k}: {len(dict_norm[k])} - {dict_norm[k]}")
    #
    # norm_cats = dict_compl.keys()
    # for k in norm_cats:
    #     print(f"B-{k}: {len(dict_compl[k])} - {dict_compl[k]}")
    #
    # norm_cats = dict_10p.keys()
    # for k in norm_cats:
    #     print(f"C-{k}: {len(dict_10p[k])} - {dict_10p[k]}")

    df_norm = pd.DataFrame(dict_norm)
    df_cmpl = pd.DataFrame(dict_compl)
    df_10pc = pd.DataFrame(dict_10p)

    norm_fp = os.path.join(resf, "Results_norm.csv")
    cmpl_fp = os.path.join(resf, "Results_complete.csv")
    fp_10pc = os.path.join(resf, "Results_10pct.csv")

    df_norm.to_csv(norm_fp, sep=";")
    df_cmpl.to_csv(cmpl_fp, sep=";")
    df_10pc.to_csv(fp_10pc, sep=';')

    return averages[0], medians[0], stabws[0], ok[0]


def modify_csv(filename, delim=";"):
    txt = ""
    with open(filename, "r") as f:
        for i, line in tqdm(enumerate(f), desc="Modifying CSV files"):
            if i == 1:
                prts = line.split(delim)
                if prts[0] != "0":
                    print("Processing already done")
                    return
                prts[0] = "Average"
                txt += delim.join(prts)
            elif i == 2:
                prts = line.split(delim)
                assert prts[0] == "1", line
                prts[0] = "Median"
                txt += delim.join(prts)
            elif i == 3:
                prts = line.split(delim)
                assert prts[0] == "2", line
                prts[0] = "Stabw"
                txt += delim.join(prts)
            elif i == 4:
                prts = line.split(delim)
                assert prts[0] == "3", line
                prts[0] = "Okay?"
                txt += delim.join(prts)
            elif i == 5:
                prts = line.split(delim)
                assert prts[0] == "4", line
                prts[0] = "Crops"
                txt += delim.join(prts)
            else:
                txt += line
    with open(filename, "w") as f:
        f.write(txt)


def visualize_results(resf, norm_fp, cmpl_fp, p10_fp, no_imgs, print_only_norm=True):
    files = {}
    files["norm"] = os.path.join(resf, norm_fp)
    files["cmpl"] = os.path.join(resf, cmpl_fp)
    files["pc10"] = os.path.join(resf, p10_fp)

    for typ in tqdm(["norm", "cmpl", "pc10"], desc="Visualizing Results"):
        df = pd.read_csv(os.path.join(resf, files[typ]), sep=";")

        cats = df.keys()[1:]
        for cat in cats:
            fld = os.path.join(resf, cat)
            os.makedirs(fld, exist_ok=True)
            vals = df[cat]
            avg = float(vals[0]) if vals[0] != "NaN" else 0
            med = float(vals[1]) if vals[1] != "NaN" else 0
            std = float(vals[2]) if vals[2] != "NaN" else 0
            oky = float(vals[3]) if vals[3] != "NaN" else 0
            ims = int(vals[4]) if vals[4] != "NaN" else 0
            k_factor = stats.t.ppf((1 + 0.9544997096061706) / 2, df=ims)


            indiv = [float(x) for x in vals[7:] if not pd.isna(x)]
            stabw, wahrsch_f, mittl_f = measurementError(indiv)
            median_unc = stabw * np.sqrt(np.pi * (2 * ims + 1) / (4 * ims)) if ims > 0 else 0
            if not print_only_norm or (print_only_norm and typ == "norm"):
                with open(os.path.join(fld, "Result.txt"), "a") as f:
                    f.write(f"{typ} - Average: {avg:.3f}nm\n")
                    f.write(f"{typ} - Median: {med:.3f}nm\n")
                    f.write(f"{typ} - Stabw: {std:.3f}nm\n")
                    f.write(f"{typ} - AvgUnc: {stabw:.3f}nm\n")
                    f.write(f"{typ} - MedianUnc: {median_unc:.3f}nm\n")
                    f.write(f"{typ} - k-factor: {k_factor:}\n")
                    f.write(f"{typ} - U_avg(k=2): {k_factor * stabw:.4f}nm\n")
                    f.write(f"{typ} - U_med(k=2): {k_factor * median_unc:.4f}nm\n")
                    f.write(f"{typ} - EvalOK: {100 * oky:.2f}%\n")
                    f.write(f"{typ} - Crops: {ims:.2f}\n")
                    f.write(f"{typ} - StabwMWT: {stabw:.2f}nm\n")
                    f.write(f"{typ} - Wahrsch.F.MWT: {wahrsch_f:.2f}nm\n")
                    f.write(f"{typ} - Mittl.F.MWT: {mittl_f:.2f}nm\n")

                    if typ == "norm":
                        sopt, nopt = get_q_errors(meas=indiv)
                        f.write(f"Estimated Achievable with Quality: StabwMWT: {sopt:.3f}nm\n")
                        f.write(f"Estimated Achievable with Quality: Images: {nopt}\n")

            # fig, ax = plt.subplots()
            plt.hist(indiv, bins=max(10, int(np.sqrt(len(indiv)))))
            plt.xlabel("Size")
            plt.ylabel("Number of Images")
            # if std != 0:
                # ax2 = ax.twinx()

            try:
                xs = np.linspace(min(indiv), max(indiv), 100)
                num_imgs = len(indiv)
                f = lambda x: num_imgs * np.exp(-((x - avg) ** 2) / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)
                ys = np.vectorize(f)(xs)
                plt.plot(xs, ys, color='red')
            except ValueError:
                print(f"Sequence empty for {typ}")
            except ZeroDivisionError:
                print("Std is zero")
            # ax2.plot(xs, ys, color='red')
            # ax2.set_ylabel("PDF", color='red')
            plt.title("Avg: {:.1f}nm, Med: {:.1f}nm, u: {:.1f}nm, Imgs: {}".format(avg, med, std/np.sqrt(len(indiv)), len(indiv)))
            plt.savefig(os.path.join(fld, f"plot_{typ}.png"), bbox_inches='tight')
            plt.close()


def export_quality(dist_theta_dict, rect_dict, gauss_dict, size_dict, count_dict, conf_dict, cpsize_dict,
                   marker_align_dict, export_folder,
                   q_indcs_file, method="Class", modelfolder=None,
                   fit_qfile="D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\"
                             "Try223_DS_StdQ_mitAngle\\quality\\quality_indcs.csv",
                   fit_stdf="D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try227_StdPlot\\results.csv"):
    files = dist_theta_dict.keys()
    distances = []
    thetas = []

    if QualityMeasurements.MODE == "STD":
        print("Using STD")
        error_fkt = QualityMeasurements.est_std
    elif QualityMeasurements.MODE == "AVG":
        print("Using AVERAGE")
        error_fkt = QualityMeasurements.est_error
    elif QualityMeasurements.MODE == "INTER":
        print("Using INTERPOLATION")
        error_fkt = QualityMeasurements.get_interpolator(
            file=fit_stdf,
            q_indcs=fit_qfile)
    elif QualityMeasurements.MODE == "FIT":
        print("Using FIT")
        error_fkt = QualityMeasurements.get_fit(file=fit_stdf,q_indcs=fit_qfile)
    else:
        raise Exception(f"Unknown Mode {QualityMeasurements.MODE}")

    for file in files:
        distances.append(dist_theta_dict[file][0])
        thetas.append(dist_theta_dict[file][1])
    tar_dist = np.average(distances)

    if method == "AI":
        assert modelfolder is not None
        est_error_ai = est_error_fkt(modelfolder)

    # Plost of Abw - Index for each type

    rect_pts = []
    gauss_pts = []
    size_pts = []
    count_pts = []
    conf_pts = []
    cpsz_pts = []
    absw = []
    est_errors = []
    marker_align = []

    for f in files:
        abw = abs(dist_theta_dict[f][0] - tar_dist) / tar_dist
        rect = rect_dict[f] if f in rect_dict.keys() else -1
        gauss = gauss_dict[f] if f in gauss_dict.keys() else -1
        size = size_dict[f] if f in size_dict.keys() else -1
        count = count_dict[f] if f in count_dict.keys() else -1
        conf = conf_dict[f] if f in conf_dict.keys() else -1
        cpsz = cpsize_dict[f] if f in cpsize_dict.keys() else -1
        mal = marker_align_dict[f] if f in marker_align_dict.keys() else -1

        if not 0 < abw < 2:
            abw = max(0, min(abw, 2))
        if not 0 < rect < 1:
            rect = max(0, min(rect, 1))
        if not 0 < gauss < 10:
            gauss = max(0, min(gauss, 10))
        if not 0 < size < 2:
            size = max(0, min(size, 2))
        if not 0 < count < 10:
            count = max(0, min(count, 10))
        if not 0 < conf < 1:
            conf = max(0, min(conf, 1))
        if not 0 < cpsz < 256:
            cpsz = max(0, min(cpsize_dict, 256))
        if mal > np.pi / 2:
            mal = np.pi - mal

        absw.append(abw)
        rect_pts.append(rect)
        size_pts.append(size)
        count_pts.append(count)
        gauss_pts.append(gauss)
        conf_pts.append(conf)
        cpsz_pts.append(cpsz)
        marker_align.append(mal)

        if dist_theta_dict[f][0] < 0:
            est_errors.append(np.infty)
        else:
            est_errors.append(error_fkt(rect, gauss, size, count, conf, mal))

    x = []
    y = []
    for i in range(len(absw)):
        if rect_pts[i] > 0:
            y.append(absw[i])
            x.append(rect_pts[i])
    plt.scatter(x, y)
    plt.title("Rectangularity")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("Rectanularity (in %)")
    plt.savefig(os.path.join(export_folder, "rectangularity.png"))
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y = []
    for i in range(len(absw)):
        if gauss_pts[i] > 0:
            y.append(absw[i])
            x.append(gauss_pts[i])
    plt.scatter(x, y)
    plt.title("Gauss Accuracy")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("Gauss |Stabw|")
    plt.savefig(os.path.join(export_folder, "gauss.png"))
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y = []
    for i in range(len(absw)):
        if size_pts[i] > 0:
            y.append(absw[i])
            x.append(size_pts[i])
    plt.scatter(x, y)
    plt.title("Marker-Size")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("Rel. Diff. in Markersize (in %)")
    plt.savefig(os.path.join(export_folder, "markersizes.png"))
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y = []
    for i in range(len(absw)):
        if count_pts[i] > 0:
            y.append(absw[i])
            x.append(count_pts[i])
    plt.scatter(x, y)
    plt.title("Number of markers")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("No. of markers")
    plt.savefig(os.path.join(export_folder, "markercounts.png"))
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y = []
    for i in range(len(absw)):
        if est_errors[i] > 0:
            y.append(absw[i])
            x.append(est_errors[i])
    plt.scatter(x, y)
    plt.title("Est. error")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("Est. Error (in %)")
    plt.savefig(os.path.join(export_folder, "est_error.png"))
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y = []
    for i in range(len(absw)):
        if conf_pts[i] > 0:
            y.append(absw[i])
            x.append(conf_pts[i])
    plt.scatter(x, y)
    plt.title("Confidence")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("confidence (in %)")
    plt.savefig(os.path.join(export_folder, "confidence.png"))
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y = []
    for i in range(len(absw)):
        if cpsz_pts[i] > 0:
            y.append(absw[i])
            x.append(cpsz_pts[i])
    plt.scatter(x, y)
    plt.title("Cropsize")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("Cropsize (in px)")
    plt.savefig(os.path.join(export_folder, "cropsize.png"))
    plt.clf()
    plt.cla()
    plt.close()
    x = []
    y = []
    for i in range(len(absw)):
        if marker_align[i] > 0:
            y.append(absw[i])
            x.append(marker_align[i])
    plt.scatter(x, y)
    plt.title("Marker Alignment")
    plt.ylabel('Distance error (in %)')
    plt.xlabel("marker Angle in bog")
    plt.savefig(os.path.join(export_folder, "markeralign.png"))
    plt.clf()
    plt.cla()
    plt.close()
    with open(os.path.join(export_folder, "rectangularity.csv"), "w") as f:
        for i in range(len(absw)):
            if rect_pts[i] > 0:
                f.write(f"{absw[i]};{rect_pts[i]}\n")
    with open(os.path.join(export_folder, "gauss.csv"), "w") as f:
        for i in range(len(absw)):
            if gauss_pts[i] > 0:
                f.write(f"{absw[i]};{gauss_pts[i]}\n")
    with open(os.path.join(export_folder, "markersizes.csv"), "w") as f:
        for i in range(len(absw)):
            if size_pts[i] > 0:
                f.write(f"{absw[i]};{size_pts[i]}\n")
    with open(os.path.join(export_folder, "markercounts.csv"), "w") as f:
        for i in range(len(absw)):
            if count_pts[i] > 0:
                f.write(f"{absw[i]};{count_pts[i]}\n")
    with open(os.path.join(export_folder, "confidence.csv"), "w") as f:
        for i in range(len(absw)):
            if conf_pts[i] > 0:
                f.write(f"{absw[i]};{conf_pts[i]}\n")
    with open(os.path.join(export_folder, "cropsizes.csv"), "w") as f:
        for i in range(len(absw)):
            if cpsz_pts[i] > 0:
                f.write(f"{absw[i]};{cpsz_pts[i]}\n")

    with open(os.path.join(export_folder, "markerAlign.csv"), "w") as f:
        for i in range(len(absw)):
            if marker_align[i] > 0:
                f.write(f"{absw[i]};{marker_align[i]}\n")

    with open(os.path.join(export_folder, "est_error.csv"), "w") as f:
        for i in range(len(absw)):
            f.write(f"{absw[i]};{est_errors[i]}\n")

    with open(q_indcs_file, "w") as fret:
        fret.write(
            f"Image,pred,theta,abw,est_error,rectangular,gauss,markersize,markercount,confidence,markerangle,rect_err,"
            f"gauss_err,markersize_err,markercount_err,confidence_err,angle_err,rect_std,gauss_std,size_std,count_std,"
            f"conf_std,angle_std,rect_{QualityMeasurements.MODE},"
            f"gauss_{QualityMeasurements.MODE},size_{QualityMeasurements.MODE},count_{QualityMeasurements.MODE},"
            f"conf_{QualityMeasurements.MODE},angle_{QualityMeasurements.MODE},ges_{QualityMeasurements.MODE}\n")
        for f in files:
            pred = dist_theta_dict[f][0]
            theta = dist_theta_dict[f][1]
            abw = abs(dist_theta_dict[f][0] - tar_dist) / tar_dist
            rect = rect_dict[f] if f in rect_dict.keys() else 1
            gauss = gauss_dict[f] if f in gauss_dict.keys() else 10
            size = size_dict[f] if f in size_dict.keys() else 2
            count = count_dict[f] if f in count_dict.keys() else 10
            conf = conf_dict[f] if f in conf_dict.keys() else 0
            mal = marker_align_dict[f] if f in marker_align_dict.keys() else -1
            err_rect = error_est_rect(rect)
            err_gauss = error_est_gauss(gauss)
            err_size = error_est_size(size)
            err_count = error_est_counts(count)
            err_conf = error_est_conf(conf)
            err_angl = error_est_angle(theta)
            std_rect = est_std_rect(rect)
            std_gauss = est_std_gauss(gauss)
            std_size = est_std_size(size)
            std_count = est_std_count(count)
            std_conf = est_std_conf(conf)
            std_angl = est_std_angle(mal)

            r_avg = 0.553
            g_avg = 0.103
            s_avg = 0.128
            a_avg = 2
            c_avg = 0.84
            t_avg = 0.08
            fkt_rect = error_fkt(rect, g_avg, s_avg, a_avg, c_avg, t_avg)
            fkt_gaus = error_fkt(r_avg, gauss, s_avg, a_avg, c_avg, t_avg)
            fkt_size = error_fkt(r_avg, g_avg, size, a_avg, c_avg, t_avg)
            fkt_cout = error_fkt(r_avg, g_avg, s_avg, count, c_avg, t_avg)
            fkt_conf = error_fkt(r_avg, g_avg, s_avg, a_avg, conf, t_avg)
            fkt_angl = error_fkt(r_avg, g_avg, s_avg, a_avg, c_avg, mal)
            fkt_gesa = error_fkt(rect, gauss, size, count, conf, mal)


            if pred < 0:
                est_err = 1000
            else:
                est_err = error_fkt(rect, gauss, size, count, conf, theta)
            fret.write(
                f"{f},{pred},{theta},{abw},{est_err},{rect},{gauss},{size},{count},{conf},{mal},{err_rect},{err_gauss},"
                f"{err_size},{err_count},{err_conf},{err_angl},{std_rect},{std_gauss},{std_size},{std_count},"
                f"{std_conf},{std_angl},"
                f"{fkt_rect},{fkt_gaus},{fkt_size},{fkt_cout},{fkt_conf},{fkt_angl},{fkt_gesa}\n")

    print(
        f"Average Quality: Dist:{np.average(absw) * 100:.1f}%,"
        f" Rect:{100 * np.average([x for x in rect_pts if x > 0]):.1f}%, Gauss:"
        f"{np.average([x for x in gauss_pts if x > 0]):.4f},"
        f" Size:{100 * np.average([x for x in size_pts if x > 0]):.1f}%, NoMark:"
        f"{np.average([x for x in count_pts if x > 0]):.1f},"
        f" EstErr:{100 * np.average([x for x in est_errors if x > 0]):.1f}%")

    return np.average([x for x in rect_pts if x > 0]), np.average([x for x in gauss_pts if x > 0]), np.average(
    [x for x in size_pts if x > 0]), np.average([x for x in count_pts if x > 0]), np.average(
    [x for x in est_errors if x > 0])


def reevaluate_with_quality(q_indcs, erg_folder, dst_folder=None, angle_thrsh=np.pi / 6, plot_hists=False,
                            rem_outer=0.0, dist_th=0.0, dist_num=0, zero_th=0.0):
    angle_resfoler = os.path.join(erg_folder, "angles")
    os.makedirs(angle_resfoler, exist_ok=True)
    print("Angle Resfolder: ", angle_resfoler)

    def eval_distarr(dists, logfile=None, plotfile=None, lims=None):
        stabw, wahrsch_f, mittl_f = measurementError(dists)
        avg = np.average(dists)
        med = np.median(dists)
        std = np.std(dists)
        oky = 1
        ims = len(dists)
        typ = "Test"
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"{typ} - Average: {avg:.2f}nm\n")
                f.write(f"{typ} - Median: {med:.2f}nm\n")
                f.write(f"{typ} - Stabw: {std:.2f}nm\n")
                f.write(f"{typ} - EvalOK: {100 * oky:.2f}%\n")
                f.write(f"{typ} - Crops: {ims:.2f}\n")
                f.write(f"{typ} - StabwMWT: {stabw:.2f}nm\n")
                f.write(f"{typ} - Wahrsch.F.MWT: {wahrsch_f:.2f}nm\n")
                f.write(f"{typ} - Mittl.F.MWT: {mittl_f:.2f}nm\n")

        if plotfile is not None:
            fig, ax = plt.subplots()
            ax.hist(dists, bins=max(10, int(np.sqrt(len(dists)))))
            if lims is not None:
                ax.set_xlim(left=lims[0], right=lims[1])
            ax.set_xlabel("Size")
            ax.set_ylabel("Number of Images")
            if std != 0:
                ax2 = ax.twinx()
                if lims is None:
                    xs = np.linspace(min(dists), max(dists), 100)
                else:
                    xs = np.linspace(lims[0], lims[1], 300)
                f = lambda x: np.exp(-((x - avg) ** 2) / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)
                ys = np.vectorize(f)(xs)
                ax2.plot(xs, ys, color='red')
                ax2.set_ylabel("PDF", color='red')
            plt.title("Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm, Imgs: {}".format(avg, med, std, len(dists)))
            plt.savefig(plotfile, bbox_inches='tight')
            plt.cla()
            plt.clf()
            plt.close()
        return ims, avg, med, std, stabw

    fns = []
    dists = []
    thetas = []
    abws = []
    est_errs = []
    rects = []
    gausss = []
    sizes = []
    counts = []
    confs = []
    marker_angles = []
    err_rects = []
    err_gausss = []
    err_sizes = []
    err_counts = []
    err_confs = []
    weights = []

    cur_rects = []
    cur_gauss = []
    cur_sizes = []
    cur_count = []
    cur_confs = []
    cur_angls = []
    cur_gesas = []
    #



    #             fret.write(f"{f},{pred},{abw},{est_err},{rect},{gauss},
    #             {size},{count},{conf},{err_rect},{err_gauss},{err_size},
    #             {err_count},{err_conf}\n")
    first = True
    with open(q_indcs, "r") as f:
        for line in tqdm(f, desc="Reading Quality indcs"):
            if first:
                first = False
                continue
            parts = line.strip().split(",")
            fns.append(parts[0])
            dists.append(float(parts[1]))
            thetas.append(float(parts[2]))
            abws.append(float(parts[3]))
            ee = parts[4]
            if ee == "nan" or float(ee) > 1e8:
                ee = np.infty
            est_errs.append(float(ee))
            rects.append(float(parts[5]))
            gausss.append(float(parts[6]))
            sizes.append(float(parts[7]))
            counts.append(float(parts[8]))
            confs.append(float(parts[9]))
            marker_angles.append(float(parts[10]))
            err_rects.append(float(parts[11]))
            err_gausss.append(float(parts[12]))
            err_sizes.append(float(parts[13]))
            err_counts.append(float(parts[14]))
            err_confs.append(float(parts[15]))

            cur_rects.append(float(parts[23]))
            cur_gauss.append(float(parts[24]))
            cur_sizes.append(float(parts[25]))
            cur_count.append(float(parts[26]))
            cur_confs.append(float(parts[27]))
            cur_angls.append(float(parts[28]))
            cur_gesas.append(float(parts[29]))

    # Plot Stuff over threshold

    dists_and_errs = []
    limits = (min(dists), min(150, max(dists)))
    for i in range(len(dists)):
        dists_and_errs.append((dists[i], est_errs[i], fns[i], thetas[i]))

    dists_and_errs_srted = sorted(dists_and_errs, key=lambda x: x[1])
    # plt.plot(dists_and_errs_srted[:][1])
    # plt.title("1")
    # plt.show()
    #
    # plt.plot(sorted(est_errs))
    # plt.title("Est errs Sorted")
    # plt.show()

    with open("DistAndErr.csv", "w") as logfile:
        for i in range(len(dists_and_errs_srted)):
            logfile.write(
                f"{dists_and_errs_srted[i][0]},{dists_and_errs_srted[i][1]},{dists_and_errs_srted[i][2]},"
                f"{dists_and_errs_srted[i][3]}\n")

    imgs = []
    err_trshs = []
    avgs = []
    meds = []
    stds = []
    stabws = []
    fld = os.path.join(erg_folder, "Hists")
    os.makedirs(fld, exist_ok=True)
    idx = 0

    opt_average = None
    opt_median = None
    opt_std = None
    opt_stabw = np.infty

    opt_fns = None
    opt_dists = None
    opt_errs = None
    opt_thetas = None

    # Plot polar with color

    plt.set_cmap(cm.RdYlGn)
    c_array = [-1 * min(5, x) for x in est_errs]
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(thetas, dists, c=c_array)
    ax.grid(True)
    ax.set_title("x direction")
    ax.set_rmax(min(max(dists), 90))

    plt.savefig(os.path.join(angle_resfoler, "PolarPlotColors.png"))
    plt.close()
    min_th = 0.2 * len(dists_and_errs)

    plt_dist = int(len([x for x in range(len(dists_and_errs), 1, -1)]) / 40) if plot_hists else np.infty
    if plt_dist == 0:
        plt_dist = 2
    for i in tqdm(range(len(dists_and_errs), 1, -1), desc="Testing out ErrThrshs"):
        idx += 1
        des = copy.deepcopy(dists_and_errs_srted[:i])
        # if idx % 50 == 0:
        #     plt.plot(des[:][1])
        #     plt.title(f"Des idx {idx}")
        #     plt.show()
        img = os.path.join(fld, f"Image{str(idx).zfill(4)}.png") if idx % plt_dist == 0 else None
        dists_loc = [x[0] for x in des]
        thetas_loc = [x[3] for x in des]

        ims, avg, med, std, stabw = eval_distarr(copy.deepcopy(dists_loc), logfile=None, plotfile=img, lims=limits)

        err_trsh = des[-1][1]
        # print(f"Testing i={i} with errs {des[0][1]} - {des[-1][1]}")
        # err_trshs.append(min(err_trsh, 10000))
        err_trshs.append(err_trsh)
        imgs.append(ims)
        avgs.append(avg)
        meds.append(med)
        stds.append(std)
        stabws.append(stabw)

        if i > min_th and stabw < opt_stabw:
            opt_average = avg
            opt_median = med
            opt_dists = copy.deepcopy(dists_loc)
            opt_thetas = copy.deepcopy(thetas_loc)
            opt_std = std
            opt_stabw = stabw
            opt_fns = [x[2] for x in des]
            opt_errs = [x[1] for x in des]


    if rem_outer is not None and rem_outer != 0.0:
        print("Removing outer")

        temp = []
        for i in range(len(opt_fns)):
            temp.append((abs(opt_dists[i] - opt_average), opt_fns[i], opt_errs[i], opt_dists[i], opt_thetas[i]))

        t_srt = sorted(temp, key = lambda x : x[0])

        t_srt = t_srt[:int((1-rem_outer) * len(temp))]
        opt_dists = []
        opt_fns = []
        opt_errs = []
        opt_thetas = []
        for tp in t_srt:
            abw, fn, err, d, t = tp
            opt_dists.append(d)
            opt_fns.append(fn)
            opt_errs.append(err)
            opt_thetas.append(t)
        opt_average = np.average(opt_dists)
        opt_median = np.median(opt_dists)

        ims, avg, med, std, stabw = eval_distarr(copy.deepcopy(opt_dists), logfile=None, plotfile=None, lims=limits)
        opt_stabw = stabw
        opt_std = std



    if dist_th != 0.0:
        temp = []
        for i in range(len(opt_fns)):
            temp.append((opt_fns[i], opt_errs[i], opt_dists[i], opt_thetas[i]))

        t_srt = sorted(temp, key=lambda x: x[2])

        opt_dists = []
        opt_fns = []
        opt_errs = []
        opt_thetas = []
        d_th = opt_average * dist_th
        for i, tp in enumerate(t_srt):
            fn, err, d, t = tp
            if i < dist_num or i > len(t_srt) - dist_num - 1:
                continue
            if d - t_srt[i-dist_num][2] > d_th:
                continue
            if t_srt[i+dist_num][2] - d > d_th:
                continue
            opt_dists.append(d)
            opt_fns.append(fn)
            opt_errs.append(err)
            opt_thetas.append(t)
        opt_average = np.average(opt_dists)
        opt_median = np.median(opt_dists)

        ims, avg, med, std, stabw = eval_distarr(copy.deepcopy(opt_dists), logfile=None, plotfile=None, lims=limits)
        opt_stabw = stabw
        opt_std = std

    if zero_th != 0.0:
        temp = []
        for i in range(len(opt_fns)):
            temp.append((opt_fns[i], opt_errs[i], opt_dists[i], opt_thetas[i]))


        opt_dists = []
        opt_fns = []
        opt_errs = []
        opt_thetas = []
        d_th = opt_average * zero_th
        smaller = []
        larger = []

        opt_pairs = []

        for elem in temp:
            if elem[2] < opt_average:
                smaller.append(elem)
            else:
                larger.append(elem)

        smaller = sorted(smaller, key = lambda x : -x[2])
        larger = sorted(larger, key = lambda  x : x[2])

        opt_pairs.append(smaller[0])
        opt_pairs.append(larger[0])
        for i in range(1, len(smaller)):
            if abs(smaller[i][2] - smaller[i-1][2]) < d_th:
                opt_pairs.append(smaller[i])
            else:
                break
        for i in range(1, len(larger)):
            if abs(larger[i][2] - larger[i-1][2]) < d_th:
                opt_pairs.append(larger[i])
            else:
                break

        for tp in opt_pairs:
            fn, err, d, t = tp
            opt_dists.append(d)
            opt_fns.append(fn)
            opt_errs.append(err)
            opt_thetas.append(t)




        opt_average = np.average(opt_dists)
        opt_median = np.median(opt_dists)

        ims, avg, med, std, stabw = eval_distarr(copy.deepcopy(opt_dists), logfile=None, plotfile=None, lims=limits)
        opt_stabw = stabw
        opt_std = std

    with open(os.path.join(erg_folder, "err_thrs.csv"), "w") as f:
        f.write(f"err_thrsh,imgs,avgs,meds,stds,stabws\n")
        for i in range(len(err_trshs)):
            f.write(f"{err_trshs[i]},{imgs[i]},{avgs[i]},{meds[i]},{stds[i]},{stabws[i]}\n")

    with open(os.path.join(erg_folder, "OptResults.csv"), "w") as optifle:
        optifle.write("Filename;Error;Distance;Theta\n")
        for i in range(len(opt_fns)):
            optifle.write(f"{opt_fns[i]};{opt_errs[i]};{opt_dists[i]};{opt_thetas[i]}\n")


    with open(os.path.join(erg_folder, "AllImgsIsOpt.csv"), "w") as optifle:
        optifle.write("Filename;Error;RectE;GaussE;SizeE;CountE;ConfE;AnglE;Dist;InOPT\n")
        for i in range(len(fns)):
            optifle.write(f"{fns[i]};{cur_gesas[i]};{cur_rects[i]};{cur_gauss[i]};{cur_sizes[i]};"
                          f"{cur_count[i]};{cur_confs[i]};{cur_angls[i]};{dists[i]};{1 if fns[i] in opt_fns else 0}\n")

    PLOT_ALL = True
    if PLOT_ALL:
        plt.cla()
        plt.clf()
        plt.plot(err_trshs, imgs)
        plt.title("Image over Threshold")
        plt.xlabel("Threshold")
        plt.xlim([0, min(5, max(err_trshs))])
        plt.ylabel("Images")
        plt.savefig(os.path.join(erg_folder, "im_thrsh.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(err_trshs, avgs)
        plt.xlim([0, min(5, max(err_trshs))])
        plt.title("Average over Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Average")
        plt.savefig(os.path.join(erg_folder, "avg_thrsh.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(err_trshs, meds)
        plt.xlim([0, min(5, max(err_trshs))])
        plt.title("Median over Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Median")
        plt.savefig(os.path.join(erg_folder, "med_thrsh.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(err_trshs, stds)
        plt.xlim([0, min(5, max(err_trshs))])
        plt.title("Std over Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Std.Deviation")
        plt.savefig(os.path.join(erg_folder, "std_thrsh.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(err_trshs, stabws)
        plt.xlim([0, min(5, max(err_trshs))])
        plt.title("Std. MWT over Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Std. average")
        plt.savefig(os.path.join(erg_folder, "stdmwt_thrsh.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(imgs, avgs)
        plt.title("Average over Images")
        plt.xlabel("Images")
        plt.ylabel("Average")
        plt.savefig(os.path.join(erg_folder, "avg_imgs.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(imgs, meds)
        plt.title("Median over Images")
        plt.xlabel("Images")
        plt.ylabel("Median")
        plt.savefig(os.path.join(erg_folder, "med_imgs.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(imgs, stds)
        plt.title("Std over Images")
        plt.xlabel("Images")
        plt.ylabel("Std.Deviation")
        plt.savefig(os.path.join(erg_folder, "std_imgs.png"))
        plt.cla()
        plt.clf()
        plt.close()
        plt.plot(imgs, stabws)
        plt.title("Std. MWT over Images")
        plt.xlabel("Images")
        plt.ylabel("Std. average")
        plt.savefig(os.path.join(erg_folder, "stdmwt_imgs.png"))
        plt.cla()
        plt.clf()
        plt.close()
        fig, ax = plt.subplots()
        ax.plot(err_trshs, avgs)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Average")
        ax.set_xlim(left=0, right=5)
        ax2 = ax.twinx()
        ax2.plot(err_trshs, stabws, color='red')
        ax2.set_ylabel("Std.MWT", color='red')
        plt.title("Avg and std(avg) over threshold")
        plt.savefig(os.path.join(erg_folder, "avg_std_thrsh.png"))
        plt.cla()
        plt.clf()
        plt.close()
        fig, ax = plt.subplots()
        ax.plot(imgs, avgs)
        ax.set_xlabel("images")
        ax.set_ylabel("Average")
        ax2 = ax.twinx()
        ax2.plot(imgs, stabws, color='red')
        ax2.set_ylabel("Std.MWT", color='red')
        plt.title("Avg and std(avg) over images")
        plt.savefig(os.path.join(erg_folder, "avg_std_imgs.png"))
        plt.cla()
        plt.clf()
        plt.close()
    # ToDo: Return optimum

    print("Avg Dists: ", np.average(opt_dists))
    print("Avg Geg: ", opt_average)

    if dst_folder is not None:
        ois = os.path.join(erg_folder, "optimum_image_set")
        os.makedirs(ois, exist_ok=True)
        for img in tqdm(opt_fns, desc="Copying optimal images"):
            try:
                imgf = os.path.join(dst_folder, img + ".png")
                shutil.copy(imgf, os.path.join(ois, img + ".png"))
            except FileNotFoundError:
                print('Could not Copy file {}'.format(img))

    if dst_folder is not None:

        def in_y(theta):
            for i in range(-10, 10):
                if i * np.pi - angle_thrsh <= theta < i * np.pi + angle_thrsh:
                    return True
            return False
            # return -angle_thrsh <= theta < angle_thrsh or np.pi - angle_thrsh <= theta < np.pi + angle_thrsh or \
            #        2 * np.pi - angle_thrsh <= theta < 2 * np.pi + angle_thrsh

        def in_x(theta):
            for j in range(-10, 10):
                i = 2*j + 1
                if (i * np.pi / 2) - angle_thrsh <= theta < (i * np.pi / 2) + angle_thrsh:
                    return True
            return False
            # return (np.pi / 2) - angle_thrsh <= theta < (np.pi / 2) + angle_thrsh or \
            #        (3 * np.pi / 2) - angle_thrsh <= theta < (3 * np.pi / 2) + angle_thrsh

        angle_folder = os.path.join(erg_folder, "angle_imageset")
        angle_x = os.path.join(angle_folder, "x")
        angle_y = os.path.join(angle_folder, "y")
        angle_all = os.path.join(angle_folder, "all")
        resdict = {}
        os.makedirs(angle_x, exist_ok=True)
        os.makedirs(angle_all, exist_ok=True)
        os.makedirs(angle_y, exist_ok=True)
        for i in tqdm(range(len(fns)), desc="Copying files"):
            try:
                filename = fns[i]
                theta = thetas[i]
                thetafn = int(round((theta / np.pi) * 180))
                cls = ""
                if in_x(theta):
                    fnres = os.path.join(angle_x, os.path.basename(filename).split(".")[0] + f"_{thetafn}.png")
                    shutil.copy(os.path.join(dst_folder, filename + ".png"), fnres)
                    cls += "x"
                if in_y(theta):
                    fnres = os.path.join(angle_y, os.path.basename(filename).split(".")[0] + f"_{thetafn}.png")
                    shutil.copy(os.path.join(dst_folder, filename + ".png"), fnres)
                    cls += "y"


                if thetafn in resdict.keys():
                    idx = resdict[thetafn]
                    resdict[thetafn] += 1
                else:
                    idx = 0
                    resdict[thetafn] = 1
                fnres = os.path.join(angle_all, f"I{thetafn}_{idx}_{cls}.png")
                shutil.copy(os.path.join(dst_folder, filename + ".png"), fnres)
            except FileNotFoundError:
                continue

    fig, ax = plt.subplots()
    ax.hist(opt_dists, bins=max(10, int(np.sqrt(len(dists)))))
    ax.set_xlabel("Size")
    ax.set_xlim(left=0, right=130)
    ax.set_ylabel("Number of Images")
    if opt_std != 0:
        ax2 = ax.twinx()
        xs = np.linspace(0, 130, 300)
        f = lambda x: np.exp(-((x - opt_average) ** 2) / (2 * opt_std ** 2)) / np.sqrt(2 * np.pi * opt_std ** 2)
        ys = np.vectorize(f)(xs)
        ax2.plot(xs, ys, color='red')
        ax2.set_ylabel("PDF", color='red')
    plt.title("Avg: {:.1f}nm, Med: {:.1f}nm, StdMWT: {:.1f}nm, Imgs: {}".format(opt_average, opt_median, opt_stabw,
                                                                                len(opt_dists)))
    plt.savefig(os.path.join(erg_folder, "optimum_hist.png"), bbox_inches='tight')
    plt.close()

    # Reevaluate Angles
    evaluate_angles(opt_dists, opt_thetas, angle_resfoler, angle_thrsh=angle_thrsh)

    with open(os.path.join(erg_folder, "Result.txt"), "w") as f:
        f.write("Result: {}nm, Stabw: {}nm".format(opt_average, opt_stabw))

    return opt_average, opt_stabw


def evaluate_angles(distances, thetas, resfolder, angle_thrsh=np.pi / 6):
    def d2b(deg):
        return deg * np.pi / 180

    def b2d(bog):
        return bog * 180 / np.pi

    def plot_stuff(dists, direction="None"):
        avg = np.average(dists)
        med = np.median(dists)
        std = np.std(dists)
        ims = len(dists)

        stabw, wahrsch_f, mittl_f = measurementError(dists)

        fig, ax = plt.subplots()
        ax.hist(dists, bins=max(10, int(np.sqrt(len(dists)))))
        ax.set_xlabel("Size")
        ax.set_ylabel("Number of Images")
        if std != 0:
            ax2 = ax.twinx()
            xs = np.linspace(min(dists), max(dists), 100)
            f = lambda x: np.exp(-((x - avg) ** 2) / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)
            ys = np.vectorize(f)(xs)
            ax2.plot(xs, ys, color='red')
            ax2.set_ylabel("PDF", color='red')
        plt.title("Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm, Imgs: {}".format(avg, med, std, len(dists)))
        plt.savefig(os.path.join(resfolder, f"dists_{direction}.png"), bbox_inches='tight')
        plt.close()

    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.scatter(thetas, distances)
    # ax.grid(True)
    # ax.set_title("Results with thetas")
    # plt.show()

    # def in_y(theta):
    #     return -angle_thrsh <= theta < angle_thrsh or np.pi - angle_thrsh <= theta < np.pi + angle_thrsh or \
    #            2 * np.pi - angle_thrsh <= theta < 2 * np.pi + angle_thrsh
#
    # def in_x(theta):
    #     return (np.pi / 2) - angle_thrsh <= theta < (np.pi / 2) + angle_thrsh or \
    #            (3 * np.pi / 2) - angle_thrsh <= theta < (3 * np.pi / 2) + angle_thrsh

    def in_y(theta):
        for i in range(-10, 10):
            if i * np.pi - angle_thrsh <= theta < i * np.pi + angle_thrsh:
                return True
        return False
        # return -angle_thrsh <= theta < angle_thrsh or np.pi - angle_thrsh <= theta < np.pi + angle_thrsh or \
        #        2 * np.pi - angle_thrsh <= theta < 2 * np.pi + angle_thrsh

    def in_x(theta):
        for j in range(-10, 10):
            i = 2 * j + 1
            if (i * np.pi / 2) - angle_thrsh <= theta < (i * np.pi / 2) + angle_thrsh:
                return True
        return False
        # return (np.pi / 2) - angle_thrsh <= theta < (np.pi / 2) + angle_thrsh or \
        #        (3 * np.pi / 2) - angle_thrsh <= theta < (3 * np.pi / 2) + angle_thrsh

    assert len(distances) == len(thetas)

    x_dists = []
    x_thetas = []
    y_dists = []
    y_thetas = []

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(thetas, distances, marker="x", c="black")
    ax.grid(True)
    ax.set_title("x direction")
    ax.set_rmax(min(max(distances), 90))
    plt.savefig(os.path.join(resfolder, "PolarPlotAll.png"))
    plt.close()

    for i in range(len(distances)):
        if in_x(thetas[i]) and 30 < distances[i] < 150:
            x_dists.append(distances[i])
            x_thetas.append(thetas[i])
        if in_y(thetas[i]) and 30 < distances[i] < 150:
            y_dists.append(distances[i])
            y_thetas.append(thetas[i])

    if len(x_dists) > 0:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(x_thetas, x_dists)
        ax.grid(True)
        ax.set_title("x direction")
        ax.set_rmax(min(max(x_dists), 90))
        plt.savefig(os.path.join(resfolder, "PolarPlotX.png"))
        plt.close()
        print("Save Angle Eval as ", os.path.join(resfolder, "PolarPlotX.png"))
        plot_stuff(x_dists, "X")

    if len(y_dists) > 0:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(y_thetas, y_dists)
        ax.grid(True)
        ax.set_title("y direction")
        ax.set_rmax(min(max(y_dists), 90))
        plt.savefig(os.path.join(resfolder, "PolarPlotY.png"))
        plt.close()
        plot_stuff(y_dists, "Y")

    with open(os.path.join(resfolder, "polar_all.csv"), "w") as f:
        for i in range(len(distances)):
            if thetas[i] > 0:
                f.write(f"{thetas[i]};{distances[i]}\n")

    with open(os.path.join(resfolder, "polar_x.csv"), "w") as f:
        for i in range(len(distances)):
            if thetas[i] > 0 and in_x(thetas[i]):
                f.write(f"{thetas[i]};{distances[i]}\n")

    with open(os.path.join(resfolder, "polar_y.csv"), "w") as f:
        for i in range(len(distances)):
            if thetas[i] > 0 and in_y(thetas[i]):
                f.write(f"{thetas[i]};{distances[i]}\n")

    avgx = np.average(x_dists) if len(x_dists) > 0 else -1
    medx = np.median(x_dists) if len(x_dists) > 0 else -1
    stdx = np.median(x_dists) if len(x_dists) > 0 else -1
    smix = np.std(x_dists) / np.sqrt(len(x_dists)) if len(x_dists) > 0 else -1
    numx = len(x_dists)
    avgy = np.average(y_dists) if len(y_dists) > 0 else -1
    medy = np.median(y_dists) if len(y_dists) > 0 else -1
    stdy = np.median(y_dists) if len(y_dists) > 0 else -1
    smiy = np.std(y_dists) / np.sqrt(len(y_dists)) if len(y_dists) > 0 else -1
    numy = len(y_dists)

    with open(os.path.join(resfolder, "results.txt"), "w") as f:
        f.write(f"Average X: {avgx:.3f} nm\n")
        f.write(f"Median X: {medx:.3f} nm\n")
        f.write(f"Std.deriv X: {stdx:.3f} nm\n")
        f.write(f"Std MWT X: {smix:.3f} nm\n")
        f.write(f"Number X: {numx}\n\n")

        f.write(f"Average Y: {avgy:.3f} nm\n")
        f.write(f"Median Y: {medy:.3f} nm\n")
        f.write(f"Std.deriv Y: {stdy:.3f} nm\n")
        f.write(f"Std MWT Y: {smiy:.3f} nm\n")
        f.write(f"Number Y: {numy}\n")


def copy_yolo_pred(yolo_tempdict, yolo_basefolder, crop_folder, sample_png):
    for k in yolo_tempdict.keys():
        dest = yolo_tempdict[k]
        sp = os.path.basename(sample_png)
        cf = os.path.basename(crop_folder)
        dest = dest.replace(sp, cf)
        try:
            shutil.copy(os.path.join(yolo_basefolder, k), os.path.join(dest, "YOLOprediction.png"))
        except FileNotFoundError:
            os.makedirs(dest, exist_ok=False)
            shutil.copy(os.path.join(yolo_basefolder, k), os.path.join(dest, "YOLOprediction.png"))

def rename_samplelabels(yolo_tempdict, provided_fld, sample_png, resfld):
    for key in tqdm(yolo_tempdict.keys(), desc="Renaming Sample Labels", total=len(yolo_tempdict.keys())):
        newname = key
        oldname = yolo_tempdict[key]
        last_wholename = sample_png.split("\\")[-1]
        # print("LWN: ", last_wholename )
        parts = oldname.split("\\")
        for i in range(len(parts)):
            if parts[i] == last_wholename:
                subpth = "\\".join(parts[i+1:])
                break
        new_total_path = provided_fld + "\\" + subpth + ".png"
        # print("new total Path: ", new_total_path)
        target_path = os.path.join(resfld, newname)
        # print("target Path: ", new_total_path)
        shutil.copy(new_total_path, target_path)



def evaluate_dataset(folder, resfolder):
    """
    Complete Evaluation of a dataset from bare images to distances
    """

    # Make Directories in Resultfolder
    spm_folder = os.path.join(resfolder, "spm")
    sample_png = os.path.join(resfolder, "sample_png")
    pp_sample = os.path.join(resfolder, "pp_sample")
    yolo_pred = os.path.join(resfolder, "yolo_prediction")
    yolo_model = os.path.join("Models", "YOLO", "Mix4k256pt.pt")
    crop_origami = os.path.join(resfolder, "crop_origami")
    all_crop_origami = os.path.join(resfolder, "crop_origami_allpng")
    all_crop_origami_resc = os.path.join(resfolder, "crop_origami_allpng_resc")
    pp_crop_origami_npy = os.path.join(resfolder, "pp_crop_origami_npy")
    pp_crop_origami_png = os.path.join(resfolder, "pp_crop_origami_png")
    ss_results = os.path.join(resfolder, "ss_results")
    ss_labels = os.path.join(resfolder, "ss_labels")
    ss_model = os.path.join("Models", "SS", "ModelSize35.pth")
    # ss_model = os.path.join("Models", "SS", "ModelNewPP2410.pth")

    distance_results_all = os.path.join(resfolder, "distance_results_all")
    quality_folder = os.path.join(resfolder, "quality")
    fit_parameter_folder = os.path.join(resfolder, "fit_parameters")
    evaluation_folder = os.path.join(resfolder, "Eval_Results")
    settings_file = os.path.join(evaluation_folder, "settings.txt")
    q_indcs_file = os.path.join(quality_folder, "quality_indcs.csv")
    quality_reeval = os.path.join(resfolder, "quality_reeval")
    quality_method = "Class"
    quality_modelfolder = os.path.join("Models", "ErrorNN")
    yolo_conf_thrsh = 0.6  # 0.82
    print_only_norm = True

    for fld in [sample_png, quality_reeval, all_crop_origami, quality_folder, fit_parameter_folder,
                all_crop_origami_resc, ss_labels, pp_sample, crop_origami, pp_crop_origami_npy,
                pp_crop_origami_png, ss_results, distance_results_all, evaluation_folder]:
        os.makedirs(fld, exist_ok=True)

    all_dists = os.path.join(evaluation_folder, "all_dists.csv")

    # Copy SPM files to Resultfolder
    shutil.copytree(folder, spm_folder, dirs_exist_ok=True)
    IMAGE_SIZE = 64
    # Turn SPM to grayscale
    spm_files = []
    spm_foldernames = []

    spm_files, spm_foldernames = get_folder_files(spm_folder)

    # 1. Rename files to avoid errors with _ and .
    try:
        spm_files, spm_foldernames = rename_files(spm_files, spm_foldernames)
    except FileExistsError:
        pass
    png_foldernames = []
    for i in tqdm(range(len(spm_files)), desc="Reading SPM"):
        f = spm_files[i]
        fld = spm_foldernames[i]
        # print(fld)
        spmf = str(spm_folder).split("\\")[-1]
        flderg = fld.split("\\")[-1]
        assert len(spm_files) == len(spm_foldernames)
        assert "\\" + spmf + "\\" in fld, fld + spmf
        parts = str(fld).split("\\")
        for i in range(len(parts)):
            if parts[i] == spmf:
                flderg = "\\".join(parts[i + 1:])
                break
        # print(flderg)
        # input()
        filename = os.path.join(fld, f)
        resf = os.path.join(sample_png, flderg, ".".join(f.split(".")[:-1]))
        try:
            read_input(filename, resf)
        except FileExistsError:
            pass
        png_foldernames.append(resf)

    # 2. prepare for YOLO input
    yaml_file, yolo_tempdict, dataset_path, yolo_pp_args = prepare_for_yolo(png_foldernames,
                                                                            os.path.join(resfolder, "temp_yolo"),
                                                                            pp_sample)

    # 3. Perform YOLO Prediction
    if not os.path.isdir(yolo_pred):
        predict_yolo(yolo_model, yaml_file, conf_thrsh=yolo_conf_thrsh)
        yolo_res = os.path.join("EvalChain", "EvalChain")
        if len(os.listdir("EvalChain")) != 1:
            yolo_res = os.path.join("EvalChain", f"EvalChain{len(os.listdir('EvalChain'))}")
        shutil.move(yolo_res, yolo_pred)
        print(yolo_res)
        print(yolo_pred)
    clear_yolo_dataset(dataset_path)

    # 4. Crop images
    drop_border = True
    extend = 1.5
    crp_tempdict, conf_dict, crpsize_dict = crop_images(yolo_tempdict, yolo_pred, crop_origami, sample_png,
                                                        tempfolder2=all_crop_origami, drop_border=drop_border,
                                                        extend=extend)

    # 5. Preprocess cropped Images for SS
    pp_origami_args = preprocess_origami(all_crop_origami, pp_crop_origami_npy, pp_crop_origami_png,
                                         img_size=IMAGE_SIZE)

    # 6. Perform Semantic Segmentation on images
    get_semantic_seg(ss_model, pp_crop_origami_npy, ss_results)

    # 7. get Distances from Images + SS Masks
    extract_gsc_ss(ss_results, ss_labels)

    rescale_orig_imgs(all_crop_origami, all_crop_origami_resc, IMAGE_SIZE)

    distances, _ = get_distances(all_crop_origami_resc, ss_labels, distance_results_all, fit_parameter_folder,
                                 threads=THREADS)
    while distances == False:
        print("Recieved Distances ", distances, "Retyring SemSeg")
        yaml_file, yolo_tempdict, dataset_path, yolo_pp_args = prepare_for_yolo(png_foldernames,
                                                                                os.path.join(resfolder, "temp_yolo"),
                                                                                pp_sample)

        # 3. Perform YOLO Prediction
        if not os.path.isdir(yolo_pred):
            predict_yolo(yolo_model, yaml_file, conf_thrsh=yolo_conf_thrsh)
            yolo_res = os.path.join("EvalChain", "EvalChain")
            if len(os.listdir("EvalChain")) != 1:
                yolo_res = os.path.join("EvalChain", f"EvalChain{len(os.listdir('EvalChain'))}")
            shutil.move(yolo_res, yolo_pred)
            print(yolo_res)
            print(yolo_pred)
        clear_yolo_dataset(dataset_path)

        # 4. Crop images
        drop_border = True
        extend = 1.5
        crp_tempdict, conf_dict, crpsize_dict = crop_images(yolo_tempdict, yolo_pred, crop_origami, sample_png,
                                                            tempfolder2=all_crop_origami,
                                                            drop_border=drop_border, extend=extend)

        # 5. Preprocess cropped Images for SS
        pp_origami_args = preprocess_origami(all_crop_origami, pp_crop_origami_npy, pp_crop_origami_png,
                                             img_size=IMAGE_SIZE)

        get_semantic_seg(ss_model, pp_crop_origami_npy, ss_results)
        # 7. get Distances from Images + SS Masks
        extract_gsc_ss(ss_results, ss_labels)
        rescale_orig_imgs(all_crop_origami, all_crop_origami_resc, IMAGE_SIZE)
        distances, _ = get_distances(all_crop_origami_resc, ss_labels, distance_results_all, fit_parameter_folder,
                                     threads=THREADS)

    make_csv(all_crop_origami_resc, distances, None, all_dists)
    dist_thata_dict = reorganize_files(all_crop_origami_resc, distance_results_all, all_dists, crp_tempdict,
                                       img_size=IMAGE_SIZE)
    avg, med, stabw, ok = export_distances(crop_origami, evaluation_folder)
    print(
        "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm,"
        " Std: {:.1f}nm, Evaluated Origami: {:.1f}% of {}".format(avg,med,stabw,100 * ok,len(crp_tempdict.keys())))
    modify_csv(os.path.join(evaluation_folder, "Results_norm.csv"), delim=";")
    modify_csv(os.path.join(evaluation_folder, "Results_10pct.csv"), delim=";")
    modify_csv(os.path.join(evaluation_folder, "Results_complete.csv"), delim=";")

    visualize_results(evaluation_folder, norm_fp="Results_norm.csv", cmpl_fp="Results_complete.csv",
                      p10_fp="Results_10pct.csv", no_imgs=len(crp_tempdict.keys()), print_only_norm=print_only_norm)

    gauss_acc = gauss_accuracy(fit_parameter_folder)
    size_acc, count_acc = count_compare_labels(ss_labels, threads=THREADS)
    rect_acc = rectangulartiy(ss_labels, THREADS, resolution=4, sparse=0.3, theta_res=9, show=False)
    export_quality(dist_thata_dict, rect_acc, gauss_acc, size_acc, count_acc, conf_dict, crpsize_dict, quality_folder,
                   q_indcs_file, method=quality_method, modelfolder=quality_modelfolder)

    avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval, dst_folder=distance_results_all)

    print(f"Result: D={avg:.2f}nm, std: {stdmwt:.2f}nm")
    with open(settings_file, "w") as f:
        f.write(f"Input: {folder}\n")
        f.write(f"Output: {resfolder}\n")
        f.write(f"spm_folder: {spm_folder}\n")
        f.write(f"sample_png: {sample_png}\n")
        f.write(f"pp_sample: {pp_sample}\n")
        f.write(f"yolo_pred: {yolo_pred}\n")
        f.write(f"yolo_model: {yolo_model}\n")
        f.write(f"crop_origami: {crop_origami}\n")
        f.write(f"all_crop_origami: {all_crop_origami}\n")
        f.write(f"all_crop_origami_resc: {all_crop_origami_resc}\n")
        f.write(f"pp_crop_origami_npy: {pp_crop_origami_npy}\n")
        f.write(f"pp_crop_origami_png: {pp_crop_origami_png}\n")
        f.write(f"ss_results: {ss_results}\n")
        f.write(f"ss_labels: {ss_labels}\n")
        f.write(f"ss_model: {ss_model}\n")
        f.write(f"distance_results_all: {distance_results_all}\n")
        f.write(f"evaluation_folder: {evaluation_folder}\n")
        f.write(f"settings_file: {settings_file}\n")
        f.write(f"all_dists: {all_dists}\n")
        f.write(f"yolo_conf_thrsh: {yolo_conf_thrsh}\n")
        f.write(f"IMAGE_SIZE: {IMAGE_SIZE}\n")
        f.write(f"yaml_file: {yaml_file}\n")
        f.write(f"dataset_path: {dataset_path}\n")
        f.write(f"yolo_pp_args: {yolo_pp_args}\n")
        f.write(f"drop_border: {drop_border}\n")
        f.write(f"pp_origami_args: {pp_origami_args}\n")
        f.write(f"10pRes: {folder}\n")
        f.write(f"Q-method: {quality_method}\n")
        f.write(f"Q-Model: {quality_modelfolder}\n")
        f.write(f"Evaluated origami: {len(crp_tempdict.keys())}\n")
        f.write(
            "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm,"
            " Evaluated Origami: {:.1f}%\n".format(avg,med,stabw,100 * ok))

def rescale_provided_labels(inpt, outpt, size, rem_outer=True):
    binarize = lambda x: 1 if x > 0 else 0
    binarize = np.vectorize(binarize)
    for fn in tqdm(os.listdir(inpt)):
        img = Image.open(os.path.join(inpt, fn))
        img = img.resize((size, size), resample=PIL.Image.NEAREST)
        # img.show()
        arr = np.array(img)[:, :, 0]
        total_arr = copy.deepcopy(arr)
        arr = binarize(arr)
        # plt.imshow(arr)
        # plt.show()

        hk = hoshen_koppelmann(arr)
        # plt.imshow(hk)
        # plt.show()

        mid_idx = hk[int(hk.shape[0]/2), int(hk.shape[1]/2)]
        assert mid_idx != -1
        if mid_idx == -1:
            img.save(os.path.join(outpt, fn))
            continue

        offs = np.argwhere(hk != mid_idx)
        newoffs = []
        for i in range(len(offs)):
            if arr[offs[i][0], offs[i][1]] != 0:
                newoffs.append(offs[i])
        offs = newoffs
        for i in range(len(offs)):
            total_arr[offs[i][0], offs[i][1]] = 0


        # plt.imshow(total_arr)
        # plt.title("Total arr")
        # plt.show()
        img_new = Image.fromarray(total_arr)
        # if len(offs) != 0:
        #     img.show()
        #     img_new.show()




        img_new.save(os.path.join(outpt, fn))

def analyze_orientations(crop_origami, resf, angle_thrsh):
    tbc = [crop_origami]
    folders_to_be_evald = [crop_origami]
    newnames = []

    def get_angle(file):
        with open(file, "r") as f:
            for line in f:
                if line.startswith("Theta") and "deg" not in line:
                    return float(line.split(":")[-1])

    def in_y(theta):
        for i in range(-10, 10):
            if i * np.pi - angle_thrsh <= theta < i * np.pi + angle_thrsh:
                return True
        return False
        # return -angle_thrsh <= theta < angle_thrsh or np.pi - angle_thrsh <= theta < np.pi + angle_thrsh or \
        #        2 * np.pi - angle_thrsh <= theta < 2 * np.pi + angle_thrsh

    def in_x(theta):
        for j in range(-10, 10):
            i = 2 * j + 1
            if (i * np.pi / 2) - angle_thrsh <= theta < (i * np.pi / 2) + angle_thrsh:
                return True
        return False

    def get_parameters_frm_filelist(files):
        xs = 0
        ys = 0
        total = 0
        angles = []
        for file in files:
            angl = get_angle(file)
            if angl >= 0:
                angles.append(angl)
                total += 1
                if in_x(angl):
                    xs += 1
                if in_y(angl):
                    ys += 1

        return xs, ys, total, angles








    def get_params(folder):
        temp = [folder]
        tbe = []

        while len(temp) > 0:
            lf = temp.pop()
            for elem in os.listdir(lf):
                if os.path.isdir(os.path.join(lf, elem)):
                    temp.append(os.path.join(lf, elem))

                if elem == "Image.txt":
                    tbe.append(os.path.join(lf, elem))

        return get_parameters_frm_filelist(tbe)






    while len(tbc) > 0:
        lf = tbc.pop()
        for elem in os.listdir(lf):
            if os.path.isdir(os.path.join(lf, elem)):
                tbc.append(os.path.join(os.path.join(lf, elem)))
                folders_to_be_evald.append(os.path.join(os.path.join(lf, elem)))

    short_fns = []
    co_subf = crop_origami.split("\\")[-1]
    for i in range(len(folders_to_be_evald)):
        name = folders_to_be_evald[i]
        parts = name.split("\\")
        for j in range(len(parts)):
            if parts[j] == co_subf:
                newnames.append(os.path.join(resf, "\\".join(parts[j+1:])))
                short_fns.append(os.path.join("\\".join(parts[j+1:])))

    assert len(folders_to_be_evald) == len(newnames)
    results = [get_params(fld) for fld in folders_to_be_evald]


    maxparts = max(len(x.split("\\")) for x in short_fns)

    with open(os.path.join(resf, "reults.csv"), "w") as f:
        firstline = ""
        for j in range(maxparts + 2):
            firstline += "Name,"
        firstline += "In_X,In_Y,total,,angles\n"
        f.write(firstline)
        for i in range(len(results)):
            line = ""
            parts = short_fns[i].split("\\")
            for j in range(maxparts+2):
                if j < len(parts):
                    line+=f"{parts[j]},"
                else:
                    line += ","
            x, y, total, angles = results[i]
            line += f"{x},{y},{total},,"
            line += ",".join([str(x) for x in angles])
            f.write(line + "\n")












# def evaluate_dataset_xy(folder, resfolder, qual_mode="STD", threads=None, angle_thrsh=np.pi/6, labels=None, use_UNet=True):
#     """
#     Complete Evaluation of a dataset from bare images to distances
#     """
#     if threads is None:
#         threads = os.cpu_count()
#     THREADS = threads
#     labels_provided = labels is not None
#     assert os.path.isdir(folder)
#     print("Selected Input Folder", folder)
#     # Make Directories in Resultfolder
#     spm_folder = os.path.join(resfolder, "spm")
#     sample_png = os.path.join(resfolder, "sample_png")
#     save_pp_of = os.path.join(resfolder, "sample_png_pp")
#     pp_sample = os.path.join(resfolder, "pp_sample")
#     yolo_pred = os.path.join(resfolder, "yolo_prediction")
#     yolo_model = os.path.join("Models", "YOLO", "Mix4k256_newpt2.pt")
#     crop_origami = os.path.join(resfolder, "crop_origami")
#     all_crop_origami = os.path.join(resfolder, "crop_origami_allpng")
#     all_crop_origami_resc = os.path.join(resfolder, "crop_origami_allpng_resc")
#     pp_crop_origami_npy = os.path.join(resfolder, "pp_crop_origami_npy")
#     pp_crop_origami_png = os.path.join(resfolder, "pp_crop_origami_png")
#     ss_results = os.path.join(resfolder, "ss_results")
#     ss_labels = os.path.join(resfolder, "ss_labels")
#
#     angle_folder = os.path.join(resfolder, "AngleEvaluation")
#     orientations_folder = os.path.join(resfolder, "Orientations")
#
#     label_crops = os.path.join(resfolder, "provided_labels")
#     sample_labels_provided = os.path.join(resfolder, "sample_labels")
#     sample_labels_provided_yolo = os.path.join(resfolder, "provided_labels_yolo")
#     sample_labels_provided_resc = os.path.join(resfolder, "provided_labels_resc")
#
#     # ss_model = os.path.join("Models", "SS", "ModelSize50.pth")
#     # ss_model = os.path.join("Models", "SS", "ModelNewPP2410.pth")
#     # bilinear_model = True
#     # ss_model = "SS_DNA_Train\\Small50px_5k\\Train\\checkpoints\\checkpoint_epoch10.pth" # cpold/4
#     ss_model = os.path.join("Models", "SS", "NewModelSmallerMarkers.pth")
#
#     bilinear_model = False
#     distance_results_all = os.path.join(resfolder, "distance_results_all")
#     quality_folder = os.path.join(resfolder, "quality")
#     fit_parameter_folder = os.path.join(resfolder, "fit_parameters")
#     evaluation_folder = os.path.join(resfolder, "Eval_Results")
#     tempdict_file = os.path.join(evaluation_folder, "tempdict.csv")
#     yolo_tempdict_file = os.path.join(evaluation_folder, "tempdict_yolo.csv")
#     settings_file = os.path.join(evaluation_folder, "settings.txt")
#     q_indcs_file = os.path.join(quality_folder, "quality_indcs.csv")
#     quality_reeval = os.path.join(resfolder, "quality_reeval")
#     quality_reeval_ro = os.path.join(resfolder, "quality_reeval_ro")
#     quality_reeval_th01_1 = None#  os.path.join(resfolder, "quality_reeval_th01_1")
#     quality_reeval_th01_3 = None#  os.path.join(resfolder, "quality_reeval_01_3")
#     quality_reeval_th02_3 = None#  os.path.join(resfolder, "quality_reeval_02_3")
#     quality_reeval_z_th1 = None #  os.path.join(resfolder, "quality_reeval_zth1")
#     quality_reeval_z_th3 = os.path.join(resfolder, "quality_reeval_zth3")
#     eval_SS_ShowAll = os.path.join(resfolder, "EvalSSDoku")
#     # eval_SS_ShowAll = None
#
#     quality_method = "Class"
#     quality_modelfolder = os.path.join("Models", "ErrorNN")
#     yolo_conf_thrsh = 0.70  # 0.82
#     print_only_norm = True
#     ss_thrsh = 0.4
#     crpsize_stds = 3  # 3
#     rem_outer=0
#     crop_pp=True
#     export_hists_folder = os.path.join(resfolder, "Histograms")
#     quality_mode = qual_mode
#     QualityMeasurements.MODE = quality_mode
#     fit_qfile = os.path.join("Models", "FitParameters", "FIT368.csv")
#     fit_stdf = os.path.join("Models", "FitParameters", "STD369.csv")
#     for fld in [sample_png, orientations_folder, save_pp_of, angle_folder, eval_SS_ShowAll, quality_reeval_z_th1, quality_reeval_z_th3, export_hists_folder, quality_reeval, quality_reeval_ro, all_crop_origami, quality_folder,
#                 fit_parameter_folder, quality_reeval_th02_3, quality_reeval_th01_3, quality_reeval_th01_1,
#                 all_crop_origami_resc, ss_labels, pp_sample, crop_origami, pp_crop_origami_npy,
#                 pp_crop_origami_png, ss_results, distance_results_all, evaluation_folder]:
#         if fld is None or fld is bool:
#             continue
#         os.makedirs(fld, exist_ok=True)
#     if labels_provided:
#         os.makedirs(label_crops)
#         os.makedirs(sample_labels_provided_yolo)
#         os.makedirs(sample_labels_provided_resc)
#
#
#
#     all_dists = os.path.join(evaluation_folder, "all_dists.csv")
#
#
#
#     # Copy SPM files to Resultfolder
#     shutil.copytree(folder, spm_folder, dirs_exist_ok=True)
#     if labels_provided:
#         shutil.copytree(labels, sample_labels_provided)
#
#     IMAGE_SIZE = 50
#     # Turn SPM to grayscale
#     spm_files = []
#     spm_foldernames = []
#
#     spm_files, spm_foldernames = get_folder_files(spm_folder, extensions=["spm", "sxm"])
#
#     # 1. Rename files to avoid errors with _ and .
#     try:
#         spm_files, spm_foldernames = rename_files(spm_files, spm_foldernames)
#     except FileExistsError:
#         pass
#
#     png_foldernames = []
#     for i in tqdm(range(len(spm_files)), desc="Reading SPM"):
#         f = spm_files[i]
#         fld = spm_foldernames[i]
#         # print(fld)
#         spmf = str(spm_folder).split("\\")[-1]
#         flderg = fld.split("\\")[-1]
#         assert len(spm_files) == len(spm_foldernames)
#         assert "\\" + spmf + "\\" in fld, fld + spmf
#         parts = str(fld).split("\\")
#         for i in range(len(parts)):
#             if parts[i] == spmf:
#                 flderg = "\\".join(parts[i + 1:])
#                 break
#         # print(flderg)
#         # input()
#         filename = os.path.join(fld, f)
#         resf = os.path.join(sample_png, flderg, ".".join(f.split(".")[:-1]))
#         try:
#             read_input(filename, resf)
#         except FileExistsError:
#             pass
#         png_foldernames.append(resf)
#
#     # 2. prepare for YOLO input
#     yaml_file, yolo_tempdict, dataset_path, yolo_pp_args = prepare_for_yolo(png_foldernames,
#                                                                             os.path.join(resfolder, "temp_yolo"),
#                                                                             pp_sample, save_pp_of=save_pp_of if crop_pp else None,
#                                                                             sample_png=sample_png)
#
#     if labels_provided:
#         rename_samplelabels(yolo_tempdict, provided_fld=sample_labels_provided, sample_png=save_pp_of if crop_pp else sample_png, resfld=sample_labels_provided_yolo)
#
#     # for key in yolo_tempdict.keys():
#     #     print(f"{key} --> {yolo_tempdict[key]}")
#     # input()
#
#     # 3. Perform YOLO Prediction
#     if not os.path.isdir(yolo_pred):
#         predict_yolo(yolo_model, yaml_file, conf_thrsh=yolo_conf_thrsh)
#         yolo_res = os.path.join("EvalChain", "EvalChain")
#         if len(os.listdir("EvalChain")) != 1:
#             yolo_res = os.path.join("EvalChain", f"EvalChain{len(os.listdir('EvalChain'))}")
#         shutil.move(yolo_res, yolo_pred)
#         print(yolo_res)
#         print(yolo_pred)
#     clear_yolo_dataset(dataset_path)
#
#     # 4. Crop images
#     drop_border = True
#     extend = 1.5
#     crp_tempdict, conf_dict, crpsize_dict = crop_images(yolo_tempdict, yolo_pred, crop_origami, sample_png if not crop_pp else save_pp_of,
#                                                         tempfolder2=all_crop_origami, drop_border=drop_border,
#                                                         extend=extend, crop_labels=labels_provided, prov_labels_folder=sample_labels_provided_yolo, label_crops=label_crops)
#
#
#     # 5. Preprocess cropped Images for SS
#     pp_origami_args = preprocess_origami(all_crop_origami, pp_crop_origami_npy, pp_crop_origami_png, threads=threads,
#                                          img_size=IMAGE_SIZE)
#
#
#     # if use_UNet:
#     # 6. Perform Semantic Segmentation on images
#     get_semantic_seg(ss_model, pp_crop_origami_npy, ss_results, threshold=ss_thrsh, bilinear=bilinear_model)
#     # 7. get Distances from Images + SS Masks
#     extract_gsc_ss(ss_results, ss_labels)
#     if not use_UNet:
#         rescale_provided_labels(inpt=label_crops, outpt=sample_labels_provided_resc, size=IMAGE_SIZE)
#
#     rescale_orig_imgs(all_crop_origami, all_crop_origami_resc, IMAGE_SIZE)
#
#     used_labels = ss_labels if use_UNet else sample_labels_provided_resc
#
#
#     distances, thetas = get_distances(all_crop_origami_resc, used_labels, distance_results_all, fit_parameter_folder,
#                                       threads=THREADS, export_hists=None,
#                                       pp_img_folder=pp_crop_origami_npy, eval_ss_SHOWALL=None)
#
#     while distances == False:
#         print("Recieved Distances ", distances, "Retyring SemSeg")
#         yaml_file, yolo_tempdict, dataset_path, yolo_pp_args = prepare_for_yolo(png_foldernames,
#                                                                                 os.path.join(resfolder, "temp_yolo"),
#                                                                                 pp_sample, save_pp_of=save_pp_of if crop_pp else None,
#                                                                             sample_png=sample_png)
#
#         # 3. Perform YOLO Prediction
#         if not os.path.isdir(yolo_pred):
#             predict_yolo(yolo_model, yaml_file, conf_thrsh=yolo_conf_thrsh)
#             yolo_res = os.path.join("EvalChain", "EvalChain")
#             if len(os.listdir("EvalChain")) != 1:
#                 yolo_res = os.path.join("EvalChain", f"EvalChain{len(os.listdir('EvalChain'))}")
#             shutil.move(yolo_res, yolo_pred)
#             print(yolo_res)
#             print(yolo_pred)
#         clear_yolo_dataset(dataset_path)
#
#         # 4. Crop images
#         drop_border = True
#         extend = 1.5
#         crp_tempdict, conf_dict, crpsize_dict = crop_images(yolo_tempdict, yolo_pred, crop_origami, sample_png if not crop_pp else save_pp_of,
#                                                             tempfolder2=all_crop_origami,
#                                                             drop_border=drop_border, extend=extend, sample_png_norm=sample_png)
#
#         # 5. Preprocess cropped Images for SS
#         pp_origami_args = preprocess_origami(all_crop_origami, pp_crop_origami_npy, pp_crop_origami_png, threads=threads,
#                                              img_size=IMAGE_SIZE)
#
#         get_semantic_seg(ss_model, pp_crop_origami_npy, ss_results, threshold=ss_thrsh, bilinear=bilinear_model)
#         # 7. get Distances from Images + SS Masks
#         extract_gsc_ss(ss_results, ss_labels)
#         if not use_UNet:
#             rescale_provided_labels(inpt=label_crops, outpt=sample_labels_provided_resc, size=IMAGE_SIZE)
#
#         used_labels = ss_labels if use_UNet else sample_labels_provided_resc
#
#         rescale_orig_imgs(all_crop_origami, all_crop_origami_resc, IMAGE_SIZE)
#         distances, thetas = get_distances(all_crop_origami_resc, used_labels, distance_results_all, fit_parameter_folder,
#                                           threads=THREADS, export_hists=None,
#                                           pp_img_folder=None)
#
#     with open(yolo_tempdict_file, "w") as f:
#         for k in yolo_tempdict.keys():
#             f.write(f"{k};{yolo_tempdict[k]}\n")
#
#     dists_x = []
#     dists_y = []
#
#     for i in range(len(distances)):
#         dists_x.append(abs(np.cos(thetas[i])) * distances[i])
#         dists_y.append(abs(np.cos(thetas[i])) * distances[i])
#
#     make_csv(all_crop_origami_resc, distances, thetas, all_dists)
#     dist_theta_dict = reorganize_files(all_crop_origami_resc, distance_results_all, all_dists, crp_tempdict,
#                                        resfile=tempdict_file, img_size=IMAGE_SIZE, filter_cropsize=True,
#                                        filter_thrsh=crpsize_stds, saveplot=False)
#
#     copy_yolo_pred(yolo_tempdict=yolo_tempdict, yolo_basefolder=yolo_pred, crop_folder=crop_origami,
#                    sample_png=sample_png)
#     if crop_pp:
#         copy_yolo_pred(yolo_tempdict=yolo_tempdict, yolo_basefolder=yolo_pred, crop_folder=crop_origami,
#                    sample_png=save_pp_of)
#     avg, med, stabw, ok = export_distances(crop_origami, evaluation_folder)
#
#     distances_nm_all = []
#     thetas_all = []
#     for k in dist_theta_dict.keys():
#         d = dist_theta_dict[k][0]
#         t = dist_theta_dict[k][1]
#         distances_nm_all.append(d)
#         thetas_all.append(t)
#
#     evaluate_angles(distances_nm_all, thetas_all, angle_folder, angle_thrsh=angle_thrsh)
#     analyze_orientations(crop_origami=crop_origami, resf=orientations_folder, angle_thrsh=angle_thrsh)
#
#     print(
#         "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm,"
#         " Evaluated Origami: {:.1f}% of {}".format(avg,med,stabw,100 * ok,len(crp_tempdict.keys())))
#     modify_csv(os.path.join(evaluation_folder, "Results_norm.csv"), delim=";")
#     modify_csv(os.path.join(evaluation_folder, "Results_10pct.csv"), delim=";")
#     modify_csv(os.path.join(evaluation_folder, "Results_complete.csv"), delim=";")
#
#     visualize_results(evaluation_folder, norm_fp="Results_norm.csv", cmpl_fp="Results_complete.csv",
#                       p10_fp="Results_10pct.csv", no_imgs=len(crp_tempdict.keys()), print_only_norm=print_only_norm)
#
#     gauss_acc, marker_alignment = gauss_accuracy(fit_parameter_folder)
#     size_acc, count_acc = count_compare_labels(used_labels, threads=THREADS)
#     rect_acc = rectangulartiy(used_labels, THREADS, resolution=2, sparse=0.5, theta_res=9, show=False)
#     export_quality(dist_theta_dict, rect_acc, gauss_acc, size_acc, count_acc, conf_dict, crpsize_dict, marker_alignment,
#                    quality_folder,
#                    q_indcs_file, method=quality_method, modelfolder=quality_modelfolder, fit_qfile=fit_qfile,
#                    fit_stdf=fit_stdf)
#
#     avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval, dst_folder=distance_results_all,
#                                           angle_thrsh=angle_thrsh, plot_hists=False, rem_outer=0.0)
#
#    #  avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval_th01_1, dst_folder=distance_results_all,
#    #                                        angle_thrsh=angle_thrsh, plot_hists=False, rem_outer=None, dist_th=0.1,
#    #                                        dist_num=1)
# #
#    #  avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval_th01_3, dst_folder=distance_results_all,
#    #                                        angle_thrsh=angle_thrsh, plot_hists=False, rem_outer=None, dist_th=0.1,
#    #                                        dist_num=3)
# #
#    #  avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval_th02_3, dst_folder=distance_results_all,
#    #                                        angle_thrsh=angle_thrsh, plot_hists=False, rem_outer=None, dist_th=0.2,
#    #                                        dist_num=3)
# #
#    #  avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval_ro, dst_folder=distance_results_all,
#    #                                        angle_thrsh=angle_thrsh, plot_hists=False, rem_outer=rem_outer)
# #
#
#
#     avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval_z_th3, dst_folder=distance_results_all,
#                                           angle_thrsh=angle_thrsh, plot_hists=False, zero_th=0.03)
#
#     print(f"Result: D={avg:.2f}nm, std: {stdmwt:.2f}nm")
#     with open(settings_file, "w") as f:
#         f.write(f"Input: {folder}\n")
#         f.write(f"Output: {resfolder}\n")
#         f.write(f"spm_folder: {spm_folder}\n")
#         f.write(f"sample_png: {sample_png}\n")
#         f.write(f"pp_sample: {pp_sample}\n")
#         f.write(f"yolo_pred: {yolo_pred}\n")
#         f.write(f"yolo_model: {yolo_model}\n")
#         f.write(f"crop_origami: {crop_origami}\n")
#         f.write(f"all_crop_origami: {all_crop_origami}\n")
#         f.write(f"all_crop_origami_resc: {all_crop_origami_resc}\n")
#         f.write(f"pp_crop_origami_npy: {pp_crop_origami_npy}\n")
#         f.write(f"pp_crop_origami_png: {pp_crop_origami_png}\n")
#         f.write(f"ss_results: {ss_results}\n")
#         f.write(f"ss_labels: {ss_labels}\n")
#         f.write(f"ss_model: {ss_model}\n")
#         f.write(f"distance_results_all: {distance_results_all}\n")
#         f.write(f"evaluation_folder: {evaluation_folder}\n")
#         f.write(f"settings_file: {settings_file}\n")
#         f.write(f"all_dists: {all_dists}\n")
#         f.write(f"yolo_conf_thrsh: {yolo_conf_thrsh}\n")
#         f.write(f"IMAGE_SIZE: {IMAGE_SIZE}\n")
#         f.write(f"yaml_file: {yaml_file}\n")
#         f.write(f"dataset_path: {dataset_path}\n")
#         f.write(f"yolo_pp_args: {yolo_pp_args}\n")
#         f.write(f"drop_border: {drop_border}\n")
#         f.write(f"pp_origami_args: {pp_origami_args}\n")
#         f.write(f"10pRes: {folder}\n")
#         f.write(f"Q-method: {quality_method}\n")
#         f.write(f"Q-Model: {quality_modelfolder}\n")
#         f.write(f"Evaluated origami: {len(crp_tempdict.keys())}\n")
#         f.write(f"Angle Threshold: {angle_thrsh}\n")
#         f.write(f"SS Threshold: {ss_thrsh}\n")
#         f.write(f"FilterCropsize STDs: {crpsize_stds}\n")
#         f.write(f"FitQFile: {fit_qfile}\n")
#         f.write(f"Fit STD File: {fit_stdf}\n")
#         f.write(f"Remove outer x percent: {rem_outer}\n")
#         f.write(f"Crop from PP: {crop_pp}\n")
#
#
#         f.write(f"Quality Mode: {QualityMeasurements.MODE}\n")
#         f.write(
#             "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm,"
#             " Evaluated Origami: {:.1f}%\n".format(avg,med,stabw,100 * ok))


def evaluate_dataset_class(folder, resfolder):
    """
    Complete Evaluation of a dataset from bare images to distances
    """
    spm_folder = os.path.join(resfolder, "spm")
    sample_png = os.path.join(resfolder, "sample_png")
    pp_sample = os.path.join(resfolder, "pp_sample")
    yolo_pred = os.path.join(resfolder, "yolo_prediction")
    yolo_model = os.path.join("Models", "YOLO", "Mix4k256pt.pt")
    crop_origami = os.path.join(resfolder, "crop_origami")
    all_crop_origami = os.path.join(resfolder, "crop_origami_allpng")
    result_folder = os.path.join(resfolder, "class_results")
    temp_folder = os.path.join(resfolder, "temp")
    crop_res_fld = os.path.join(resfolder, "SingleOrigamiRes")

    distance_results_all = os.path.join(resfolder, "distance_results_all")
    evaluation_folder = os.path.join(resfolder, "Eval_Results")
    settings_file = os.path.join(evaluation_folder, "settings.txt")

    yolo_conf_thrsh = 0.88

    for fld in [sample_png, crop_res_fld, result_folder, temp_folder, all_crop_origami, pp_sample, crop_origami,
                distance_results_all, evaluation_folder]:
        os.makedirs(fld, exist_ok=True)

    all_dists = os.path.join(evaluation_folder, "all_dists.csv")

    # Copy SPM files to Resultfolder
    shutil.copytree(folder, spm_folder, dirs_exist_ok=True)
    IMAGE_SIZE = 50
    # Turn SPM to grayscale
    spm_files = []
    spm_foldernames = []

    spm_files, spm_foldernames = get_folder_files(spm_folder)

    # 1. Rename files to avoid errors with _ and .
    try:
        spm_files, spm_foldernames = rename_files(spm_files, spm_foldernames)
    except FileExistsError:
        pass
    png_foldernames = []
    for i in tqdm(range(len(spm_files)), desc="Reading SPM"):
        f = spm_files[i]
        fld = spm_foldernames[i]
        # print(fld)
        spmf = str(spm_folder).split("\\")[-1]
        flderg = fld.split("\\")[-1]
        assert len(spm_files) == len(spm_foldernames)
        assert "\\" + spmf + "\\" in fld, fld + spmf
        parts = str(fld).split("\\")
        for k in range(len(parts)):
            if parts[k] == spmf:
                flderg = "\\".join(parts[k + 1:])
                break
        # print(flderg)
        # input()
        filename = os.path.join(fld, f)
        resf = os.path.join(sample_png, flderg, ".".join(f.split(".")[:-1]))
        try:
            read_input(filename, resf)
        except FileExistsError:
            pass
        png_foldernames.append(resf)

    # 2. prepare for YOLO input
    yaml_file, yolo_tempdict, dataset_path, yolo_pp_args = prepare_for_yolo(png_foldernames,
                                                                            os.path.join(resfolder, "temp_yolo"),
                                                                            pp_sample)

    # 3. Perform YOLO Prediction
    if not os.path.isdir(yolo_pred):
        predict_yolo(yolo_model, yaml_file, conf_thrsh=yolo_conf_thrsh)
        yolo_res = os.path.join("EvalChain", "EvalChain")
        if len(os.listdir("EvalChain")) != 1:
            yolo_res = os.path.join("EvalChain", f"EvalChain{len(os.listdir('EvalChain'))}")
        shutil.move(yolo_res, yolo_pred)
        print(yolo_res)
        print(yolo_pred)
    clear_yolo_dataset(dataset_path)

    # 4. Crop images
    drop_border = True
    extend = 1.2
    crp_tempdict, conf_dict, crpsize_dict = crop_images(yolo_tempdict, yolo_pred, crop_origami, sample_png,
                                                        tempfolder2=all_crop_origami,
                                                        drop_border=drop_border, extend=extend)

    file_folder = all_crop_origami
    res_f = open(os.path.join(result_folder, "results.csv"), "a")
    distances = []
    for f in tqdm(os.listdir(file_folder), desc="Analyzing images"):
        if f.endswith("csv"):
            continue
        dir = os.path.join(result_folder, f.split(".")[0])
        os.makedirs(dir, exist_ok=True)

        f1 = os.path.join(result_folder, f.split(".")[0], "1_ReadImg.png")
        f2 = os.path.join(result_folder, f.split(".")[0], "2_Canny.png")
        f3 = os.path.join(result_folder, f.split(".")[0], "3_Hough.png")
        f4 = os.path.join(result_folder, f.split(".")[0], "4_Filtered.png")
        f5 = os.path.join(result_folder, f.split(".")[0], "5_ScanNorm.png")
        f6 = os.path.join(result_folder, f.split(".")[0], "6_ScanBlur.png")
        f7 = os.path.join(result_folder, f.split(".")[0], "7_Distance.png")
        f8 = os.path.join(result_folder, f.split(".")[0], "8_Result.png")
        err = os.path.join(result_folder, f.split(".")[0], "99_Err.txt")
        file = os.path.join(file_folder, f)
        try:
            pt = pretransform_img(file, temp_folder)
            pt = npy2img(pt)
            img_blr, edg = canny(pt, f1, f2, show_all=SHOW_ALL)
            lines = hough(edg, f3=f3, f4=f4, no_lines=16, show_all=SHOW_ALL)
            base_auf, base_vec, scan_vec = find_direction_range(lines, img_blr, name=f.split(".")[0], show_all=SHOW_ALL)
            img_norm = pt
            # os.makedirs(os.path.join(result_folder, f.split(".")[0], "scanning"), exist_ok=True)
            averages, averages_blur = scan_origami(img_norm, img_blr, base_auf, base_vec, scan_vec, f5=f5, f6=f6,
                                                   show_all=SHOW_ALL, visualize_folder=None, mode='median')
            distance = find_maxima(averages, scan_vec, base_auf=base_auf,
                                   base_vec=base_vec, file=file, f7=f7, f8=f8, show_all=SHOW_ALL, mode='convolve',
                                   upscale=IMAGE_SIZE)
            distances.append(distance)
            res_f.write(f + ";{:.6f}\n".format(distance))
        except Exception as e:
            distances.append(-1)
            if str(e).startswith("list"):
                raise e
            with open(err, "w") as fexc:
                traceback.print_exc(file=fexc)
                # raise e
            res_f.write(f + ";-1\n")

    make_csv(all_crop_origami, distances, None, all_dists)
    reorganize_files_class(all_crop_origami, result_folder, all_dists, crp_tempdict, crop_origami, crop_res_fld,
                           img_size=IMAGE_SIZE)
    avg, med, stabw, ok = export_distances(crop_origami, evaluation_folder, filter_dist=None)
    print(
        "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm,"
        " Evaluated Origami: {:.1f}% of {}".format(avg,med,stabw,100 * ok,len(crp_tempdict.keys())))
    modify_csv(os.path.join(evaluation_folder, "Results_norm.csv"), delim=";")
    modify_csv(os.path.join(evaluation_folder, "Results_10pct.csv"), delim=";")
    modify_csv(os.path.join(evaluation_folder, "Results_complete.csv"), delim=";")

    visualize_results(evaluation_folder, norm_fp="Results_norm.csv", cmpl_fp="Results_complete.csv",
                      p10_fp="Results_10pct.csv", no_imgs=len(crp_tempdict.keys()))

    with open(settings_file, "w") as f:
        f.write(f"Input: {folder}\n")
        f.write(f"Output: {resfolder}\n")
        f.write(f"spm_folder: {spm_folder}\n")
        f.write(f"sample_png: {sample_png}\n")
        f.write(f"pp_sample: {pp_sample}\n")
        f.write(f"yolo_pred: {yolo_pred}\n")
        f.write(f"yolo_model: {yolo_model}\n")
        f.write(f"crop_origami: {crop_origami}\n")
        f.write(f"all_crop_origami: {all_crop_origami}\n")
        f.write(f"distance_results_all: {distance_results_all}\n")
        f.write(f"evaluation_folder: {evaluation_folder}\n")
        f.write(f"settings_file: {settings_file}\n")
        f.write(f"all_dists: {all_dists}\n")
        f.write(f"yolo_conf_thrsh: {yolo_conf_thrsh}\n")
        f.write(f"IMAGE_SIZE: {IMAGE_SIZE}\n")
        f.write(f"yaml_file: {yaml_file}\n")
        f.write(f"dataset_path: {dataset_path}\n")
        f.write(f"yolo_pp_args: {yolo_pp_args}\n")
        f.write(f"drop_border: {drop_border}\n")
        f.write(f"10pRes: {folder}\n")
        f.write(f"Evaluated origami: {len(crp_tempdict.keys())}\n")
        f.write(
            "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm,"
            " Evaluated Origami: {:.1f}%\n".format(avg, med,stabw,100 * ok))


def manual_segmentation(input, output, imgs, label_folder=r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\SynthCrop\Label'):
    plt.switch_backend('TkAgg')

    def binarize(val):
        if val > 190:
            return 2
        elif 64 < val <= 190:
            return 1
        return 0
    binarize = np.vectorize(binarize)
    lblimgs = os.path.join(label_folder, "bild")
    lbllbls = os.path.join(label_folder, 'data', 'PNG')
    lims = [np.array(Image.open(os.path.join(lblimgs, x)))[:, :, 0].astype(int) for x in os.listdir(lblimgs)]
    lls =  [binarize(np.array(Image.open(os.path.join(lbllbls, x)))[:, :, 0]).astype(int) for x in os.listdir(lbllbls)]

    # print(lims[0])
    # plt.imshow(lims[0])
    # plt.show()
    assert len(lls) == len(lims)
    pairs =[]
    for i in range(len(lls)):
        pairs.append((lims[i].shape[0], lims[i], lls[i]))

    for file in tqdm(os.listdir(input), desc="getting Segmentation"):
        image = Image.open(os.path.join(input, file))
        arr = np.array(image)[:, :, 0]
        # plt.imshow(arr)
        # plt.title(file)
        # plt.show()

        ss = np.zeros((imgs, imgs)).astype(int)

        for i in tqdm(range(len(pairs)), disable=True):
            if pairs[i][0] != arr.shape[0]:
                continue

            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(arr)
            # axs[1].imshow(pairs[i][1])

            err = np.sqrt(np.average(np.square(arr - pairs[i][1])))
            #plt.title(err)
            #plt.show()

            if err == 0:
                ss = pairs[i][2]
                ss = binarize(np.array(cv2.resize(127.5 * ss, dsize=(imgs, imgs), interpolation=cv2.INTER_LINEAR))).astype(int)
                # plt.imshow(ss)
                # plt.title("Label")
                # plt.show()
                break

        plt.imsave(os.path.join(output, file), ss, vmax=2, cmap='gray')

    os.makedirs(output, exist_ok=True)



def evaluate_dataset_xy_allargs(**kwargs):
    """
    Complete Evaluation of a dataset from bare images to distances
    """
    wandb.init(mode="disabled")
    # folder, resfolder, qual_mode="STD", threads=None, angle_thrsh=np.pi/6, labels=None, use_UNet=True
    assert "folder" in kwargs.keys()
    assert "resfolder" in kwargs.keys()
    folder = kwargs["folder"]
    resfolder = kwargs["resfolder"]
    qual_mode = kwargs["quality_mode"] if "quality_mode" in kwargs.keys() else "FIT"
    threads = kwargs["threads"] if "threads" in kwargs.keys() else os.cpu_count()
    labels = kwargs["labels"] if "labels" in kwargs.keys() else None
    angle_thrsh = kwargs["angle_thrsh"] if "angle_thrsh" in kwargs.keys() else np.pi/6
    use_UNet = kwargs["use_UNet"] if "use_UNet" in kwargs.keys() else True
    yolo_model = kwargs["yolo_model"] if "yolo_model" in kwargs.keys() else r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\YOLO\Mix4k256_newpt2.pt"
    ss_model = kwargs["ss_model"] if "ss_model" in kwargs.keys() else None
    # ss_model = kwargs["ss_model"] if "ss_model" in kwargs.keys() else r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Upscaled64\Train\checkpoints_NewTest\checkpoint_epoch10.pth"
    print("SS Model ---> ", ss_model)
    fit_qfile = kwargs["fit_params"] if "fit_params" in kwargs.keys() else os.path.join("Models", "FitParameters", "FIT368.csv")
    yolo_conf_thrsh = kwargs["yolo_conf_thrsh"] if "yolo_conf_thrsh" in kwargs.keys() else 0.70  # 0.82
    ss_thrsh = kwargs["ss_thrsh"] if "ss_thrsh" in kwargs.keys() else 0.4
    crpsize_stds = kwargs["crpsize_stds"] if "crpsize_stds" in kwargs.keys() else 3  # 3
    filter_cropsize = kwargs["filter_cropsize"] if "filter_cropsize" in kwargs.keys() else True
    bilinear_model = kwargs["bilinear_model"] if "bilinear_model" in kwargs.keys() else False
    perform_angleAnalysis = kwargs["perform_angleAnalysis"] if "perform_angleAnalysis" in kwargs.keys() else True
    perform_qualityAnalysis = kwargs["perform_qualityAnalysis"] if "perform_qualityAnalysis" in kwargs.keys() else True
    yolo_sr = kwargs["yolo_sr"] if "yolo_sr" in kwargs.keys() else None
    ss_sr = kwargs["ss_sr"] if "ss_sr" in kwargs.keys() else None
    sample_sr_scale = kwargs["sample_sr_scale"] if "sample_sr_scale" in kwargs.keys() else 1
    ss_sr_scale = kwargs["ss_sr_scale"] if "ss_sr_scale" in kwargs.keys() else 1
    abort_od = kwargs["abort_od"] if "abort_od" in kwargs.keys() else False

    possible_params = ["folder", "resfolder", "quality_mode", "threads", "labels", "angle_thrsh", "use_UNet",
                       "yolo_model", "ss_model", "fit_params", "yolo_conf_thrsh", "ss_thrsh", "crpsize_stds",
                       "filter_cropsize", "bilinear_model", "perform_angleAnalysis", "perform_qualityAnalysis", "abort_od", 'ss_sr', 'yolo_sr', 'sample_sr_scale', 'ss_sr_scale']

    print("Provided Parameters:")
    for key in kwargs.keys():
        print(f"{key} --> {kwargs[key]}")
        if key not in possible_params:
            raise Exception(f"Unknown Parameter: {key}. Possible Parameters are {possible_params}")


    THREADS = threads
    labels_provided = labels is not None
    assert os.path.isdir(folder)
    print("Selected Input Folder", folder)
    # Make Directories in Resultfolder
    spm_folder = os.path.join(resfolder, "spm")
    sample_png = os.path.join(resfolder, "sample_png")
    save_pp_of = os.path.join(resfolder, "sample_png_pp")
    pp_sample = os.path.join(resfolder, "pp_sample")
    sample_denoise = os.path.join(resfolder, "sample_denoise")
    yolo_pred = os.path.join(resfolder, "yolo_prediction")
    crop_origami = os.path.join(resfolder, "crop_origami")
    all_crop_origami = os.path.join(resfolder, "crop_origami_allpng")
    all_crop_origami_resc = os.path.join(resfolder, "crop_origami_allpng_resc")
    pp_crop_origami_npy = os.path.join(resfolder, "pp_crop_origami_npy")
    pp_crop_origami_png = os.path.join(resfolder, "pp_crop_origami_png")
    ss_results = os.path.join(resfolder, "ss_results")
    ss_labels = os.path.join(resfolder, "ss_labels")
    ss_denoise = os.path.join(resfolder, "ss_denoise")

    angle_folder = os.path.join(resfolder, "AngleEvaluation") if perform_angleAnalysis else None
    orientations_folder = os.path.join(resfolder, "Orientations") if perform_angleAnalysis else None

    label_crops = os.path.join(resfolder, "provided_labels")
    sample_labels_provided = os.path.join(resfolder, "sample_labels")
    sample_labels_provided_yolo = os.path.join(resfolder, "provided_labels_yolo")
    sample_labels_provided_resc = os.path.join(resfolder, "provided_labels_resc")

    # ss_model = os.path.join("Models", "SS", "ModelSize50.pth")
    # ss_model = os.path.join("Models", "SS", "ModelNewPP2410.pth")
    # bilinear_model = True
    # ss_model = "SS_DNA_Train\\Small50px_5k\\Train\\checkpoints\\checkpoint_epoch10.pth" # cpold/4

    distance_results_all = os.path.join(resfolder, "distance_results_all")
    quality_folder = os.path.join(resfolder, "quality") if perform_qualityAnalysis else None
    fit_parameter_folder = os.path.join(resfolder, "fit_parameters")
    evaluation_folder = os.path.join(resfolder, "Eval_Results")
    tempdict_file = os.path.join(evaluation_folder, "tempdict.csv")
    yolo_tempdict_file = os.path.join(evaluation_folder, "tempdict_yolo.csv")
    settings_file = os.path.join(evaluation_folder, "settings.txt")
    q_indcs_file = os.path.join(quality_folder, "quality_indcs.csv")  if perform_qualityAnalysis else None
    quality_reeval = os.path.join(resfolder, "quality_reeval")  if perform_qualityAnalysis else None
    quality_reeval_ro = os.path.join(resfolder, "quality_reeval_ro")  if perform_qualityAnalysis else None
    quality_reeval_th01_1 = None#  os.path.join(resfolder, "quality_reeval_th01_1")
    quality_reeval_th01_3 = None#  os.path.join(resfolder, "quality_reeval_01_3")
    quality_reeval_th02_3 = None#  os.path.join(resfolder, "quality_reeval_02_3")
    quality_reeval_z_th1 = None #  os.path.join(resfolder, "quality_reeval_zth1")
    quality_reeval_z_th3 = os.path.join(resfolder, "quality_reeval_zth3")  if perform_qualityAnalysis else None
    eval_SS_ShowAll = os.path.join(resfolder, "EvalSSDoku")
    # eval_SS_ShowAll = None

    if os.path.isdir(resfolder) and len(os.listdir(resfolder)) > 0:
        raise Exception(f"Result Folder {resfolder} already exists and is not empty")

    quality_method = "Class"
    quality_modelfolder = os.path.join("Models", "ErrorNN")
    print_only_norm = True
    rem_outer=0
    crop_pp=True
    export_hists_folder = os.path.join(resfolder, "Histograms")
    quality_mode = qual_mode
    QualityMeasurements.MODE = quality_mode
    fit_stdf = os.path.join("Models", "FitParameters", "STD369.csv")
    for fld in [sample_png, orientations_folder, save_pp_of, angle_folder, eval_SS_ShowAll, quality_reeval_z_th1, quality_reeval_z_th3, export_hists_folder, quality_reeval, quality_reeval_ro, all_crop_origami, quality_folder,
                fit_parameter_folder, quality_reeval_th02_3, quality_reeval_th01_3, quality_reeval_th01_1,
                all_crop_origami_resc, ss_labels, pp_sample, crop_origami, pp_crop_origami_npy,
                pp_crop_origami_png, ss_results, distance_results_all, evaluation_folder, sample_denoise, ss_denoise]:
        if fld is None or fld is bool:
            continue
        os.makedirs(fld, exist_ok=True)
    if labels_provided:
        os.makedirs(label_crops)
        os.makedirs(sample_labels_provided_yolo)
        os.makedirs(sample_labels_provided_resc)





    all_dists = os.path.join(evaluation_folder, "all_dists.csv")



    # Copy SPM files to Resultfolder
    shutil.copytree(folder, spm_folder, dirs_exist_ok=True)
    if labels_provided:
        shutil.copytree(labels, sample_labels_provided)

    IMAGE_SIZE = 50 if OLD_MODEL else 64

    # Turn SPM to grayscale
    spm_files = []
    spm_foldernames = []
    EXTENTIONS = ["spm", "sxm"]
    move = False
    for elem in os.listdir(spm_folder):
        if os.path.isfile(os.path.join(spm_folder, elem)):
            if elem.split(".")[-1] in EXTENTIONS:
                move = True

    mov_files = [os.path.join(spm_folder, elem) for elem in os.listdir(spm_folder)]
    sbpths = [elem for elem in os.listdir(spm_folder)]
    if move:
        newf = os.path.join(spm_folder, 'files')
        os.makedirs(newf)
        for i, f in enumerate(mov_files):
            # if os.path.isfile(f):
            shutil.move(f, os.path.join(newf, sbpths[i]))
            # else:
             #    shutil.movet()


    spm_files, spm_foldernames = get_folder_files(spm_folder, extensions=EXTENTIONS)



    # 1. Rename files to avoid errors with _ and .
    try:
        spm_files, spm_foldernames = rename_files(spm_files, spm_foldernames)
    except FileExistsError:
        pass

    png_foldernames = []
    for i in tqdm(range(len(spm_files)), desc="Reading SPM"):
        f = spm_files[i]
        fld = spm_foldernames[i]
        # print(fld)
        spmf = str(spm_folder).split("\\")[-1]
        flderg = fld.split("\\")[-1]
        assert len(spm_files) == len(spm_foldernames)
        assert "\\" + spmf + "\\" in fld, fld + spmf
        parts = str(fld).split("\\")
        for i in range(len(parts)):
            if parts[i] == spmf:
                flderg = "\\".join(parts[i + 1:])
                break
        # print(flderg)
        # input()
        filename = os.path.join(fld, f)
        resf = os.path.join(sample_png, flderg, ".".join(f.split(".")[:-1]))
        try:
            read_input(filename, resf)
        except FileExistsError:
            pass
        png_foldernames.append(resf)

    # 2. prepare for YOLO input
    yaml_file, yolo_tempdict, dataset_path, yolo_pp_args = prepare_for_yolo(filefolders=png_foldernames,
                                                                            tempfolder=os.path.join(resfolder, "temp_yolo"),
                                                                            resultfolder=pp_sample,
                                                                            save_pp_of=save_pp_of if crop_pp else None,
                                                                            sample_png=sample_png,
                                                                            sr_model=yolo_sr,
                                                                            sr_folder=sample_denoise,
                                                                            sr_scale=sample_sr_scale)

    if labels_provided:
        rename_samplelabels(yolo_tempdict, provided_fld=sample_labels_provided, sample_png=save_pp_of if crop_pp else sample_png, resfld=sample_labels_provided_yolo)

    # for key in yolo_tempdict.keys():
    #     print(f"{key} --> {yolo_tempdict[key]}")
    # input()

    # 3. Perform YOLO Prediction
    if not os.path.isdir(yolo_pred):
        predict_yolo(yolo_model, yaml_file, conf_thrsh=yolo_conf_thrsh)
        yolo_res = os.path.join("EvalChain", "EvalChain")
        if len(os.listdir("EvalChain")) != 1:
            yolo_res = os.path.join("EvalChain", f"EvalChain{len(os.listdir('EvalChain'))}")
        shutil.move(yolo_res, yolo_pred)
        print(yolo_res)
        print(yolo_pred)
    clear_yolo_dataset(dataset_path)

    if abort_od:
        print("Aborting due to abort_od=True")
        return -1

    # 4. Crop images
    drop_border = True
    extend = 1.5

    sxm_savefld = os.path.join(resfolder, "crop_sxms")
    crp_tempdict, conf_dict, crpsize_dict = crop_images(yolo_tempdict, yolo_pred, crop_origami, sample_png if not crop_pp else save_pp_of,
                                                        tempfolder2=all_crop_origami, drop_border=drop_border,
                                                        extend=extend, crop_labels=labels_provided,
                                                        prov_labels_folder=sample_labels_provided_yolo,
                                                        label_crops=label_crops, sxm_folder=sxm_savefld)




    # 5. Preprocess cropped Images for SS
    pp_origami_args = preprocess_origami(all_crop_origami, pp_crop_origami_npy, pp_crop_origami_png, threads=threads,
                                         img_size=IMAGE_SIZE)

    if yolo_sr is not None:
        apply_sr(ss_sr, pp_crop_origami_npy, ss_denoise, scale=ss_sr_scale)

        pp_crop_origami_npy = os.path.join(ss_denoise, "PNGs", "mix_npy")



    # if use_UNet:
    # 6. Perform Semantic Segmentation on images
    if MANUAL_SEG:
        manual_segmentation(all_crop_origami, ss_labels, IMAGE_SIZE, MS_FOLDER)
    else:
        get_semantic_seg(ss_model, pp_crop_origami_npy, ss_results, threshold=ss_thrsh, bilinear=bilinear_model)

    # 7. get Distances from Images + SS Masks
        extract_gsc_ss(ss_results, ss_labels)

    if not use_UNet:
        rescale_provided_labels(inpt=label_crops, outpt=sample_labels_provided_resc, size=IMAGE_SIZE)

    rescale_orig_imgs(all_crop_origami, all_crop_origami_resc, IMAGE_SIZE)

    used_labels = ss_labels if use_UNet else sample_labels_provided_resc


    distances, thetas = get_distances(all_crop_origami_resc, used_labels, distance_results_all, fit_parameter_folder,
                                      threads=THREADS, export_hists=None,
                                      pp_img_folder=pp_crop_origami_npy, eval_ss_SHOWALL=None)

    while distances == False:
        print("Recieved Distances ", distances, "Retyring SemSeg")
        yaml_file, yolo_tempdict, dataset_path, yolo_pp_args = prepare_for_yolo(png_foldernames,
                                                                                os.path.join(resfolder, "temp_yolo"),
                                                                                pp_sample, save_pp_of=save_pp_of if crop_pp else None,
                                                                            sample_png=sample_png)

        # 3. Perform YOLO Prediction
        if not os.path.isdir(yolo_pred):
            predict_yolo(yolo_model, yaml_file, conf_thrsh=yolo_conf_thrsh)
            yolo_res = os.path.join("EvalChain", "EvalChain")
            if len(os.listdir("EvalChain")) != 1:
                yolo_res = os.path.join("EvalChain", f"EvalChain{len(os.listdir('EvalChain'))}")
            shutil.move(yolo_res, yolo_pred)
            print(yolo_res)
            print(yolo_pred)
        clear_yolo_dataset(dataset_path)

        # 4. Crop images
        drop_border = True
        extend = 1.5
        crp_tempdict, conf_dict, crpsize_dict = crop_images(yolo_tempdict, yolo_pred, crop_origami, sample_png if not crop_pp else save_pp_of,
                                                            tempfolder2=all_crop_origami,
                                                            drop_border=drop_border, extend=extend,
                                                            sample_png_norm=sample_png,  sxm_folder=sxm_savefld)

        # 5. Preprocess cropped Images for SS
        pp_origami_args = preprocess_origami(all_crop_origami, pp_crop_origami_npy, pp_crop_origami_png, threads=threads,
                                             img_size=IMAGE_SIZE)
        if MANUAL_SEG:
            assert 1 == 2
        get_semantic_seg(ss_model, pp_crop_origami_npy, ss_results, threshold=ss_thrsh, bilinear=bilinear_model)
        # 7. get Distances from Images + SS Masks
        extract_gsc_ss(ss_results, ss_labels)
        if not use_UNet:
            rescale_provided_labels(inpt=label_crops, outpt=sample_labels_provided_resc, size=IMAGE_SIZE)

        used_labels = ss_labels if use_UNet else sample_labels_provided_resc

        rescale_orig_imgs(all_crop_origami, all_crop_origami_resc, IMAGE_SIZE)
        distances, thetas = get_distances(all_crop_origami_resc, used_labels, distance_results_all, fit_parameter_folder,
                                          threads=THREADS, export_hists=None,
                                          pp_img_folder=None)

    with open(yolo_tempdict_file, "w") as f:
        for k in yolo_tempdict.keys():
            f.write(f"{k};{yolo_tempdict[k]}\n")

    dists_x = []
    dists_y = []

    for i in range(len(distances)):
        dists_x.append(abs(np.cos(thetas[i])) * distances[i])
        dists_y.append(abs(np.cos(thetas[i])) * distances[i])

    make_csv(all_crop_origami_resc, distances, thetas, all_dists)
    dist_theta_dict = reorganize_files(all_crop_origami_resc, distance_results_all, all_dists, crp_tempdict,
                                       resfile=tempdict_file, img_size=IMAGE_SIZE, filter_cropsize=filter_cropsize,
                                       filter_thrsh=crpsize_stds, saveplot=False)

    copy_yolo_pred(yolo_tempdict=yolo_tempdict, yolo_basefolder=yolo_pred, crop_folder=crop_origami,
                   sample_png=sample_png)
    if crop_pp:
        copy_yolo_pred(yolo_tempdict=yolo_tempdict, yolo_basefolder=yolo_pred, crop_folder=crop_origami,
                   sample_png=save_pp_of)
    avg, med, stabw, ok = export_distances(crop_origami, evaluation_folder)
    if perform_angleAnalysis:

        distances_nm_all = []
        thetas_all = []
        for k in dist_theta_dict.keys():
            d = dist_theta_dict[k][0]
            t = dist_theta_dict[k][1]
            distances_nm_all.append(d)
            thetas_all.append(t)

        evaluate_angles(distances_nm_all, thetas_all, angle_folder, angle_thrsh=angle_thrsh)
        analyze_orientations(crop_origami=crop_origami, resf=orientations_folder, angle_thrsh=angle_thrsh)

    print(
        "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm,"
        " Evaluated Origami: {:.1f}% of {}".format(avg,med,stabw,100 * ok,len(crp_tempdict.keys())))
    stdmwt = stabw / np.sqrt(len(crp_tempdict.keys()) + 1)
    modify_csv(os.path.join(evaluation_folder, "Results_norm.csv"), delim=";")
    modify_csv(os.path.join(evaluation_folder, "Results_10pct.csv"), delim=";")
    modify_csv(os.path.join(evaluation_folder, "Results_complete.csv"), delim=";")

    visualize_results(evaluation_folder, norm_fp="Results_norm.csv", cmpl_fp="Results_complete.csv",
                      p10_fp="Results_10pct.csv", no_imgs=len(crp_tempdict.keys()), print_only_norm=print_only_norm)

    if perform_qualityAnalysis:
        gauss_acc, marker_alignment = gauss_accuracy(fit_parameter_folder)
        size_acc, count_acc = count_compare_labels(used_labels, threads=THREADS)
        rect_acc = rectangulartiy(used_labels, THREADS, resolution=2, sparse=0.5, theta_res=9, show=False)
        export_quality(dist_theta_dict, rect_acc, gauss_acc, size_acc, count_acc, conf_dict, crpsize_dict, marker_alignment,
                       quality_folder,
                       q_indcs_file, method=quality_method, modelfolder=quality_modelfolder, fit_qfile=fit_qfile,
                       fit_stdf=fit_stdf)

        avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval, dst_folder=distance_results_all,
                                              angle_thrsh=angle_thrsh, plot_hists=False, rem_outer=0.0)


        # avg, stdmwt = reevaluate_with_quality(q_indcs_file, quality_reeval_z_th3, dst_folder=distance_results_all,
        #                                     angle_thrsh=angle_thrsh, plot_hists=False, zero_th=0.03)

        print(f"Result: D={avg:.2f}nm, std: {stdmwt:.2f}nm")
    else:
        print(f"Result: D={avg:.2f}nm")
    with open(settings_file, "w") as f:
        f.write(f"Input: {folder}\n")
        f.write(f"Output: {resfolder}\n")
        f.write(f"spm_folder: {spm_folder}\n")
        f.write(f"sample_png: {sample_png}\n")
        f.write(f"pp_sample: {pp_sample}\n")
        f.write(f"yolo_pred: {yolo_pred}\n")
        f.write(f"yolo_model: {yolo_model}\n")
        f.write(f"crop_origami: {crop_origami}\n")
        f.write(f"all_crop_origami: {all_crop_origami}\n")
        f.write(f"all_crop_origami_resc: {all_crop_origami_resc}\n")
        f.write(f"pp_crop_origami_npy: {pp_crop_origami_npy}\n")
        f.write(f"pp_crop_origami_png: {pp_crop_origami_png}\n")
        f.write(f"ss_results: {ss_results}\n")
        f.write(f"ss_labels: {ss_labels}\n")
        f.write(f"ss_model: {ss_model}\n")
        f.write(f"distance_results_all: {distance_results_all}\n")
        f.write(f"evaluation_folder: {evaluation_folder}\n")
        f.write(f"settings_file: {settings_file}\n")
        f.write(f"all_dists: {all_dists}\n")
        f.write(f"yolo_conf_thrsh: {yolo_conf_thrsh}\n")
        f.write(f"IMAGE_SIZE: {IMAGE_SIZE}\n")
        f.write(f"yaml_file: {yaml_file}\n")
        f.write(f"dataset_path: {dataset_path}\n")
        f.write(f"yolo_pp_args: {yolo_pp_args}\n")
        f.write(f"drop_border: {drop_border}\n")
        f.write(f"pp_origami_args: {pp_origami_args}\n")
        f.write(f"10pRes: {folder}\n")
        f.write(f"Q-method: {quality_method}\n")
        f.write(f"Q-Model: {quality_modelfolder}\n")
        f.write(f"Evaluated origami: {len(crp_tempdict.keys())}\n")
        f.write(f"Angle Threshold: {angle_thrsh}\n")
        f.write(f"SS Threshold: {ss_thrsh}\n")
        f.write(f"FilterCropsize: {filter_cropsize}\n")
        f.write(f"FilterCropsize STDs: {crpsize_stds}\n")
        f.write(f"FitQFile: {fit_qfile}\n")
        f.write(f"Fit STD File: {fit_stdf}\n")
        f.write(f"Remove outer x percent: {rem_outer}\n")
        f.write(f"Crop from PP: {crop_pp}\n")


        f.write(f"Quality Mode: {QualityMeasurements.MODE}\n")
        f.write(
            "Evaluation finished: Avg: {:.1f}nm, Med: {:.1f}nm, Std: {:.1f}nm,"
            " Evaluated Origami: {:.1f}%\n".format(avg,med,stabw,100 * ok))


    return  avg, stdmwt, quality_reeval if perform_qualityAnalysis else evaluation_folder


OLD_MODEL = False
SAVE_SXM_CROPS = False
MANUAL_SEG = False
CROP_LABELS = False
CROP_LABELFLD = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\Set161_EvalYOLO_63\data\PNG'
if __name__ == "__main__":

    # rectangulartiy("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Testfiles\\QualityRect", threads=4)
    # count_compare_labels("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try22\\ss_labels")
    # exit(0)
    os.system("wandb disabled")
    NO_WARNINGS = True
    start = time.perf_counter()

    print(torch.cuda.is_available())

    if NO_WARNINGS:
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"
    else:
        if not sys.warnoptions:
            warnings.simplefilter("default")
            os.environ["PYTHONWARNINGS"] = "default"

    start = time.perf_counter()
    THREADS = 14
    # inpt = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Test\\TestDS_Tiny"

    ds1 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\DS1"
    ds2 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\DS2"
    ds3 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\DS3"
    ds4 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\DS4"
    dscf42 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\CF42"
    dsIndiv = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\INDIV"
    dsGood = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Good"
    dsAIR = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\AirNew0212"
    dsFAST = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\X0212_SS"
    dsSLOW = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Y0212_SS"
    dsSYx = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Synthetic\\Noise3X"
    dsSYy = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Synthetic\\Noise3Y"
    dsP1 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Synthetic\\Parallel1"
    dsP2 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Synthetic\\Parallel2"
    dsAIR1012 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Air1012"
    inpt = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\DS"
    lbl_inpt = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\SynthMix2"
    labels = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\SynthMix2Labels"

    air2 = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\20122022"
    ds = r"D:\Dateien\KI_Speicher\EvalChainDS\Time\Set2"
    res = r"D:\Dateien\KI_Speicher\EvalChainDS\Time\ResultSet2_DirectDistance"
    qual_mode = "FIT"
    useUnet = True
    labels=None
    scale=1.0
    #res = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res_cont\\Try{}_{}_{}_UseU_{}".format(
    #len(os.listdir("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res_cont")), os.path.basename(ds), qual_mode, useUnet)

    yolo_conf=0.70

    spm_folder = r"D:\seifert\PycharmProjects\DNAmeasurement\datasets\INDIV"
    length = len(os.listdir(r'D:\seifert\PycharmProjects\DNAmeasurement\AIResults')) if os.path.isdir(
        r'D:\seifert\PycharmProjects\DNAmeasurement\AIResults') else 0
    result_folder = os.path.join(r'D:\seifert\PycharmProjects\DNAmeasurement\AIResults',
                                 f"Try_{length}_ClassOD_{os.path.basename(spm_folder)}_cnf{int(round(100*yolo_conf))}")
    # os.makedirs(result_folder, exist_ok=True)



    ds = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\NoBirka'
    # ds = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\Set161_EvalYOLO_63\sxm'
    # ds = r'C:\Users\seifert\PycharmProjects\STM_Simulation\bildordner\SingleDNA_V2\Set63_YOLO_short\sxm'
    # ds = r'C:\Users\seifert\Downloads\Testdata'
    # ds = r'C:\Users\seifert\PycharmProjects\STM_Simulation\bildordner\SingleDNA_V2\Set10\sxm'
    # ds = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\NoBirka'
    # ds = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\AirNewCF'
    # ds = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\AirSingle'

    dss = [r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\NoBirka',
           r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\Set161_EvalYOLO_63\sxm',
           r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\AirNewCF']

    sss = [r'C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\M0907_1129\checkpoint_epoch28.pth',
           r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\SynB4_70EP.pth",
           r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\SynB2_70EP.pth",
           r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\68nm_64px.pth",
           r'C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\M0824_1540\checkpoint_epoch28.pth'
           ]

    yolo_sr = r'D:\seifert\PycharmProjects\SwinDNA\swinir_sr_classical_patch64_x1_SampleLONG\models\70000_G.pth'
    ss_sr = r'D:\seifert\PycharmProjects\SwinDNA\swinir_sr_classical_patch64_x1_MoleculeLONG\models\200000_G.pth'

    for ds in dss:
        fld = r'D:\seifert\PycharmProjects\DNAmeasurement\SR\SwinIR_Include'
        os.makedirs(fld, exist_ok=True)
        res = os.path.join(fld, f"{len(os.listdir(fld))}_{os.path.basename(ds)}_{int(100*scale)}_{os.path.basename(yolo_sr).split('.')[0] if yolo_sr is not None else ''}_{os.path.basename(ss_sr).split('.')[0] if ss_sr is not None else ''}")
        os.makedirs(res, exist_ok=True)

        ss = sss[0]
        params = {"folder": ds,
                  "resfolder": res,
                  "quality_mode": qual_mode,
                  "threads": THREADS,
                  "labels": labels,
                  "use_UNet": useUnet,
                  "yolo_conf_thrsh": yolo_conf,
                  "ss_model": ss,
                  'bilinear_model': False,
                  "perform_angleAnalysis": False,
                  "perform_qualityAnalysis": False,
                  "filter_cropsize": False,
                  "yolo_sr": yolo_sr,
                  "ss_sr": ss_sr,
                  'sample_sr_scale': 1,
                  'ss_sr_scale':1
                  }
        evaluate_dataset_xy_allargs(**params)

        continue


        for ss in sss:
            res = os.path.join(r'D:\seifert\PycharmProjects\DNAmeasurement\TotalKomp', os.path.basename(ds), f"{os.path.basename(os.path.dirname(ss))}_{os.path.basename(ss).split('.')[0]}")
            try:
                if os.path.isfile(os.path.join(res, 'quality_reeval', 'Result.txt')):
                    continue
            except FileNotFoundError:
                continue
            os.makedirs(res, exist_ok=True)

            params = {"folder": ds,
                      "resfolder": res,
                      "quality_mode": qual_mode,
                      "threads": THREADS,
                      "labels": labels,
                      "use_UNet": useUnet,
                      "yolo_conf_thrsh": yolo_conf,
                      "ss_model": ss,
                      'bilinear_model': False,
                      "perform_angleAnalysis": False,
                      "perform_qualityAnalysis": True,
                      "filter_cropsize": False
                      }
            evaluate_dataset_xy_allargs(**params)

    assert  1 == 2




    # ss_model =  r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\F1Synth_0804_065010.pth"
    # ss_model =  r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\Synth0730_095058_40.pth"
    # ss_model =  r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\SynB4_70EP.pth"
    ss_model = r'C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\M0907_1129\checkpoint_epoch28.pth'
    # ss_model = r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\68nm_64px.pth"
    # ss_model = r'C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\0823_1434_MoreVar.pth'
    # ss_model = r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\68nm_50px.pth"

    # ss_model =  r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\SynB2_70EP.pth"
    # ss_model =  r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\Real0731_095142_100.pth"

    # ss_model = r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SS\NewModelSmallerMarkers.pth"
    # ss_model = r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\DiffTDTest\OldModelNewData2.pth"

    res = "D:\seifert\PycharmProjects\DNAmeasurement\Output\Try{}_{}_{}_UseU_{}_LaterTime_Conf{}_{}_{}_{}_{}".format(
        len(os.listdir("D:\seifert\PycharmProjects\DNAmeasurement\Output")), os.path.basename(ds),
        qual_mode, useUnet, int(100*yolo_conf),  os.path.basename(os.path.dirname(ss_model)) ,os.path.basename(ss_model.split(".")[0]), "MS" if MANUAL_SEG else "", "OM" if OLD_MODEL else "")
    # "OLD_MODEL")


    # spm_folder = r'D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Data\INDIV'
    # result_folder = r'D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Results\INDIV\normal'

    if len(sys.argv) > 1:
        spm_folder = sys.argv[1]
        result_folder = sys.argv[2]
        yolo_conf = float(sys.argv[3])

    params = {"folder": ds,
              "resfolder": res,
              "quality_mode": qual_mode,
              "threads": THREADS,
              "labels": labels,
              "use_UNet": useUnet,
              "yolo_conf_thrsh": yolo_conf,
              "ss_model": ss_model,
              'bilinear_model': False,
              "perform_angleAnalysis": False,
              "perform_qualityAnalysis": True,
              "filter_cropsize" : False
              }
    evaluate_dataset_xy_allargs(**params)

    # ds = dsP2
    # qual_mode = "FIT"
    # res = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try{}_{}_{}_ppSample".format(
    #     len(os.listdir("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res")), os.path.basename(ds), qual_mode)
    # evaluate_dataset_xy(ds, res, qual_mode=qual_mode)

    # ds = dsSYy
    # qual_mode = "FIT"
    # res = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try{}_{}_{}_wo55".format(
    #     len(os.listdir("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res")), os.path.basename(ds), qual_mode)
    # evaluate_dataset_xy(ds, res, qual_mode=qual_mode)

    # ds = dsSYy
    # qual_mode = "FIT"
    # res = "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try{}_{}_{}_YOLOppNewModel".format(
    #     len(os.listdir("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res")), os.path.basename(ds), qual_mode)
    # evaluate_dataset_xy(ds, res, qual_mode=qual_mode)
    print("Duration: {:.1f}s".format(time.perf_counter() - start))
