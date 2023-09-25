import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import Preprocessing
from PIL import Image
# matplotlib.use('TkAgg')
import os
import torch
from MyUNet.Pytorch_UNet.unet import UNet
import shutil

import SemanticSegmentation
import Evaluate_SS

def extract_gsc_ss(start_folder, end_folder):
    img_subdirs = [os.path.join(start_folder, x) for x in os.listdir(start_folder)]
    for imgs in img_subdirs:
        if not os.path.isdir(imgs):
            continue
        for file in os.listdir(imgs):
            if file.split(".")[0][-4:] == "_gsc":
                shutil.copy(os.path.join(imgs, file), os.path.join(end_folder, file.split(".")[0][:-4] + ".png"))


if __name__ == "__main__":
    # input_folder = "D:\\Dateien\\KI_Speicher\\DNA\\Eval_Folder"
    input_folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\CompleteEvalF"
    # input_folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\Test2"
    result_ss = os.path.join(input_folder, "Result_SS")
    os.makedirs(result_ss, exist_ok=True)
    pret_imgs = os.path.join(input_folder, "Pret_Files")
    os.makedirs(pret_imgs, exist_ok=True)
    result_dist = os.path.join(input_folder, "Result_Dist")
    os.makedirs(result_dist, exist_ok=True)
    ss_labelfolder = os.path.join(input_folder, "SS_Labels")
    os.makedirs(ss_labelfolder, exist_ok=True)
    TRAIN_BINARY = False
    img_subdir='images'
    TRANSPOSE = True

    histo  = "D:\Dateien\KI_Speicher\DNA\SS_TrainData\Rects_Histo\Train\checkpoints\checkpoint_epoch4.pth"
    halfSig = "D:\Dateien\KI_Speicher\DNA\SS_TrainData\Rects_halfSigmoid\Train\checkpoints\checkpoint_epoch4.pth"
    new_halfsig = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NewHS_L90_V3\\Train\\checkpoints\\checkpoint_epoch1.pth"
    new_bin_HS = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\Train\\checkpoints_bin\\checkpoint_epoch5.pth"
    nb2 = "D:\\Dateien\\KI_Speicher\\DNA\\BinModels\\10101400.pth"
    nb3AdamW = "D:\\Dateien\\KI_Speicher\\DNA\\BinModels\\10101515AdamW.pth"
    nb4testlarge = "D:\\Dateien\\KI_Speicher\\DNA\\BinModels\\TestlargeSet.pth"
    nb5testFTI = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k_brd\\Train\\checkpoints_bin\\checkpoint_epoch1.pth"
    nb6_Testing = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement\\SS_DNA_Train\\NewHS_V3_2M_5k\\Train\\checkpoints\\checkpoint_epoch2.pth"
    hsLong3Class = "D:\Dateien\KI_Speicher\DNA\BinModels\\Long3Class2.pth"
    hsMyLoss3EP = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement\\SS_DNA_Train\\NewHS_V3_2M_5k\\Train\\checkpoints\\checkpoint_epoch2.pth"
    test_model = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement\\SS_DNA_Train\\NewHS_L90_V3_5k_newPP\\Train\\checkpoints\\checkpoint_epoch5.pth"
    ns = "D:\\Dateien\\KI_Speicher\\DNA\\Complete_Eval\\NewGaussResc\\noisySet2.pth"
    model_path = test_model
    # model_path = "D:\\Dateien\\KI_Speicher\\DNA\\Complete_Eval\\TestZiba2308_HS_Thrsh01\\checkpoint_epoch4.pth"
    SemanticSegmentation.TRAIN_BINARY = TRAIN_BINARY
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    accuracy = None
    threads = 10
    threshold = 0.3
    Preprocessing.normalize_soft = Preprocessing.normalize_half_sigmoid
    n_classes = 2 if TRAIN_BINARY else 3
    net = UNet(n_channels=1, n_classes=n_classes, bilinear=True).to(device=DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = net.to(device=DEVICE)

    SemanticSegmentation.evaluate_real(model=model, device=DEVICE, folder=os.path.join(input_folder, img_subdir),
                                       folder_pt=os.path.join(input_folder,  "Pret_Files"),
                                       folder_results=os.path.join(input_folder, "Result_SS"),
                                       save_pret=os.path.join(input_folder,  "pret_imgs"),
                                       label_folder=None,
                                       line_corr=False,
                                       do_flatten=False, do_flatten_border=True, enhance_contrast=True,  flip=False,
                                       threshold=threshold)

    extract_gsc_ss(result_ss, ss_labelfolder)
    Evaluate_SS.SHOW_ALL = False
    transpose_labels = TRANSPOSE

    if transpose_labels:
        labels = [os.path.join(ss_labelfolder, x) for x in os.listdir(ss_labelfolder)]
        for lbl in labels:
            img = Image.open(lbl)
            img = img.transpose(method=Image.TRANSPOSE)
            img.save(lbl)

    distances = Evaluate_SS.evaluate_ss(pret_imgs, ss_labelfolder, result_dist, threads=0, accuracy=accuracy,
                                        show_all=False)
    res = {}
    for i, elem in enumerate(os.listdir(os.path.join(input_folder, img_subdir))):
        res[elem.split(".")[0]] = [distances[i]]

    df = pd.DataFrame.from_dict(res)
    df = df.transpose()
    df.to_csv(os.path.join(input_folder, "results.csv"))




