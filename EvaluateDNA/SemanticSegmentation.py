from dataclasses import dataclass

import scipy.special
import torch
import os
from pathlib import Path
import logging
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import shutil
from SPM_Filetype import SPM

# matplotlib.use('TkAgg')

import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import Preprocessing
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import copy
import SS_Mask2Img

from MyUNet.Pytorch_UNet.predict import predict_img
from MyUNet.Pytorch_UNet.utils.dice_score import multiclass_dice_coeff, dice_coeff
from MyUNet.Pytorch_UNet.utils.data_loading import BasicDataset, EvalDataset, BasicDatasetOLD, EvalDatasetOLD
from MyUNet.Pytorch_UNet.utils.dice_score import dice_loss
from MyUNet.Pytorch_UNet.evaluate import evaluate
from MyUNet.Pytorch_UNet.unet import UNet
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pandas as pd
import datetime
import torch
import sys
from torchmetrics import JaccardIndex
from CustomLoss import *
#from torchmetrics.classification import MulticlassJaccardIndex


is_on_d = False
# set = "NewHS_L90_V3" # "OnlySS_Bin_5k" # "NewHS_L90_V3"
# set = "OnlySS_Bin_5k_brd"
# set = "OnlySS_Bin_5k"
# set = "NewHS_V3_2M_5k"
# set ="NewHS_L90_V3_5k"
# set = "NewHS_L90_V3_5k"
# set = "NewHS_V3_2Markers"
# set = "NewHS_L90_V3_5k_newPP"
#  = "Upscaled64"
# set= "AllAlignX"
# set= "AllAlignX"
set = "Set50_morevar"

if is_on_d:
    set_short = set
    set = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\" + set
else:
    set_short = set
    set = os.path.join("SS_DNA_Train", set)



TRAIN_BINARY=False
BILINEAR = False
IMAGE_SET = os.path.join(set, "Train")
TEST_SET = os.path.join(set, "Test")
PRET_IMG = os.path.join(set, "Train_pt")
PRET_TEST = os.path.join(set, "Test_pt")


date = datetime.datetime.now()
kls = date.strftime("%m_%d___%H_%M")

RES_DIR = os.path.join(r'C:\Users\seifert\PycharmProjects\DNA_Measurement\SS_DNA_Train\Results', set_short, kls)
EVAL_TRAIN = r'C:\Users\seifert\PycharmProjects\DNA_Measurement\SS_DNA_Train\RealExamples'
EVAL_TRAIN_RES = os.path.join(RES_DIR, "RealExamples")



#
# IMAGE_SET = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Train")
# TEST_SET = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Test")
# PRET_IMG = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Train_pt")
# PRET_TEST = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Test_pt")


# images = os.path.join(IMAGE_SET, "sxm", "numpy") #  Normal
images = os.path.join(IMAGE_SET, "bild")
labels = os.path.join(IMAGE_SET, "data", "PNG")
dir_img = Path(images)
dir_mask = Path(labels)
dir_img_pret = os.path.join(PRET_IMG, "sxm", "numpy")
if TRAIN_BINARY:
    dir_mask_pret = os.path.join(PRET_IMG, "data_bin", "numpy")
else:
    dir_mask_pret = os.path.join(PRET_IMG, "data", "numpy")

# dir_img_test = os.path.join(TEST_SET, "sxm", "numpy") # Normal
dir_img_test = os.path.join(TEST_SET, "bild")

dir_mask_test = os.path.join(TEST_SET, "data", "PNG")
dir_img_test_pret = os.path.join(PRET_TEST, "sxm", "numpy")
if TRAIN_BINARY:
    dir_mask_test_pret = os.path.join(PRET_TEST, "data_bin", "numpy")
else:
    dir_mask_test_pret = os.path.join(PRET_TEST, "data", "numpy")

# sorting_data_folder = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", "NoNoise1Test", "data")
if TRAIN_BINARY:
    results_dir = os.path.join(RES_DIR, "results_bin")
else:
    results_dir = os.path.join(RES_DIR, "results")


fig_fn_loss = None
fig_fn_val = None
fig_fn_both = None
losses = []
validations = []
cuda0 = torch.device('cuda:0')
DEVICE = cuda0 if torch.cuda.is_available() else torch.device('cpu')
LOAD = False
EPOCHS = 5
LR = 5e-4
BATCH_SIZE = 24
IMG_SCALE = 1
VAL_PERCENT = 0.1
AMP = True
LOG_STEPS = 10
IMAGE_SIZE = None
VALIDATION_STEP = 10  # 1 means 10 per epoch
SHOW_BACTH = False
LOSS_FKT = 'mwl' # 'dice'
MAX_IMAGES = None
ONLY_PT=False
CEmax=0.4
CExsc=2

def before_anything():
    global fig_fn_loss, fig_fn_val, losses, validations, fig_fn_both
    currentdatetime = datetime.datetime.now()
    fig_folder = os.path.join("Log", "Model{}_{}__{}{}{}".format(str(currentdatetime.day).zfill(2),
                                                                 str(currentdatetime.month).zfill(2),
                                                                 str(currentdatetime.hour).zfill(2),
                                                                 str(currentdatetime.minute).zfill(2),
                                                                 str(currentdatetime.second).zfill(2)))
    os.makedirs(fig_folder, exist_ok=True)
    fig_fn_loss = os.path.join(fig_folder, "Loss")
    fig_fn_val = os.path.join(fig_folder, "val")
    fig_fn_both = os.path.join(fig_folder, "both")

    losses = []
    validations = []
    print("Running Skript")

    os.makedirs(results_dir, exist_ok=True)


if TRAIN_BINARY:
    dir_checkpoint = Path(RES_DIR, "checkpoints_bin")
else:

    dir_checkpoint = Path(RES_DIR, "checkpoints")

use_pret = True
loaded_epochs = 0


def plot_losses_vals():
    return
    if len(losses) == 0:
        return

    xs = []
    ys = []
    for pair in losses:
        xs.append(pair[0])
        ys.append(pair[1])

    plt.plot(xs, ys)
    plt.title("Loss")
    plt.xlabel('steps')
    plt.ylabel("train loss")
    plt.savefig(fig_fn_loss)
    # print("Saved loss to ", fig_fn_loss)
    plt.clf()
    if len(validations) == 0:
        return
    xsv = []
    ysv = []
    for pair in validations:
        xsv.append(pair[0])
        ysv.append(pair[1])

    plt.plot(xsv, ysv)
    plt.title("Validation loss")
    plt.xlabel('steps')
    plt.ylabel("val loss")
    plt.savefig(fig_fn_val)
    plt.clf()
    plt.plot(xs, ys, label='train')
    plt.plot(xsv, ysv, label='val')
    plt.legend()
    plt.title("loss")
    plt.xlabel('steps')
    plt.ylabel("loss")
    plt.savefig(fig_fn_both)
    plt.clf()


def test_mask_availablility():
    success = True
    if use_pret:
        lbls = [x for x in os.listdir(dir_mask_pret)]
        for x in tqdm(os.listdir(dir_img_pret)):
            if os.path.isdir(os.path.join(dir_img_pret, x)):
                continue
            lbl_fn = x[:-4] + "_mask"
            availables = []
            for lfn in lbls:
                if lfn.startswith(lbl_fn):
                    availables.append(lfn)

            if len(availables) == 1:
                # print(x, " --> ", availables[0])
                pass
            elif len(availables) == 0:
                print("XXX no fiiting mask in lables found for ", x)
                success = False

            else:
                print("XXX Multiple labels for ", x, " ---> ", availables)
                success = False

    else:
        lbls = [x for x in os.listdir(dir_mask)]
        # print(lbls)
        for x in tqdm(os.listdir(dir_img)):
            if os.path.isdir(os.path.join(dir_img, x)):
                continue
            lbl_fn = x[:-4] + "_mask"
            availables = []
            # print("expecting ", lbl_fn)
            # time.sleep(1)
            for lfn in lbls:
                if lfn.startswith(lbl_fn):
                    availables.append(lfn)

            if len(availables) == 1:
                # print(x, " --> ", availables[0])
                pass
            elif len(availables) == 0:
                print("XXX no fiiting mask in lables found for ", x)
                success = False
            else:
                print("XXX Multiple labels for ", x, " ---> ", availables)
                success = False

    if success:
        print("All labels complete")
    else:
        print("Missing labels")

    time.sleep(1)


def rename_all(zfil=5):
    # Images:
    nps = [os.path.join(dir_img, x) for x in os.listdir(dir_img) if x.endswith("npy")]
    for elem in tqdm(nps):
        parts = elem.split("\\")
        fn = parts[-1]
        fn2 = "Image" + str(fn[5:-4]).zfill(zfil) + ".npy"

        fn_new = os.path.join("\\".join(parts[:-1]), fn2)
        os.rename(elem, fn_new)

    nps = [os.path.join(dir_img_pret, x) for x in os.listdir(dir_img_pret) if x.endswith("npy")]
    for elem in tqdm(nps):
        parts = elem.split("\\")
        fn = parts[-1]
        fn2 = "Image" + str(fn[5:-4]).zfill(zfil) + ".npy"

        fn_new = os.path.join("\\".join(parts[:-1]), fn2)
        os.rename(elem, fn_new)

    lbls = [os.path.join(dir_mask, x) for x in os.listdir(dir_mask) if x.endswith("png")]
    for elem in tqdm(lbls):
        parts = elem.split("\\")
        fn = parts[-1]
        fn2 = "Image" + str(fn[8:-4]).zfill(zfil) + "_mask.png"

        fn_new = os.path.join("\\".join(parts[:-1]), fn2)
        os.rename(elem, fn_new)

    lbls = [os.path.join(dir_mask_pret, x) for x in os.listdir(dir_mask_pret) if x.endswith("npy")]
    for elem in tqdm(lbls):
        parts = elem.split("\\")
        fn = parts[-1]
        fn2 = "Image" + str(fn[8:-4]).zfill(zfil) + "_mask.npy"

        fn_new = os.path.join("\\".join(parts[:-1]), fn2)
        os.rename(elem, fn_new)


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    validation_step = 5
    # 0. Generate pretransformed images

    # Preprocessing.sort_data_folder(sorting_data_folder)
    # Preprocessing.zfill_img_folder(dir_img)
    # Preprocessing.zfill_lbl_folder(dir_mask)
    # Preprocessing.zfill_img_folder(dir_img_test)
    # Preprocessing.zfill_lbl_folder(dir_mask_test)
    if not ONLY_PT:
        Preprocessing.normalize_soft = Preprocessing.normalize_half_sigmoid

        show = False
        enhance_contrast = True
        zfill = 6
        threads = 10
        img_size = 50
        overwrite = False
        lines_corr = False
        do_flatten = False
        flip = False
        flatten_line_90 = False
        do_flatten_border = True
        use_masks = True
        use_Test = True
        keep_name = False
        resize_method = "bilinear"
        Preprocessing.pretransform_all(dir_img=dir_img, dir_mask=dir_mask, dir_img_test=dir_img_test,
                                       dir_mask_test=dir_mask_test, dir_img_pret=dir_img_pret,
                                       dir_mask_pret=dir_mask_pret,
                                       dir_img_test_pret=dir_img_test_pret, dir_mask_test_pret=dir_mask_test_pret,
                                       show=show, enhance_contrast=enhance_contrast, zfill=zfill,
                                       threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                                       do_flatten=do_flatten,
                                       do_flatten_border=do_flatten_border, flip=flip, flatten_line_90=flatten_line_90,
                                       keep_name=keep_name,
                                       use_Test=use_Test, use_masks=use_masks, resize_method=resize_method)

        #Preprocessing.pretransform_all(dir_img, dir_mask, dir_img_test, dir_mask_test, dir_img_pret, dir_mask_pret,
        #                           dir_img_test_pret, dir_mask_test_pret, show=False, enhance_contrast=True, zfill=6,
        #                           threads=10, img_size=IMAGE_SIZE, overwrite=False, lines_corr=False, do_flatten=False,
        #                           do_flatten_border=True, flip=False, flatten_line_90=False, resize_method="bilinear")

    # if use_pret:
    #     os.makedirs(dir_mask_pret, exist_ok=True)
    #     os.makedirs(dir_img_pret, exist_ok=True)
    #     if len([x for x in os.listdir(dir_mask) if not os.path.isdir(os.path.join(dir_mask,x))]) != len(os.listdir(dir_mask_pret)):
    #         for fn in tqdm(os.listdir(dir_mask), desc="Pretransforming Masks"):
    #             pretransform_image(os.path.join(dir_mask, fn), os.path.join(dir_mask_pret, fn[:-4] + ".npy"), is_mask=True)
    #     if len([x for x in os.listdir(dir_img) if not os.path.isdir(os.path.join(dir_img,x))]) != len(os.listdir(dir_img_pret)):
    #         for fn in tqdm(os.listdir(dir_img), desc="Pretransforming"):
    #             pretransform_image(os.path.join(dir_img, fn), os.path.join(dir_img_pret, fn), is_mask=False)

    # rename_all(5)

    # 1. Create dataset
    #if use_pret:
    dataset = BasicDataset(dir_img_pret, dir_mask_pret, 1.0, "_mask", image_num=MAX_IMAGES)
    #else:
    #    dataset = BasicDataset(dir_img, dir_mask, 1.0, "_mask")

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project="U-Net {}".format(set_short), resume='allow', entity="tims57gtw")
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {DEVICE} # Device.type
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    print("Device: ", next(net.parameters()).device)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.02, total_iters=int(len(train_set) * EPOCHS/batch_size))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=LR, pct_start=0.1, total_steps=int(len(train_set) * EPOCHS / batch_size) + 10)
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, step_size_up=100, base_lr=0.5*LR, max_lr=2*LR)
    print("Total Steps: ", int(len(train_set) * EPOCHS) / batch_size)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.1*learning_rate, max_lr=learning_rate, mode='triangular2', step_size_up=int(len(train_set) * EPOCHS / 5) , step_size_down=None, cycle_momentum=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ce = nn.CrossEntropyLoss()
    ce_max = CEmax
    ce_xsc = CExsc
    g = (ce_max * ce_xsc) /( ce_max * np.real(scipy.special.lambertw(-np.exp(-ce_xsc/ce_max)*ce_xsc/ce_max)) + ce_xsc)
    ce_fak = lambda x : g * (1-np.exp(-ce_xsc*x/g))


    if LOSS_FKT == 'dice':
        # criterion = lambda x, y: my_dice_loss(x,y,multiclass=False, classes=2)
        ce = nn.CrossEntropyLoss()
        criterion = lambda x, y: my_dice_loss(x, y) + ce(x, y)
        # criterion = lambda x, y: dice_loss(F.softmax(x, dim=1).float(), F.one_hot(y, 2).permute(0, 3, 1, 2).float()) + ce(x, y)
    elif LOSS_FKT.lower() == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif LOSS_FKT.lower() == 'wdl':
        wdl = WDL(classes=3, weights=[0.017974, 0.042809, 0.939218])
        criterion = lambda x, y: wdl(F.softmax(x, dim=1).float(),
                                         F.one_hot(y, 2 if TRAIN_BINARY else 3).permute(0, 3, 1, 2).float())

    elif LOSS_FKT.lower() == 'mwl':
        wdl = MyWeightedLoss(classes=3, weights=[0.017974, 0.042809, 0.939218])

        criterion = lambda x, y: wdl(F.softmax(x, dim=1).float(),F.one_hot(y, 2 if TRAIN_BINARY else 3).permute(0, 3, 1, 2).float())

    elif LOSS_FKT.lower() == 'tversky':
        criterion = TverskyLoss()
    elif LOSS_FKT.lower() == "focaltversky":
        criterion = focal_tversky
    else:
        print("Unknown Keyword")
        criterion = lambda x, y: dice_loss(F.softmax(x, dim=1).float(),
                                           F.one_hot(y, 2 if TRAIN_BINARY else 3).permute(0, 3, 1, 2).float(),
                                       multiclass=False)
    global_step = 0
    total_steps = int(len(train_set) * EPOCHS/batch_size)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # 5. Begin training
    with torch.cuda.device(0):
        for epoch in range(1, epochs + 1):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    # SHOW_BACTH = np.random.uniform() < 0.01
                    images = batch['image']
                    true_masks = batch['mask']
                    if SHOW_BACTH:
                        fig, axs = plt.subplots(nrows=4, ncols=4)
                        for i in range(min(16, len(images))):
                            axs[i // 4, i % 4].imshow(images[i][0], cmap='gray')
                        plt.show()
                        fig, axs = plt.subplots(nrows=4, ncols=4)
                        for i in range(min(16, len(true_masks))):
                            axs[i // 4, i % 4].imshow(true_masks[i], cmap='gray')
                        plt.show()

                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'


                    images = images.to(device=DEVICE)
                    true_masks = true_masks.to(device=DEVICE)

                    with torch.cuda.amp.autocast(enabled=amp):
                        # print("Images: ", images.device)
                        masks_pred = net(images)

                        if SHOW_BACTH:
                            mp = torch.sigmoid(masks_pred).detach().cpu().permute(0, 2, 3, 1).numpy().astype(float)
                            fig, axs = plt.subplots(nrows=4, ncols=4)
                            for i in range(min(16, len(mp))):
                                axs[i // 4, i % 4].imshow(mp[i], cmap='gray')
                            plt.show()


                        cef = ce_fak(global_step / total_steps)
                        cr_loss = criterion(masks_pred, true_masks)
                        if cef != 0:
                            ce_loss = ce(masks_pred, true_masks)
                        else:
                            ce_loss = torch.from_numpy(np.array([0])).to(device=DEVICE)

                        loss = (1 - cef) * cr_loss + cef * ce_loss

#
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    losses.append((global_step, loss.item()))
                    if global_step % LOG_STEPS == 0:
                        plot_losses_vals()
                        experiment.log({
                            'train loss': loss.item(),
                            'CE loss': ce_loss.item(),
                            'CR loss': cr_loss.item(),
                            'step': global_step,
                            'epoch': epoch,
                            'lr': get_lr(optimizer=optimizer)
                        })

                    experiment.log({
                        'train loss': loss.item(),
                        'CE loss': ce_loss.item(),
                        'CR loss': cr_loss.item(),
                        'CE factor': cef,
                        'step': global_step,
                        'epoch': epoch,
                        'lr': get_lr(optimizer=optimizer)
                    })
                    scheduler.step()
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'step': global_step})

                    # Evaluation round
                    division_step = VALIDATION_STEP * (n_train // (10 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:

                            histograms = {}
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())


                            val_score = evaluate(net, val_loader, DEVICE)
                            validations.append((global_step, val_score.cpu().numpy()))
                            plot_losses_vals()
                            # scheduler.step(val_score)

                            logging.info('Validation Dice score: {}'.format(val_score))
                            print("Val Score: ", val_score)
                            try:  # experiment.
                                wandb.log({
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(
                                            torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except FileNotFoundError:
                                logging.warning("File for experiment log not found")
                                pass

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(),
                           str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + loaded_epochs)))
                logging.info(f'Checkpoint {epoch + loaded_epochs} saved!')
                print("Saved ", str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + loaded_epochs)))

    return experiment
    # wdl.plot_ws()

def predict_all(net, folder=None):
    if folder is not None:
        set = folder
        if is_on_d:
            set_short = set
            set = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NoiseSeries\\" + set
        else:
            set_short = set
            set = os.path.join("SS_DNA_Train", set)
        IMAGE_SET = os.path.join(set, "Train")
        TEST_SET = os.path.join(set, "Test")
        PRET_IMG = os.path.join(set, "Train_pt")
        PRET_TEST = os.path.join(set, "Test_pt")

        #
        # IMAGE_SET = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Train")
        # TEST_SET = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Test")
        # PRET_IMG = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Train_pt")
        # PRET_TEST = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", set, "Test_pt")

        # images = os.path.join(IMAGE_SET, "sxm", "numpy") #  Normal
        images = os.path.join(IMAGE_SET, "bild")
        labels = os.path.join(IMAGE_SET, "data", "PNG")
        dir_img = Path(images)
        dir_mask = Path(labels)
        dir_img_pret = os.path.join(PRET_IMG, "sxm", "numpy")
        if TRAIN_BINARY:
            dir_mask_pret = os.path.join(PRET_IMG, "data_bin", "numpy")
        else:
            dir_mask_pret = os.path.join(PRET_IMG, "data", "numpy")

        # dir_img_test = os.path.join(TEST_SET, "sxm", "numpy") # Normal
        dir_img_test = os.path.join(TEST_SET, "bild")

        dir_mask_test = os.path.join(TEST_SET, "data", "PNG")
        dir_img_test_pret = os.path.join(PRET_TEST, "sxm", "numpy")
        if TRAIN_BINARY:
            dir_mask_test_pret = os.path.join(PRET_TEST, "data_bin", "numpy")
        else:
            dir_mask_test_pret = os.path.join(PRET_TEST, "data", "numpy")

        # sorting_data_folder = os.path.join("/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train", "NoNoise1Test", "data")
        if TRAIN_BINARY:
            results_dir = os.path.join(TEST_SET, "results_bin")
        else:
            results_dir = os.path.join(TEST_SET, "results")
    ce = nn.CrossEntropyLoss()
    gen_folders = True
    if gen_folders:
        os.makedirs(os.path.join(results_dir, "0_image"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "0_label"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "0_pred"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "0_npy"), exist_ok=True)


    wdl = MyWeightedLoss(classes=3, weights=[0.017974, 0.042809, 0.939218])
    mwl_fk = lambda x, y: wdl(F.softmax(x, dim=1).float(),
                                 F.one_hot(y, 2 if TRAIN_BINARY else 3).permute(0, 3, 1, 2).float())

    amp = True
    use_pret = True
    batch_size = 20

    imgs = []
    tmasks = []
    preds = []
    sms = []
    crents = []
    dices = []
    mwls = []
    losses = []



    idx = 0

    if not ONLY_PT:
        Preprocessing.normalize_soft = Preprocessing.normalize_half_sigmoid

        show = False
        enhance_contrast = True
        zfill = 4
        threads = 10
        img_size = 50
        overwrite = False
        lines_corr = False
        do_flatten = False
        flip = False
        flatten_line_90 = False
        do_flatten_border = True
        use_masks = True
        use_Test = True
        keep_name = False
        resize_method = "bilinear"
        Preprocessing.pretransform_all(dir_img=dir_img, dir_mask=dir_mask, dir_img_test=dir_img_test,
                                       dir_mask_test=dir_mask_test, dir_img_pret=dir_img_pret,
                                       dir_mask_pret=dir_mask_pret,
                                       dir_img_test_pret=dir_img_test_pret, dir_mask_test_pret=dir_mask_test_pret,
                                       show=show, enhance_contrast=enhance_contrast, zfill=zfill,
                                       threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                                       do_flatten=do_flatten,
                                       do_flatten_border=do_flatten_border, flip=flip, flatten_line_90=flatten_line_90,
                                       keep_name=keep_name,
                                       use_Test=use_Test, use_masks=use_masks, resize_method=resize_method)


    if use_pret:
        dataset = BasicDataset(dir_img_test_pret, dir_mask_test_pret, 1.0, "_mask")
        fnames_img = [x for x in os.listdir(dir_img_test_pret)]
        fnames_mask = [x for x in os.listdir(dir_mask_test_pret)]


    else:
        dataset = BasicDataset(dir_img_test, dir_mask_test, 1.0, "_mask")
        fnames_img = [x for x in os.listdir(dir_img_test)]
        # fnames_mask = [x for x in os.listdir(dir_mask_test)]

    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=False)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)

    for batch in tqdm(test_loader, desc="Evaluating Images", unit="batch"):
        images = batch['image']
        true_masks = batch['mask']

        assert images.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        # images = images.to(device=DEVICE, dtype=torch.float32)
        # true_masks = true_masks.to(device=DEVICE, dtype=torch.long)

        # images = torch.from_numpy(images).cuda(device=cuda0) #(device=cuda0, dtype=torch.float32)
        # true_masks = torch.from_numpy(true_masks).cuda(device=cuda0)# device=cuda0, dtype=torch.long)

        # img = images[0][0]
        # plt.imshow(img.cpu())
        # plt.title("Img")
        # plt.show()
        #
        # mask = true_masks[0]
        # plt.imshow(mask.cpu())
        # plt.title("Mask")
        # plt.show()
        with torch.cuda.amp.autocast(enabled=amp):
            images = images.to(device=DEVICE)
            masks_pred = net(images)

            # plt.imshow(masks_pred[0][0].detach().cpu().numpy())
            # plt.show()

            # plt.imshow(masks_pred[0][1].detach().cpu().numpy())
            # plt.show()

            sm = F.softmax(masks_pred, dim=1).float()
            sm_eval = sm.detach().cpu()
            sm_eval = sm_eval.permute(0, 2, 3, 1)

            imgs_eval = images.detach().cpu()
            preds_eval = masks_pred.detach().cpu()
            true_eval = true_masks.detach().cpu()

        for i in range(batch_size):

            try:
                fn = fnames_img[idx]
            except IndexError:
                continue



            idx += 1

            thispth = os.path.join(results_dir, fn[:-4])
            os.makedirs(thispth, exist_ok=True)

            fnmat = os.path.join(thispth, fn[:-4] + "_gsc.png")
            fnmask = os.path.join(thispth, fn[:-4] + "_mask.png")
            real_pth = os.path.join(thispth, fn[:-4] + ".png")
            fncol = os.path.join(thispth, fn[:-4] + "_col.png")

            img = imgs_eval[i][0]

            # img = img.T

            true_mask = true_eval[i]
            mask_pred = preds_eval[i]
            sm_loc = sm_eval[i].cpu().numpy()

            # padded_pred = mask_pred[np.newaxis, ...].to(device=DEVICE, dtype=torch.float32)
            # padded_true = true_mask[np.newaxis, ...].to(device=DEVICE, dtype=torch.long)
            padded_pred = mask_pred[np.newaxis, ...]
            padded_true = true_mask[np.newaxis, ...]

            padded_pred = padded_pred.to(dtype=torch.float32)
            padded_true = padded_true.to(dtype=torch.long)

            # padded_pred = torch.from_numpy(mask_pred).cuda(device=cuda0) # , device=cuda0, dtype=torch.float32)
            # padded_true = torch.from_numpy(padded_true).cuda(device=cuda0)# , device=cuda0, dtype=torch.long)

            cross_entr = ce(padded_pred, padded_true).cpu().numpy()

            # 1 evtl
            oh = F.one_hot(padded_true, net.n_classes).permute(0, 3, 1, 2).float()
            dice = dice_loss(F.softmax(padded_pred, dim=1).float(),
                             oh,
                             multiclass=True)
            dice = dice.cpu().numpy()


            mwl = mwl_fk(padded_pred, padded_true).cpu().numpy()

            imgs.append(real_pth)
            plt.imsave(real_pth, img, cmap="gray")
            plt.imsave(os.path.join(results_dir, "0_image", f"Image{str(idx).zfill(4)}.png"), img, cmap="gray")

            tmasks.append(fnmask)
            plt.imsave(fnmask, true_mask, cmap="gray")
            plt.imsave(os.path.join(results_dir, "0_label", f"Image{str(idx).zfill(4)}.png"), true_mask, cmap="gray")


            sms.append(fncol)
            plt.imsave(fncol, sm_loc)

            gsc = np.zeros(sm_loc.shape[:-1])
            for i in range(sm_loc.shape[0]):
                for j in range(sm_loc.shape[1]):
                    gsc[i, j] = np.argmax(sm_loc[i, j])

            preds.append(fnmat)
            plt.imsave(fnmat, gsc, cmap='gray')
            plt.imsave(os.path.join(results_dir, "0_pred", f"Image{str(idx).zfill(4)}.png"), gsc, cmap='gray')

            shutil.copy(os.path.join(dir_img_test_pret, fn), os.path.join(results_dir, "0_npy", f"Image{str(idx).zfill(4)}.npy"))

            crents.append(float(cross_entr))
            dices.append(float(dice))
            mwls.append(float(mwl))
            losses.append(float(cross_entr + mwl + dice))

    # print("imgs: ", len(imgs))
    # print("tmasks: ", len(tmasks))
    # print("sms: ", len(sms))
    # print("preds: ", len(preds))
    # print("crits: ", len(crits))
    # print("dices: ", len(dices))
    # print("losses: ", len(losses))
    # print("zip: ", len(list(zip(imgs, tmasks, sms, preds, crits, dices, losses))))

    with open(os.path.join(results_dir, "results.csv"), "w") as f:
        f.write('Image,CrossEntr,Dice,MWL,Sum\n')
        for i in range(len(losses)):
            temp = imgs[i].split('\\')[-1].split('.')[0]
            f.write(f"{temp},{crents[i]},{dices[i]},{mwls[i]},{losses[i]}\n")

    # df = pd.DataFrame(list(zip(imgs, tmasks, sms, preds, crents, dices, mwls,  losses)),
    #                   columns=['Image', "Mask", "Col-Path", "Gsc-Path", "CrossEntr", "Dice", "MWL", "Sum"])
    # # df.sort_values("Image")
    # df.to_csv(os.path.join(results_dir, "results.csv"), sep=";", decimal=',')
    print("Saved results to ", os.path.join(results_dir, "results.csv"))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def evaluate_real(model, device, folder, folder_pt, folder_results, label_folder=None, save_pret=None, line_corr=False,
                  do_flatten=False, do_flatten_border=False,
                  enhance_contrast=True, flip=False, threshold=0.3, flatten_line_90=True, workers=2, skip_pp=False, old_model=False):

    fnames_img = [x for x in os.listdir(folder)]

    os.makedirs(folder_pt, exist_ok=True)
    os.makedirs(folder_results, exist_ok=True)
    if save_pret is not None:
        os.makedirs(save_pret, exist_ok=True)

    if len(os.listdir(folder)) != len(os.listdir(folder_pt)):

        for fn in tqdm(os.listdir(folder), desc="Pretransform Real"):
            fn_prev = os.path.join(folder, fn)
            fn_after = os.path.join(folder_pt, fn[:-3] + "npy")

            Preprocessing.pretransform_image(fn_prev, fn_after, IMAGE_SIZE, line_corr=line_corr, do_flatten=do_flatten,
                                             do_flatten_border=do_flatten_border,
                                             enhance_contrast=enhance_contrast, flip=flip, show=False,
                                             flatten_line_90=flatten_line_90, skip_all=False)
            if save_pret is not None:
                prett = np.load(fn_after, allow_pickle=True)
                # plt.imshow(prett)
                # plt.show()
                plt.imsave(os.path.join(save_pret, fn[:-3] + "png"), prett, cmap="gray")

    compare_labels = label_folder is not None

    if compare_labels:
        dataset = BasicDataset(folder_pt, label_folder, mask_suffix="_mask") if not old_model else BasicDatasetOLD(folder_pt, label_folder, mask_suffix="_mask")
        batch_size = 20
        if TRAIN_BINARY:
            jaccard = JaccardIndex(num_classes=2)
        else:
            jaccard = JaccardIndex(num_classes=3, ignore_index=0)
        iou_s = []
    else:
        dataset = EvalDataset(folder_pt) if not old_model else EvalDatasetOLD(folder_pt)
        batch_size = min(20, len(os.listdir(folder)))


    batch_size = 1

    amp = True
    use_pret = True

    imgs = []
    preds = []
    sms = []

    idx = 0
    dice_score = 0


    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=False)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)
    ious = []

    model.eval()

    for batch in tqdm(test_loader, desc="Semantic Segmentation Real files", unit="batch"):
        images = batch['image']


        if compare_labels:
            mask_true = batch['mask']


        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        with torch.cuda.amp.autocast(enabled=amp):
            images = images.to(device=DEVICE)
            masks_pred = model(images)

            sm = F.softmax(masks_pred, dim=1).float()
            sm_eval = sm.detach().cpu()
            sm_eval = sm_eval.permute(0, 2, 3, 1)



            imgs_eval = images.detach().cpu()

        for i in tqdm(range(len(sm_eval)), disable=True):

            try:
                fn = fnames_img[idx]
            except IndexError:
                print("Index Error")
                print("Idx: ", idx)
                print("BS: ", batch_size)
                print("Fns: ", fnames_img)
                continue

            idx += 1

            thispth = os.path.join(folder_results, fn[:-4])
            os.makedirs(thispth, exist_ok=True)

            fnmat = os.path.join(thispth, fn[:-4] + "_gsc.png")
            real_pth = os.path.join(thispth, fn[:-4] + ".png")
            fncol = os.path.join(thispth, fn[:-4] + "_col.png")

            img = imgs_eval[i][0]

            sm_loc = sm_eval[i].cpu().numpy()
            #print(sm_loc.shape)
            #plt.imshow(sm_loc[:, :, 0])
            #plt.title(0)
            #plt.show()
            #plt.imshow(sm_loc[:, :, 1])
            #plt.title(1)
            #plt.show()

            if TRAIN_BINARY:
                sm_image = np.zeros((sm_loc.shape[0], sm_loc.shape[1], 3))
                sm_image[:, :, 0] = sm_loc[:, :, 0]
                sm_image[:, :, 1] = sm_loc[:, :, 1]
            else:
                sm_image = sm_loc

            # plt.imshow(sm_image)
            # plt.title("Sm image")
            # plt.show()
            plt.imsave(fncol, sm_image)


            if compare_labels:
                mt = mask_true[i]
                if TRAIN_BINARY:
                    mt = np.int32(mt)
                    # plt.imshow(mt)
                    # plt.title("MT")
                    # plt.show()
#
                    # mploc = masks_pred[i].detach().cpu().numpy()
                    # plt.imshow(mploc[0])
                    # plt.title("MP0")
                    # plt.show()
                    # plt.imshow(mploc[1])
                    # plt.title("MP1")
                    # plt.show()
                else:
                    mt = np.int32(mt[:, :, 0] / 127)
                mt = torch.from_numpy(mt)
                mt = mt.to(dtype=int)




                # plt.imshow(mt)
                # plt.title("True")
                # plt.show()
                mp = torch.sigmoid(masks_pred[i]).permute(1, 2, 0).detach().cpu().numpy().astype(float)

                # plt.imshow(mp)
                # plt.title("Pred")
                # plt.show()

                # onehot = np.zeros((mp.shape[0], mp.shape[1]))
                # for i in range(onehot.shape[0]):
                #     for j in range(onehot.shape[1]):
                #         onehot[i, j] = np.argmax(mp[i, j, :])

                # plt.imshow(onehot)
                # plt.title("onehot")
                # plt.show()

                mp = mp.astype(float)
                mp = torch.as_tensor(mp)
                # mt = torch.as_tensor(mt)
                # print("MP: ", mp)
                # print(mp.shape)
                # print("MT: ", mt)
                # print(mt.shape)
                # mt = mt[:3, :3, :]
                # true_oh = np.zeros((mt.shape[0], mt.shape[1], 3))
                # for i in range(mt.shape[0]):
                #     for j in range(mt.shape[1]):
                #         true_oh[i, j, int(np.argmax(mt[i, j]))] = 1
                # true_oh = torch.as_tensor(true_oh, dtype=int)
                # plt.imshow(mt)
                # plt.show()
                if TRAIN_BINARY:
                    true_oh = F.one_hot(mt, num_classes=2)
                else:
                    true_oh = F.one_hot(mt, num_classes=3)
                # print("MT: ", mt)
                # print("OH: ", true_oh)
                # #toh = F.one_hot(mt, num_classes=3)
                #print("TOH: ", toh)


                # print("TOh: ", true_oh)
                # print(true_oh.shape)
                iou = jaccard(mp, true_oh)
                ious.append(jaccard(mp, true_oh))


                plt.cla()
                fig, axs = plt.subplots(nrows=1, ncols=3)
                axs[0].imshow(img, cmap='gray')
                if TRAIN_BINARY:
                    temp = np.zeros((true_oh.shape[0], true_oh.shape[0], 3))
                    temp[:, :, 0] = true_oh[:, :, 0]
                    temp[:, :, 1] = true_oh[:, :, 1]
                    temp2 = np.zeros((sm_loc.shape[0], sm_loc.shape[0], 3))
                    temp2[:, :, 0] = sm_loc[:, :, 0]
                    temp2[:, :, 1] = sm_loc[:, :, 1]
                    # print("a")
                    axs[1].imshow(temp)
                    axs[2].imshow(temp2)
                    # print(fncol)
                    plt.imsave(fncol, temp2)


                else:
                    axs[1].imshow(255*true_oh)
                    axs[2].imshow(sm_loc)
                    plt.imsave(fncol, sm_loc)

                plt.suptitle(f"IoU: {iou.item()}")
                iou_s.append(iou.item())
                # plt.show()
                plt.savefig(os.path.join(thispth, "dice.png"))
                plt.cla()

            imgs.append(real_pth)
            plt.imsave(real_pth, img, cmap="gray")

            # plt.imshow(img)
            # plt.title("Image")
            # plt.show()

            sms.append(fncol)
            target_colums = 1 if TRAIN_BINARY else 2
            gsc = np.zeros(sm_loc.shape[:-1])
            for i in range(sm_loc.shape[0]):
                for j in range(sm_loc.shape[1]):
                    argm = np.argmax(sm_loc[i, j])
                    if argm > 0:  # If is molecule or ss
                        if threshold is not None and sm_loc[i, j, target_colums] > threshold:
                            gsc[i, j] = target_colums
                        else:
                            gsc[i, j] = argm

            # plt.imshow(gsc)
            # plt.show()

            preds.append(fnmat)
            plt.imsave(fnmat, gsc, vmax=2, cmap='gray')


    df = pd.DataFrame(list(zip(imgs, sms, preds)),
                      columns=['Image', "Col-Path", "Gsc-Path"])
    df.to_csv(os.path.join(folder_results, "results.csv"), sep=";", decimal=',')
    print("Saved results to ", os.path.join(folder_results, "results.csv"))

    if compare_labels:
        iou = np.average(iou_s)
        with open(os.path.join(folder_results, "iou.txt"), "w") as f:
            f.write("IOU: " + str(iou))

    if len(ious) > 0:
        with open(os.path.join(folder_results, "IOUs.csv"), "w") as f:
            for wl in ious:
                f.write(f"{wl}\n")







def evaluate_real_2(model, device, folder, folder_pt, folder_results, label_folder=None, save_pret=None, line_corr=False,
                  do_flatten=False, do_flatten_border=False,
                  enhance_contrast=True, flip=False, threshold=0.3, flatten_line_90=True, workers=2, skip_pp=False, old_model=False):

    fnames_img = [x for x in os.listdir(folder)]
    assert label_folder is None

    os.makedirs(folder_pt, exist_ok=True)
    os.makedirs(folder_results, exist_ok=True)
    if save_pret is not None:
        os.makedirs(save_pret, exist_ok=True)

    if len(os.listdir(folder)) != len(os.listdir(folder_pt)):

        for fn in tqdm(os.listdir(folder), desc="Pretransform Real"):
            fn_prev = os.path.join(folder, fn)
            fn_after = os.path.join(folder_pt, fn[:-3] + "npy")

            Preprocessing.pretransform_image(fn_prev, fn_after, IMAGE_SIZE, line_corr=line_corr, do_flatten=do_flatten,
                                             do_flatten_border=do_flatten_border,
                                             enhance_contrast=enhance_contrast, flip=flip, show=False,
                                             flatten_line_90=flatten_line_90, skip_all=True)


            if save_pret is not None:
                prett = np.load(fn_after, allow_pickle=True)
                # plt.imshow(prett)
                # plt.show()
                plt.imsave(os.path.join(save_pret, fn[:-3] + "png"), prett, cmap="gray")



    dataset = EvalDataset(folder_pt) if not old_model else EvalDatasetOLD(folder_pt)

    batch_size = 32

    amp = True

    imgs = []
    preds = []
    sms = []

    idx = 0


    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=False)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)
    ious = []

    model.eval()

    for batch in tqdm(test_loader, desc="Semantic Segmentation Real files", unit="batch"):
        images = batch['image']

        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        with torch.cuda.amp.autocast(enabled=amp):
            images = images.to(device=DEVICE)
            masks_pred = model(images)

            images = images.detach().cpu().numpy()
            plt.switch_backend('TkAgg')
            plt.imshow(images[0, 0])
            plt.title("Images")
            plt.show()
            sm = torch.softmax(masks_pred, dim=1).detach().cpu().numpy()
            mp = torch.softmax(masks_pred, dim=1).argmax(dim=1).detach().cpu().numpy()


        for i in range(images.shape[0]):

            try:
                fn = fnames_img[idx]
            except IndexError:
                print("Index Error")
                print("Idx: ", idx)
                print("BS: ", batch_size)
                print("Fns: ", fnames_img)
                continue

            idx += 1

            thispth = os.path.join(folder_results, fn[:-4])
            os.makedirs(thispth, exist_ok=True)

            fnmat = os.path.join(thispth, fn[:-4] + "_gsc.png")
            real_pth = os.path.join(thispth, fn[:-4] + ".png")
            fncol = os.path.join(thispth, fn[:-4] + "_col.png")

            img = images[i][0]

            smplt = sm.transpose(0, 2, 3, 1)
            plt.imsave(fncol, smplt[0])


            imgs.append(real_pth)
            plt.imsave(real_pth, img, cmap="gray")

            sms.append(fncol)


            preds.append(fnmat)
            plt.imsave(fnmat, mp[i], vmin=0, vmax=2, cmap='gray')


    df = pd.DataFrame(list(zip(imgs, sms, preds)),
                      columns=['Image', "Col-Path", "Gsc-Path"])
    df.to_csv(os.path.join(folder_results, "results.csv"), sep=";", decimal=',')
    print("Saved results to ", os.path.join(folder_results, "results.csv"))


    if len(ious) > 0:
        with open(os.path.join(folder_results, "IOUs.csv"), "w") as f:
            for wl in ious:
                f.write(f"{wl}\n")



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION', )
    os.system("nvcc --version")  # does not work
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    if DEVICE != torch.device('cpu'):
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())

    before_anything()
    print(torch.__version__)
    print(torch.cuda.is_available())



    # print("Testing CUDA Stuff")
    # arr = np.array([[1, 2, 3], [4, 7, 8]])
    # tens = torch.from_numpy(arr).to(device=DEVICE)
    # print(tens)
    if TRAIN_BINARY:
        net = UNet(n_channels=1, n_classes=2, bilinear=BILINEAR).to(device=DEVICE)
    else:
        net = UNet(n_channels=1, n_classes=3, bilinear=BILINEAR).to(device=DEVICE)

    # print("Initializing Wandb")
    # experiment = wandb.init(project="U-Net {}".format(set), resume='allow', entity="tims57gtw", settings=wandb.Settings(start_method="thread"))

    if LOAD:
        try:
            MODEL_PATH = os.path.join(dir_checkpoint, "checkpoint_epoch{}.pth".format(
                len(os.listdir(dir_checkpoint))))
            loaded_epochs = len(os.listdir(dir_checkpoint))
        except FileNotFoundError:
            MODEL_PATH = None
            LOAD = False
    else:
        MODEL_PATH = None

    # MODEL_PATH = "SS_DNA_Train\\Small50px_5k\\Train\\checkpoints\\checkpoint_epoch4.pth"
    # MODEL_PATH = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement3\\Models\\SS\\ModelNewPP2410.pth"
    print("Selected Device: ", DEVICE)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if LOAD:
        print("Loaded ", MODEL_PATH)
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logging.info(f'Model loaded from {MODEL_PATH}')

    # net.load_state_dict(torch.load(r'C:\Users\seifert\PycharmProjects\DNA_Measurement\SS_DNA_Train\Results\62nm_10k_64px\07_21___14_38\checkpoints\checkpoint_epoch5.pth', map_location=DEVICE))


    print(" Moving net to device ", DEVICE)
    model = net.to(device=DEVICE)
    if EPOCHS != 0:
        try:
            run = train_net(net=model,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      learning_rate=LR,
                      device=DEVICE,
                      img_scale=IMG_SCALE,
                      val_percent=VAL_PERCENT,
                      amp=AMP)
            os.makedirs(RES_DIR, exist_ok=True)
            with open(os.path.join(RES_DIR, "settings.txt"), 'w') as f:
                f.write(f"{run.id}\n{run.name}")
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            raise

    # pbar = tqdm(desc="Evakuating Noise series", total=24)
    # for e in ["X", "Y"]:
    #     for i in range(6):
    #         for j in [0, 5]:
    #             predict_all(net, folder=f"{e}\\N{i}K{j}")
    #             pbar.update(1)
    # exit()

    print("Evaluating real images")

    real_folder = os.path.join("EvaluateReal", "Ziba2308", "Crops")

    #evaluate_real(model, DEVICE, folder=os.path.join(real_folder, "images"), folder_pt=os.path.join(real_folder, "pret"),
    #              folder_results=os.path.join(real_folder, "results_both"), save_pret=os.path.join(real_folder, "pret_imgs"),label_folder=None, flip=False)

    # Eval Test
    # real_folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\Test2"
    # if TRAIN_BINARY:
    #     label_folder = os.path.join(real_folder, "labels_bin")
    # else:
    #     label_folder = os.path.join(real_folder, "labels")
    # evaluate_real(model, DEVICE, folder=os.path.join(real_folder, "images"), folder_pt=os.path.join(real_folder, "pret"),
    #               folder_results=os.path.join(real_folder, "results_both"), save_pret=os.path.join(real_folder, "pret_imgs"),
    #               label_folder=label_folder,
    #               flip=False, threshold=0.2, do_flatten_border=True, flatten_line_90=False)

    evaluate_real(model, DEVICE, folder=os.path.join(EVAL_TRAIN, "bild"),
                  folder_pt=os.path.join(EVAL_TRAIN_RES, "pret"),
                  folder_results=os.path.join(EVAL_TRAIN_RES, "results_both"),
                  save_pret=os.path.join(EVAL_TRAIN_RES, "pret_imgs"),
                  flip=False, threshold=0.5, do_flatten_border=False, flatten_line_90=False, skip_pp=True)





    # eval_train = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\Eval_Train"
    # evaluate_real(model, DEVICE, folder=os.path.join(eval_train, "images"),
    #               folder_pt=os.path.join(eval_train, "pret"),
    #               folder_results=os.path.join(eval_train, "results_both"),
    #               save_pret=os.path.join(eval_train, "pret_imgs"),
    #               label_folder=os.path.join(eval_train, "labels"),
    #               flip=False, threshold=0.5, do_flatten_border=True, flatten_line_90=False)
#
    # real_folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\EvalF"
#
    # evaluate_real(model, DEVICE, folder=os.path.join(real_folder, "images"),
    #               folder_pt=os.path.join(real_folder, "pret"),
    #               folder_results=os.path.join(real_folder, "results_both"),
    #               save_pret=os.path.join(real_folder, "pret_imgs"),
    #               label_folder=None,
    #               flip=False, threshold=0.5, do_flatten_border=True, flatten_line_90=False)

    # real_folder = os.path.join("EvaluateReal", "LotsWithMarks")
# 
# evaluate_real(model, DEVICE, os.path.join(real_folder, "images"), os.path.join(real_folder, "pret"),
#               os.path.join(real_folder, "results_both"), os.path.join(real_folder, "pret_imgs"), flip=False)
# 
# real_folder = os.path.join("EvaluateReal", "ImagesOld")
# 
# evaluate_real(model, DEVICE, os.path.join(real_folder, "images"), os.path.join(real_folder, "pret"),
#               os.path.join(real_folder, "results_both"), os.path.join(real_folder, "pret_imgs"), flip=False)
# print("Predicting Test Set")
# predict_all(net, DEVICE)
