import copy

import matplotlib.pyplot as plt

import wandb
from pprint import pprint
import Preprocessing
import os
import datetime
from pathlib import Path
from MyUNet.Pytorch_UNet.utils.data_loading import BasicDataset, EvalDataset
from MyUNet.Pytorch_UNet.utils.dice_score import dice_loss
from MyUNet.Pytorch_UNet.evaluate import evaluate, evaluateMWL
from MyUNet.Pytorch_UNet.unet import UNet
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch
import logging
import torch.nn as nn
import numpy as np
import scipy
import torch.nn.functional as F
from CustomLoss import *
from tqdm import tqdm
from dataclasses import dataclass
CLUSTER = False


def validate_model(model, bilinear):

    TRAIN_SET = "62nm_10k_64px"
    TRAIN_SET_REAL = "Real400"
    set_short = TRAIN_SET
    set_short_REAL = TRAIN_SET_REAL
    train_set = os.path.join("SS_DNA_Train", TRAIN_SET)
    train_set_REAL = os.path.join("SS_DNA_Train", TRAIN_SET_REAL)
    RES_BP = r'C:\Users\seifert\PycharmProjects\DNA_Measurement\SS_DNA_Train\EvalNetworks'
    threads = 10
    run_name="P_"

    img_scale = 1
    val_percent = 0.15
    batch_size = 32
    MAX_IMAGES = None
    loaded_epochs = 0
    LOG_STEPS = 10
    VALIDATION_STEP = 4  # 1 means 10 per epoch
    save_checkpoint = 1
    amp = True
    DEVICE = torch.device("cuda:0")
    assert torch.cuda.is_available()

    IMAGE_SET = os.path.join(train_set, "Train")
    TEST_SET = os.path.join(train_set, "Test")
    PRET_IMG = os.path.join(train_set, "Train_pt")
    PRET_TEST = os.path.join(train_set, "Test_pt")
    IMAGE_SET_REAL = os.path.join(train_set_REAL, "Train")
    TEST_SET_REAL = os.path.join(train_set_REAL, "Test")
    PRET_IMG_REAL = os.path.join(train_set_REAL, "Train_pt")
    PRET_TEST_REAL = os.path.join(train_set_REAL, "Test_pt")
    losses = []

    date = datetime.datetime.now()
    kls = date.strftime("%m_%d___%H_%M_%S")
    run_name += kls

    images = os.path.join(IMAGE_SET, "bild")
    labels = os.path.join(IMAGE_SET, "data", "PNG")
    dir_img = Path(images)
    dir_mask = Path(labels)
    dir_img_pret = os.path.join(PRET_IMG, "sxm", "numpy")
    dir_mask_pret = os.path.join(PRET_IMG, "data", "numpy")
    dir_img_test = os.path.join(TEST_SET, "bild")
    dir_mask_test = os.path.join(TEST_SET, "data", "PNG")
    dir_img_test_pret = os.path.join(PRET_TEST, "sxm", "numpy")
    dir_mask_test_pret = os.path.join(PRET_TEST, "data", "numpy")

    REAL_images = os.path.join(IMAGE_SET_REAL, "bild")
    REAL_labels = os.path.join(IMAGE_SET_REAL, "data", "PNG")
    REAL_dir_img = Path(REAL_images)
    REAL_dir_mask = Path(REAL_labels)
    REAL_dir_img_pret = os.path.join(PRET_IMG_REAL, "sxm", "numpy")
    REAL_dir_mask_pret = os.path.join(PRET_IMG_REAL, "data", "numpy")
    REAL_dir_img_test = os.path.join(TEST_SET_REAL, "bild")
    REAL_dir_mask_test = os.path.join(TEST_SET_REAL, "data", "PNG")
    REAL_dir_img_test_pret = os.path.join(PRET_TEST_REAL, "sxm", "numpy")
    REAL_dir_mask_test_pret = os.path.join(PRET_TEST_REAL, "data", "numpy")

    RES_DIR = os.path.join(RES_BP, set_short, kls)
    dir_checkpoint = Path(RES_DIR, "checkpoints")
    results_dir = os.path.join(RES_DIR, "results")

    net = UNet(n_channels=1, n_classes=3, bilinear=bilinear).to(device=DEVICE)
    net.load_state_dict(torch.load(model))
    net = net.to(device=DEVICE)
    net.eval()

    dataset = BasicDataset(dir_img_pret, dir_mask_pret, 1.0, "_mask", image_num=MAX_IMAGES,
                           noise_level=0)
    dataset_REAL = BasicDataset(REAL_dir_img_pret, REAL_dir_mask_pret, 1.0, "_mask", image_num=MAX_IMAGES,
                                noise_level=0)

    # else:
    #    dataset = BasicDataset(dir_img, dir_mask, 1.0, "_mask")

    # 2. Split into train / validation partitions

    train_set = dataset
    train_set_REAL =  dataset_REAL

    imfs  = os.path.join(RES_DIR, 'image_synth')
    os.makedirs(imfs, exist_ok=True)
    imfr = os.path.join(RES_DIR, 'image_real')
    os.makedirs(imfr, exist_ok=True)
    idx = 0
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=False)
    train_loader_SYN = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    train_loader_REAL = DataLoader(train_set_REAL, shuffle=True, drop_last=False, **loader_args)
    MAXNUM = 100
    with tqdm(total=MAXNUM, desc="Synth") as pbar:
        for batch in train_loader_SYN:
            # SHOW_BACTH = np.random.uniform() < 0.01
            images = batch['image']
            true_masks = batch['mask']

            assert images.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=DEVICE)
            true_masks = true_masks.to(device=DEVICE)

            with torch.cuda.amp.autocast(enabled=amp):
                # print("Images: ", images.device)
                masks_pred = net(images)

            images = images.detach().cpu().numpy()
            plt.imshow(images[0, 0])
            plt.title("Images")
            plt.show()
            true_masks = true_masks.detach().cpu().numpy()
            mp = torch.softmax(masks_pred, dim=1).argmax(dim=1).detach().cpu().numpy()

            # masks_pred = masks_pred.detach().cpu().numpy()

            for i in range(images.shape[0]):

                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(images[i, 0], cmap='gray')
                axs[0].set_title("Image")
                axs[1].imshow(true_masks[i], cmap='gray')
                axs[1].set_title("True")

                axs[2].imshow(mp[i], cmap='gray')
                axs[2].set_title("Pred")
                plt.savefig(os.path.join(imfs, f"Image_{str(idx).zfill(4)}"))
                plt.clf()
                idx += 1
                pbar.update(1)
                if idx >= MAXNUM:
                    break
            if idx >= MAXNUM:
                break

    idx = 0
    with tqdm(desc='Real', total=MAXNUM) as pbar:
        for batchR in tqdm(train_loader_REAL):
            # SHOW_BACTH = np.random.uniform() < 0.01
            images = batchR['image']
            true_masks = batchR['mask']

            assert images.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=DEVICE)
            true_masks = true_masks.to(device=DEVICE)

            with torch.cuda.amp.autocast(enabled=amp):
                # print("Images: ", images.device)
                masks_pred = net(images)

            images = images.detach().cpu().numpy()
            true_masks = true_masks.detach().cpu().numpy()
            mp = torch.softmax(masks_pred, dim=1).argmax(dim=1).cpu().numpy()


            for i in range(images.shape[0]):

                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(images[i, 0], cmap='gray')
                axs[0].set_title("Image")
                axs[1].imshow(true_masks[i], cmap='gray', vmax=2)
                axs[1].set_title("True")

                axs[2].imshow(mp[i], cmap='gray', vmax=2)
                axs[2].set_title("Pred")
                plt.savefig(os.path.join(imfr, f"Image_{str(idx).zfill(4)}"))
                plt.clf()
                idx += 1
                pbar.update(1)
                if idx >= MAXNUM:
                    break
            if idx >= MAXNUM:
                break


def train(config=None):
    passed_config = config is not None

    if CLUSTER:
        TRAIN_SET = "62nm_10k_64px"
        TRAIN_SET_REAL = "Real400"

        set_short = TRAIN_SET
        set_short_REAL = TRAIN_SET_REAL

        train_set = os.path.join(r'/beegfs/work/seifert/DNA_TrainData/2023_07', TRAIN_SET)
        train_set_REAL = os.path.join(r'/beegfs/work/seifert/DNA_TrainData/2023_07', TRAIN_SET_REAL)

        RES_BP = r'/beegfs/work/seifert/DNA_TrainData/2023_07/Res_SynthReal'
        threads = 2
        run_name="C"
    else:
        TRAIN_SET = "Set070923_GV_RelPos"
        TRAIN_SET_REAL = "Real400"
        set_short = TRAIN_SET
        set_short_REAL = TRAIN_SET_REAL
        train_set = os.path.join("SS_DNA_Train", TRAIN_SET)
        train_set_REAL = os.path.join("SS_DNA_Train", TRAIN_SET_REAL)
        RES_BP = r'C:\Users\seifert\PycharmProjects\DNA_Measurement\SS_DNA_Train\Results_SynthReal250'
        threads = 10
        run_name="P_CentMV_"


        set_short = TRAIN_SET
        set_short_REAL = TRAIN_SET_REAL



    img_scale=1
    val_percent = 0.05
    MAX_IMAGES = None
    loaded_epochs = 0
    LOG_STEPS = 10
    VALIDATION_STEP = 4  # 1 means 10 per epoch
    save_checkpoint = 4
    amp = True
    DEVICE = torch.device("cuda:0")
    assert torch.cuda.is_available()


    IMAGE_SET = os.path.join(train_set, "Train")
    TEST_SET = os.path.join(train_set, "Test")
    PRET_IMG = os.path.join(train_set, "Train_pt")
    PRET_TEST = os.path.join(train_set, "Test_pt")

    IMAGE_SET_REAL = os.path.join(train_set_REAL, "Train")
    TEST_SET_REAL= os.path.join(train_set_REAL, "Test")
    PRET_IMG_REAL= os.path.join(train_set_REAL, "Train_pt")
    PRET_TEST_REAL = os.path.join(train_set_REAL, "Test_pt")

    losses = []

    date = datetime.datetime.now()
    kls = date.strftime("%m_%d___%H_%M_%S")
    run_name +=kls

    images = os.path.join(IMAGE_SET, "bild")
    labels = os.path.join(IMAGE_SET, "data", "PNG")
    dir_img = Path(images)
    dir_mask = Path(labels)
    dir_img_pret = os.path.join(PRET_IMG, "sxm", "numpy")
    dir_mask_pret = os.path.join(PRET_IMG, "data", "numpy")
    dir_img_test = os.path.join(TEST_SET, "bild")
    dir_mask_test = os.path.join(TEST_SET, "data", "PNG")
    dir_img_test_pret = os.path.join(PRET_TEST, "sxm", "numpy")
    dir_mask_test_pret = os.path.join(PRET_TEST, "data", "numpy")

    REAL_images = os.path.join(IMAGE_SET_REAL, "bild")
    REAL_labels = os.path.join(IMAGE_SET_REAL, "data", "PNG")
    REAL_dir_img = Path(REAL_images)
    REAL_dir_mask = Path(REAL_labels)
    REAL_dir_img_pret = os.path.join(PRET_IMG_REAL, "sxm", "numpy")
    REAL_dir_mask_pret = os.path.join(PRET_IMG_REAL, "data", "numpy")
    REAL_dir_img_test = os.path.join(TEST_SET_REAL, "bild")
    REAL_dir_mask_test = os.path.join(TEST_SET_REAL, "data", "PNG")
    REAL_dir_img_test_pret = os.path.join(PRET_TEST_REAL, "sxm", "numpy")
    REAL_dir_mask_test_pret = os.path.join(PRET_TEST_REAL, "data", "numpy")

    RES_DIR = os.path.join(RES_BP, set_short, kls)
    dir_checkpoint = Path(RES_DIR, "checkpoints")
    results_dir = os.path.join(RES_DIR, "results")

    with wandb.init(config=config, name=run_name, project=f"ApplyDNAModel"):
       #if not passed_config:
       #     config = wandb.config
        config = wandb.config

        net = UNet(n_channels=1, n_classes=3, bilinear=1 == config.bilinear).to(device=DEVICE)
        net = net.to(device=DEVICE)
        learning_rate = 10**config.learning_rate_exp
        learning_rate_real = 10**config.learning_rate_inc

        config = wandb.config
        Preprocessing.normalize_soft = Preprocessing.normalize_half_sigmoid

        show = False
        enhance_contrast = True
        zfill = 6
        img_size = 64
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
        Preprocessing.pretransform_all(dir_img=REAL_dir_img, dir_mask=REAL_dir_mask, dir_img_test=REAL_dir_img_test,
                                   dir_mask_test=REAL_dir_mask_test, dir_img_pret=REAL_dir_img_pret,
                                   dir_mask_pret=REAL_dir_mask_pret,
                                   dir_img_test_pret=REAL_dir_img_test_pret, dir_mask_test_pret=REAL_dir_mask_test_pret,
                                   show=show, enhance_contrast=enhance_contrast, zfill=zfill,
                                   threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                                   do_flatten=do_flatten,
                                   do_flatten_border=do_flatten_border, flip=flip, flatten_line_90=flatten_line_90,
                                   keep_name=keep_name,
                                   use_Test=use_Test, use_masks=use_masks, resize_method=resize_method)

        dataset = BasicDataset(dir_img_pret, dir_mask_pret, 1.0, "_mask", image_num=MAX_IMAGES, noise_level=config.noise_level)
        dataset_REAL = BasicDataset(REAL_dir_img_pret, REAL_dir_mask_pret, 1.0, "_mask", image_num=MAX_IMAGES, noise_level=config.noise_level)

        # else:
        #    dataset = BasicDataset(dir_img, dir_mask, 1.0, "_mask")

        # 2. Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val

        n_val_REAL = int(len(dataset_REAL) * val_percent)
        n_train_REAL = len(dataset_REAL) - n_val_REAL
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        train_set_REAL, val_set_REAL = random_split(dataset_REAL, [n_train_REAL, n_val_REAL], generator=torch.Generator().manual_seed(0))

        # 3. Create data loaders
        loader_args = dict(batch_size=config.batch_size, num_workers=4, pin_memory=False)
        print("Batch Size: ", config.batch_size)
        val_loader_args = dict(batch_size=min(config.batch_size, int(np.floor(n_val/2))), num_workers=4, pin_memory=False)
        train_loader_SYN = DataLoader(train_set, shuffle=True,drop_last=False, **loader_args)
        val_loader_SYN = DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args)
        train_loader_REAL = DataLoader(train_set_REAL, shuffle=True, drop_last=False, **loader_args)
        val_loader_REAL = DataLoader(val_set_REAL, shuffle=False, drop_last=False, **val_loader_args)

        # (Initialize logging)
        experiment = wandb.init(project="SS_DNA_Test_Real{}".format(REAL), resume='allow', entity="tims57gtw")
        experiment.config.update(dict(epochs_syn=config.epochs_syn, epochs_real=config.epochs_real, batch_size=config.batch_size, learning_rate=learning_rate, learning_rate_real=learning_rate_real,
                                     val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale))

        logging.info(f'''Starting training:
                    Epochs-SYN:       {config.epochs_real}
                    Epochs-REAL:     {config.epochs_syn}
                    Batch size:      {config.batch_size}
                    Learning rate-SYN:   {10**config.learning_rate_exp}
                    Learning rate-REAL:   {10**config.learning_rate_inc}
                    Training size:   {n_train}
                    Validation size: {n_val}
                    Checkpoints:     {save_checkpoint}
                    Device:          {DEVICE} # Device.type
                    Images scaling:  {img_scale}
                    Mixed Precision: {amp}
                ''')

        print("Device: ", next(net.parameters()).device)

        epochs = config.epochs_syn + config.epochs_real
        epochs = max(1, epochs)
        syn_steps = int(np.ceil(len(train_set) * config.epochs_syn / config.batch_size) )
        real_steps = int(np.ceil(len(train_set) * config.epochs_real / config.batch_size))
        total_steps = max(int(len(train_loader_REAL) * config.epochs_real / config.batch_size) + 5, syn_steps + real_steps)

        if config.epochs_syn > 0:
            if config.optimizer == 'adam':
                optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
            elif config.optimizer == 'sgd':
                optimizer = optim.SGD(net.parameters(), lr=learning_rate)
            elif config.optimizer == 'adamw':
                optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-8)
            else:
                raise NotImplementedError()

            if config.scheduler == 'onecycle':
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate, pct_start=0.1,
                                                      total_steps=syn_steps)
            elif config.scheduler == 'linear':
                scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.05, total_iters=syn_steps)
            elif config.scheduler == 'plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
            else:
                raise NotImplementedError()

        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        ce = nn.CrossEntropyLoss()
        ce_max = config.CEmax
        ce_xsc = config.CExsc
        if ce_xsc == 0:
            ce_fak = lambda x : ce_max
        else:
            g = (ce_max * ce_xsc) / (
                        ce_max * np.real(scipy.special.lambertw(-np.exp(-ce_xsc / ce_max) * ce_xsc / ce_max)) + ce_xsc)
            ce_fak = lambda x: g * (1 - np.exp(-ce_xsc * x / g))



        wdl = MyWeightedLoss(classes=3, weights=[0.017974, 0.042809, 0.939218])
#
        criterionMWL = lambda x, y: wdl(F.softmax(x, dim=1).float(),
                                          F.one_hot(y, 3).permute(0, 3, 1, 2).float())
#
        myfb = MyF1Loss(beta=config.beta, classes=3, weights=[0.017974, 0.042809, 0.939218])

        criterion = lambda x, y: myfb(F.softmax(x, dim=1).float(),
                                          F.one_hot(y, 3).permute(0, 3, 1, 2).float())

        def criterion_CECR(masks_pred, true_masks, cef):
            cr_loss = criterionMWL(masks_pred, true_masks)
            if cef != 0:
                ce_loss = ce(masks_pred, true_masks)
            else:
                ce_loss = torch.from_numpy(np.array([0])).to(device=DEVICE)

            loss = (1 - cef) * cr_loss + cef * ce_loss
            return loss, ce_loss, cr_loss



        global_step = 0
        # total_steps = int(len(train_set) * config.epochs / config.batch_size)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']



        # 5. Begin training
        with torch.cuda.device(0):
            for epoch in range(1, epochs + 1):
                train_loader = train_loader_SYN if epoch < config.epochs_syn else train_loader_REAL
                n_train_loc = n_train if epoch < config.epochs_syn + 1 else n_train_REAL
                if epoch == config.epochs_syn + 1:
                    print("Switch to real")
                    if config.optimizer == 'adam':
                        optimizer = optim.Adam(net.parameters(), lr=learning_rate_real, weight_decay=1e-8)
                    elif config.optimizer == 'sgd':
                        optimizer = optim.SGD(net.parameters(), lr=learning_rate_real)
                    elif config.optimizer == 'adamw':
                        optimizer = optim.AdamW(net.parameters(), lr=learning_rate_real, weight_decay=1e-8)
                    else:
                        raise NotImplementedError()

                    if config.scheduler == 'onecycle':
                        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate_real, pct_start=0.1,
                                                                  total_steps=real_steps)
                    elif config.scheduler == 'linear':
                        scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.05,
                                                                total_iters=real_steps)
                    elif config.scheduler == 'plateau':
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
                    else:
                        raise NotImplementedError()

                net.train()
                epoch_loss = 0
                ep_losses = []
                sizes = []
                with tqdm(total=n_train_loc, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                    # print("Train Loader", "train_loader_SYN" if epoch < config.epochs_syn else "train_loader_REAL")
                    # print("Length Syn: ", len(train_loader_SYN))
                    # print("Length Real: ", len(train_loader_REAL))
                    # print("Length: ", len(train_loader))

                    for batch in train_loader:
                        # SHOW_BACTH = np.random.uniform() < 0.01
                        images = batch['image']
                        true_masks = batch['mask']

                        assert images.shape[1] == net.n_channels, \
                            f'Network has been defined with {net.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        images = images.to(device=DEVICE)
                        true_masks = true_masks.to(device=DEVICE)

                        with torch.cuda.amp.autocast(enabled=amp):
                            # print("Images: ", images.device)
                            masks_pred = net(images)

                            cef = ce_fak(global_step / total_steps)




                            loss, ce_loss,cr_loss  = criterion_CECR(masks_pred, true_masks, cef)


                        #
                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                        pbar.update(images.shape[0])
                        global_step += 1
                        epoch_loss += loss.item()
                        losses.append((global_step, loss.item()))

                        wandb.log({
                            'train_loss': loss.item(),
                            'loss': ce_loss.item() * cr_loss.item() / (ce_loss.item() + cr_loss.item()),
                            'CE loss': ce_loss.item(),
                            'CR loss': cr_loss.item(),
                            'CE factor': cef,
                            'step': global_step,
                            'epoch': epoch,
                            'lr': get_lr(optimizer=optimizer),
                            'set':  0 if epoch < config.epochs_syn else 1

                        })
                        if config.scheduler != 'plateau':
                            scheduler.step()
                        else:
                            scheduler.step(loss)
                        if np.isnan(loss.item()):
                            print("NaN Loss detected")
                            return
                        pbar.set_postfix(**{'loss (batch)': loss.item(), 'step': global_step})


                # if epoch == epochs or epochs < 10 or epochs % 2 == 1:
                if True:
                    try:  # experiment.
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        crit = lambda a, b: criterion_CECR(a, b, cef)[0]

                        val_WDL_REAL, val_Dice_REAL, val_f2_REAL, val_loss_REAL = evaluateMWL(net, val_loader_REAL, DEVICE, crit)
                        val_WDL_SYNTH, val_Dice_SYNTH, val_f2_SYNTH, val_loss_SYNTH = evaluateMWL(net, val_loader_SYN, DEVICE, crit)



                    # scheduler.step(val_score)

                    # logging.info('Validation Dice score: {}'.format(val_Dice))

                        wandb.log({
                            'valDICE_SYN': val_Dice_SYNTH,
                            'valDICE_REAL': val_Dice_REAL,
                            'valF2_REAL': val_f2_REAL,
                            'valMWL_SYN': val_WDL_SYNTH,
                            'valF2_SYN': val_f2_SYNTH,
                            'valMWL_REAL': val_WDL_REAL,
                            'val_loss_REAL': val_loss_REAL,
                            'val_loss_SYNTH': val_loss_SYNTH,

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
                    except ValueError:
                        logging.warning("Infinite Value in histogram")
                        pass

                if epoch % save_checkpoint == 0 or (save_checkpoint < 100 and epoch == epochs - 1):
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    torch.save(net.state_dict(),
                               str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + loaded_epochs)))
                    logging.info(f'Checkpoint {epoch + loaded_epochs} saved!')
                    # print("Saved ", str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + loaded_epochs)))



@dataclass
class TestConfig:
    optimizer: str
    scheduler: str
    CEmax: float
    CExsc: float
    beta: float
    epochs_syn: int
    epochs_real: int
    learning_rate_exp: float
    learning_rate_inc: float
    noise_level: float
    bilinear: int
    batch_size: int

if __name__ == "__main__":

    # validate_model(r"C:\Users\seifert\PycharmProjects\DNA_Measurement\Models\SweepTest\Synth0730_095058_40.pth", True)
    # assert 3 == 4


    sweep_config = {
        'method': 'bayes'
    }

    metric = {
        'name': 'train_loss',
        'goal': 'minimize'
    }

    REAL = False
    parameters_dict = {
            'optimizer': {
                'values': ['adam', 'sgd', 'adamw']
            },
            'scheduler': {
                'values': ['linear', 'onecycle', 'plateau']
            },
            'CEmax': {
                'values': [1e-8, 0.2, 0.4, 0.6, 0.8, 1-(1e-8)]
            },
            'CExsc': {
                'values': [1, 2, 3, 4, 5]
            },
            'epochs_syn': {
                'values': [2, 4, 6, 10, 20, 30, 40]
            },
            'epochs_real': {
                'value': 0
            },
            'learning_rate_exp': {
                'distribution': 'uniform',
                'min': -4,
                'max': -1
            },
            'beta': {
                'values': [1, 2, 4]
            },
            'learning_rate_inc': {
                'distribution': 'uniform',
                'min': -4,
                'max': 1
            },
            'noise_level': {
                'distribution': 'uniform',
                'min': 0,
                'max': 0.3
            },
            'bilinear': {
                'values': [1, 0]
            },
            'batch_size': {
                'values': [8, 16, 32, 64]
            }


    }
    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric
    os.system("wandb enabled")
    # pprint(parameters_dict)
    # sweep_id = wandb.sweep(sweep_config, project="SweepSS_DNA_Synth_Fb")
    # sweep_id = "tims57gtw/SweepSS_DNA_Synth_Fb/gibzjl2t"
     #wandb.agent(sweep_id, train)

    # sweep_id = "tims57gtw/SweepSS_DNA_Mix/l35cvxab"
    # wandb.agent(sweep_id, train)
    #class TestConfig:
    # optimizer: str
    # scheduler: str
    # CEmax: float
    # CExsc: float
    # epochs_syn: int
    # epochs_real: int
    # learning_rate_exp: float
    # noise_level: float
    # bilinear: int
    # batch_size: int
     #assert 1 == 2
    testcnf64 = TestConfig(optimizer='adamw',
                         scheduler='linear',
                         CEmax=0.4,
                         CExsc=4,
                         epochs_syn=40,
                         epochs_real=0,
                         beta=2,
                         learning_rate_exp=-2.8816,
                         learning_rate_inc=-0.15329416509998994,
                         noise_level=0.1395903,
                         bilinear=0,
                         batch_size=64
                         )

    testcnf = TestConfig(optimizer='adamw',
                         scheduler='linear',
                         CEmax=0.75,
                         CExsc=0,
                         epochs_syn=50,
                         beta=2,
                         epochs_real=0,
                         learning_rate_exp=-4.20,
                         learning_rate_inc=-0.15329416509998994,
                         noise_level=0.1395903,
                         bilinear=0,
                         batch_size=8
                         )



    train(testcnf)

