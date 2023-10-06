import copy
import os.path
import math
import argparse
import time
import random

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from tqdm import tqdm
import wandb
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''


    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.init(project=f"SwinIR_{opt['task']}_x{opt['scale']}_{os.path.basename(opt['datasets']['train']['dataroot_H'])}_{os.path.basename(opt['datasets']['train']['dataroot_L']) if opt['scale'] != 1 else 'DN'}", config=opt)
    wandb.log({"dataset": opt['datasets']['train']['dataroot_H']})


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    lastspeed = 0

    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, train_data in pbar:

            current_step += 1
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            start = time.perf_counter()
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            # wandb.log({'speed': time.perf_counter() - start})
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, g-loss:{:.3e}> '.format(epoch, current_step, model.current_learning_rate(), logs['G_loss'])
                pbar.set_description(message)
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.log(level=1, msg=message)

                wandb.log({'G-Loss': logs['G_loss']})
                wandb.log({'epoch': epoch})
                wandb.log({'duration': (time.perf_counter() - start)})




            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if i == 0:
                if epoch % 10 == 0:
                    model.plot(epoch)
            # -------------------------------
            # 6) testing
            # -------------------------------
            if True and current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
            # if True and current_step % 250 == 0:

                avg_psnr = 0.0
                idx = 0
                avg_ssim = 0.0
                avg_mse = 0.0

                stat_fld = os.path.join(os.path.dirname(opt['path']['images']), 'stats')
                os.makedirs(stat_fld, exist_ok=True)
                statfile = os.path.join(stat_fld,  '{:d}.csv'.format(current_step))


                with open(statfile, 'w') as f:

                    f.write(f"Img;PSNR;SSIM;MSE\n")

                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)

                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)
                        # print("ImgDir: ", img_dir)

                        model.feed_data(test_data)
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

                        res_fld = os.path.join(img_dir, 'residual')
                        comp_fld = os.path.join(img_dir, 'comp')
                        os.makedirs(res_fld, exist_ok=True)
                        os.makedirs(comp_fld, exist_ok=True)




                       # print("E: ", E_img.shape)
                       # print("H: ", H_img.shape)


                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.imsave(E_img, save_img_path)

                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                        ssim = util.calculate_ssim(E_img, H_img, border=border)
                        mse = util.calculate_mse(E_img, H_img, border=border)
                        f.write(f"{img_name};{current_psnr};{ssim};{mse}\n")





                        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))


                        if False and current_step % 500 * opt['train']['checkpoint_test'] == 0:
                            plt.close()
                            plt.cla()
                            plt.clf()
                            plt.imshow(diff, cmap='seismic')
                            plt.colorbar()
                            plt.title(f"PSNR: {current_psnr:.3f}dB")
                            plt.savefig(os.path.join(res_fld, '{:s}_{:d}.png'.format(img_name, current_step)))
                            plt.clf()




                            # fig, axs = plt.subplots(2, 2)
        #
                            # axs[0, 0].imshow(L_fl, cmap='gray', vmax=255, vmin=0)
                            # axs[0, 0].set_title("L")
                            # axs[0, 0].axis('off')
                            # axs[0, 1].imshow(E_fl, cmap='gray', vmax=255, vmin=0)
                            # axs[0, 1].set_title("E")
                            # axs[0, 1].axis('off')
                            # axs[1, 0].imshow(H_fl, cmap='gray', vmax=255, vmin=0)
                            # axs[1, 0].set_title("H")
                            # axs[1, 0].axis('off')
                            # axs[1, 1].imshow(diff, cmap='seismic')
                            # axs[1, 1].set_title("diff")
                            # axs[1, 1].axis('off')
                            # plt.title(f"PSNR: {current_psnr:.3f}dB")
                            # plt.savefig(os.path.join(comp_fld, '{:s}_{:d}.png'.format(img_name, current_step)))
                            # plt.clf()
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
                            plt.savefig(os.path.join(comp_fld, '{:s}_{:d}.png'.format(img_name, current_step)))
                            plt.clf()
                        avg_psnr += current_psnr
                        avg_ssim += ssim
                        avg_mse += mse


                combine_stats(stat_fld)
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_mse = avg_mse  / idx
                wandb.log({'V-E-Img': E_img, 'V-H-Img': H_img, 'T-PSNR': avg_psnr, 'T-SSIM': avg_ssim, 'T-MSE': avg_mse})
                print({'V-E-Img': E_img, 'V-H-Img': H_img, 'T-PSNR': avg_psnr, 'T-SSIM': avg_ssim, 'T-MSE': avg_mse})

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))


def combine_stats(fld):
    d = {}
    keys = []
    for file in os.listdir(fld):
        step = int(file.split('.')[0])
        keys.append(step)
        with open(os.path.join(fld, file)) as f:
            num = 0
            mse = 0
            ssim = 0
            psnr = 0
            for line in f:
                if "PSNR" in line:
                    continue
                num += 1
                parts = line.split(';')
                psnr += float(parts[1])
                ssim += float(parts[2])
                mse += float(parts[3])
            if num != 0:
                psnr /= num
                ssim /= num
                mse /= num
                d[step] = (psnr, ssim, mse)
            else:
                d[step] = (0, 0, 0)

    steps = []
    psnrs = []
    ssims = []
    mses = []
    with open(os.path.join(os.path.dirname(fld), 'stats.csv'), 'w') as f:
        f.write("Step;PSNR;SSIM;MSE\n")
        for key in sorted(keys):
            steps.append(key)
            psnrs.append(d[key][0])
            ssims.append(d[key][1])
            mses.append(d[key][2])
            f.write(f"{key};{d[key][0]};{d[key][1]};{d[key][2]}\n")

    plt.plot(steps, psnrs)
    plt.title("PSNR")
    plt.savefig(os.path.join(os.path.dirname(fld), 'psnr.png'))
    plt.clf()
    plt.plot(steps, ssims)
    plt.title("SSIM")
    plt.savefig(os.path.join(os.path.dirname(fld), 'ssim.png'))
    plt.clf()
    plt.plot(steps, mses)
    plt.title("MSE")
    plt.savefig(os.path.join(os.path.dirname(fld), 'mse.png'))
    plt.clf()



if __name__ == '__main__':
    json_path = "options/swinir/train_swinir_sr_classical.json"
    # json_path = "options/train_dncnn.json"
    # json_path = "options/swinir/train_swinir_denoising_gray.json"

    main(json_path)
