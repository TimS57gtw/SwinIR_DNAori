import os
import shutil
import time
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from utils import utils_option as option
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_image as util
from utils import utils_logger
import logging
import random
import torch
import copy
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from models.model_plain import ModelPlain as M
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def estimate_folder(opt, model, infld, outfld, statdf, border, step, comp=True, res=True):

    timimgs = {}

    def save_timings():
        ml = max([len(timimgs[x]) for x in timimgs.keys()])
        for k in timimgs.keys():
            while len(timimgs[k]) < ml:
                timimgs[k].append(0)
        df = pd.DataFrame(timimgs)
        df.to_csv(os.path.join(outfld, f"timings_{step}.csv"))
    def append_timings(stat, val):
        if stat not in timimgs.keys():
            timimgs[stat] = []
        timimgs[stat].append(val)

    start = time.perf_counter()
    dataset_opt = opt['datasets']['test']
    dataset_opt['dataroot_L'] = os.path.join(infld, 'NS_npy')
    if os.path.isdir(os.path.join(infld, 'GT')):
        dataset_opt['dataroot_H'] = os.path.join(infld, 'GT_npy')
    else:
        print("Setting H Data equal to L")
        dataset_opt['dataroot_H'] = dataset_opt['dataroot_L']

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    append_timings("Testset_generation", time.perf_counter() - start)
    start = time.perf_counter()
    idx = 0
    imns = []
    psnes = []
    ssims = []
    mses = []

    for test_data in tqdm(test_loader, desc=f'Testing step {step}', leave=True, position=0):
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)
        resdir = os.path.join(outfld, "indivImgs", os.path.basename(img_name))
        os.makedirs(resdir, exist_ok=True)
        if comp:
            resC = os.path.join(resdir, "comp")
            os.makedirs(resC, exist_ok=True)
        if res:
            resR = os.path.join(resdir, "residual")
            os.makedirs(resR, exist_ok=True)
        resI = os.path.join(resdir, 'Predict')
        os.makedirs(resI, exist_ok=True)
        hrI = os.path.join(resdir, "GT.png")
        lrI = os.path.join(resdir, "NS.png")
        if not os.path.isfile(hrI):
            shutil.copy(test_data['H_path'][0], hrI)
        if not os.path.isfile(lrI):
            shutil.copy(test_data['L_path'][0], lrI)

        append_timings("ObtainData", time.perf_counter() - start)
        start = time.perf_counter()

        model.feed_data(test_data)

        append_timings("FeedData", time.perf_counter() - start)
        start = time.perf_counter()

        model.test()

        append_timings("Test", time.perf_counter() - start)

        start = time.perf_counter()

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

        append_timings("Extractimg", time.perf_counter() - start)
        start = time.perf_counter()

        save_img_name = '{:s}_{:d}.png'.format(img_name, step)
        util.imsave(E_img, os.path.join(resI, save_img_name))

        append_timings("SaveP", time.perf_counter() - start)
        start = time.perf_counter()

        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
        ssim = util.calculate_ssim(E_img, H_img, border=border)
        mse = util.calculate_mse(E_img, H_img, border=border)

        append_timings("CalcMetrics", time.perf_counter() - start)
        start = time.perf_counter()

        imns.append(img_name)
        psnes.append(current_psnr)
        ssims.append(ssim)
        mses.append(mse)

        append_timings("AppMetrics", time.perf_counter() - start)
        start = time.perf_counter()

        if res:
            plt.close()
            plt.cla()
            plt.clf()
            plt.imshow(diff, cmap='seismic')
            plt.colorbar()
            plt.title(f"PSNR: {current_psnr:.3f}dB")
            plt.savefig(os.path.join(resR, '{:s}_{:d}.png'.format(img_name, step)))
            plt.clf()

        append_timings("Residual", time.perf_counter() - start)
        start = time.perf_counter()

        if comp:
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
            plt.savefig(os.path.join(resC, '{:s}_{:d}.png'.format(img_name, step)))
            plt.clf()

        append_timings("Comp", time.perf_counter() - start)
        start = time.perf_counter()

    ap = np.average(psnes)
    ai = np.average(ssims)
    am = np.average(mses)
    mp = np.median(psnes)
    mi = np.median(ssims)
    mm = np.median(mses)
    anz = len(imns)
    dap = np.std(psnes) / np.sqrt(anz)
    dai = np.std(ssims) / np.sqrt(anz)
    dam = np.std(mses) / np.sqrt(anz)
    dmp = dap / np.sqrt(np.pi * (2 * anz + 1) / (4 * anz))
    dmi = dai / np.sqrt(np.pi * (2 * anz + 1) / (4 * anz))
    dmm = dam / np.sqrt(np.pi * (2 * anz + 1) / (4 * anz))

    imns.append("average")
    psnes.append(ap)
    ssims.append(ai)
    mses.append(am)
    imns.append("D_avg")
    psnes.append(dap)
    ssims.append(dai)
    mses.append( dam)
    imns.append("Median")
    psnes.append(mp)
    ssims.append(mi)
    mses.append( mm)
    imns.append("D_med")
    psnes.append(dmp)
    ssims.append(dmi)
    mses.append( dmm)

    if 'ImageName' not in statdf.keys():
        statdf['ImageName'] = imns

    statdf[f"PSNE_{step}"] = psnes
    statdf[f"SSIM_{step}"] = ssims
    statdf[f"MSE_{step}"] = mses

    append_timings("Stats", time.perf_counter() - start)
    start = time.perf_counter()

    save_timings()


def load_model(opt, E_pth, G_pth, O_pth):

    opt['path']['pretrained_netG'] = G_pth
    opt['path']['pretrained_netE'] = E_pth
    opt['path']['pretrained_optimizerG'] = O_pth
    model = M(opt)
    model.init_train()
    return model


def eval_model_series(modelfld, json_path, infld, outfld, comp=True, res=True, only_large=False):
    files  = list(os.listdir(modelfld))
    parts = [int(fl.split('_')[0].strip()) for fl in files]
    steps = np.unique(parts)

    os.makedirs(outfld, exist_ok=True)
    stats = pd.DataFrame()

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

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if only_large:
        steps = sorted(steps)
        steps = [steps[-1]]

    steps = sorted(steps, reverse=True)
    steps = steps[::3]

    for step in tqdm(steps, desc='Testing Models'):
        E_pth = os.path.join(modelfld, f"{step}_E.pth")
        G_pth = os.path.join(modelfld, f"{step}_G.pth")
        O_pth = os.path.join(modelfld, f"{step}_optimizerG.pth")

        model = load_model(opt, E_pth, G_pth, O_pth)

        estimate_folder(opt=opt, model=model, infld=infld, outfld=outfld, statdf=stats, border=border, step=step,
                        comp=comp, res=res)

    stats.to_csv(os.path.join(outfld, 'stats.csv'))


def find_json(mp):
    if len(os.listdir(mp)) == 1:
        return os.path.join(mp, list(os.listdir(mp))[0])

    ndt = []

    for elem in os.listdir(mp):
        parts = elem.split("_")
        date = parts[-2]
        time = parts[-1].split('.')[0]

        ndt.append( (elem, int(date), int(time)))

    ndt = sorted(ndt, key=lambda k : k[1])
    newdate = ndt[-1][1]

    keeps = []
    for elem in ndt:
        if elem[1] == newdate:
            keeps.append(elem)

    keeps = sorted(keeps, key = lambda k : k[2])

    name = keeps[-1][0]

    print("Out of ", list(os.listdir(mp)), ' selected ', name)
    return os.path.join(mp, name)

def initial_metrics(oufld):

    ims = os.path.join(oufld, 'indivImgs')
    files = list(os.listdir(ims))
    stat = {'img': [],
            'psnr': [],
            'ssim': [],
            'mse': []}

    for file in tqdm(files):
        gt = np.array(Image.open(os.path.join(ims, file, "GT.png")))[:, :, 0]
        ns = np.array(Image.open(os.path.join(ims, file, "NS.png")))[:, :, 0]

        plt.switch_backend('TkAgg')
        current_psnr = util.calculate_psnr(ns, gt, border=0)
        ssim = util.calculate_ssim(ns, gt, border=0)
        mse = util.calculate_mse(ns, gt, border=0)

       # fix, axs = plt.subplots(1, 2)
       # axs[0].imshow(gt, cmap='gray')
       # axs[1].imshow(ns, cmap='gray')
       # plt.suptitle(f"MSE: {mse:.3}, PSNR: {current_psnr:.3}, SSIM: {ssim:.3}")
       # plt.show()

        stat['img'].append(file)
        stat['psnr'].append(current_psnr)
        stat['ssim'].append(ssim)
        stat['mse'].append(mse)

    df = pd.DataFrame(stat)
    df.to_csv(os.path.join(oufld, 'initial_img_stats.csv'))



if __name__ == "__main__":
    train_scenario = "ImagesV2"
    # modelfld = r'D:\Dateien\KI_Speicher\SwinSTM_Denoise\Runs\swinir_sr_classical_patch64_x1_N65'
    # modelfld = r'D:\Dateien\KI_Speicher\SwinSTM_Denoise\Runs\swinir_sr_classical_patch64_x1_N65_LVL_MM_NP_LargeHS'
    modelfld = r'D:\Dateien\KI_Speicher\SwinSTM_Denoise\Runs\swinir_N65_npy_W8_H64_C3'
    # modelfld = r'D:\Dateien\KI_Speicher\SwinSTM_Denoise\Runs\swinir_sr_classical_patch64_x1_N65_LVL_MM_NP_LargeHS_WS16'
    modelfld = os.path.join(modelfld, 'models')
    outfld = os.path.join(r'D:\Dateien\KI_Speicher\SwinSTM_Denoise\Tests', train_scenario, os.path.basename(os.path.dirname(modelfld)))
    os.makedirs(outfld, exist_ok=True)
    json_path = find_json(os.path.join(os.path.dirname(modelfld), 'options'))
    # infld = r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Graphite_V1\Test'
    infld = r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\ImagesV2\Test'
    # infld = r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Ph5\Test'


    #
    # assert 4 == 5
    eval_model_series(modelfld, json_path, infld, outfld, comp=True, res=True, only_large=True)
    initial_metrics(outfld)

