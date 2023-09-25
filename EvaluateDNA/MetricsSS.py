import random
import sys
from itertools import product
import os
import time

from EvaluationChain import preprocess_origami, get_semantic_seg, extract_gsc_ss
from tqdm import tqdm
import Preprocessing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sklearn.metrics as skm
def predict_ss(infld, outfld, ss_model, skip_pp=False):
    IMAGE_SIZE = 50 if OLD_MODEL else 64
    png_path = os.path.join(infld, "bild")
    npy_res = os.path.join(outfld, "pp", "npy")
    png_pp = os.path.join(outfld, "pp", "png")
    ss_res = os.path.join(outfld, "ss", 'col')
    ss_gsc = os.path.join(outfld, "ss", 'gsc')
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(npy_res, exist_ok=True)
    os.makedirs(png_pp, exist_ok=True)
    os.makedirs(ss_res, exist_ok=True)
    os.makedirs(ss_gsc, exist_ok=True)

     # tempfolder, resf_npy, resf_png,
    dir_img = png_path
    dir_img_pret = npy_res
    dir_mask = os.path.join(infld, 'data', 'png')
    dir_img_test = None
    dir_mask_test = None
    dir_mask_pret = os.path.join(outfld, "mask", "npy")
    dir_img_test_pret = None
    dir_mask_test_pret = None
    show = False
    enhance_contrast = True
    zfill = 4
    threads = THREADS
    img_size = IMAGE_SIZE
    overwrite = True
    lines_corr = False
    do_flatten = False
    flip = False
    flatten_line_90 = False
    do_flatten_border = True
    use_masks = True
    use_Test = False
    keep_name = True
    skip_all = False if skip_pp is None else skip_pp
    print("Skip All: ", skip_all )
    resize_method = "bilinear"  # bilinear
    Preprocessing.pretransform_all(dir_img=dir_img, dir_mask=dir_mask, dir_img_test=dir_img_test,
                                   dir_mask_test=dir_mask_test, dir_img_pret=dir_img_pret, dir_mask_pret=dir_mask_pret,
                                   dir_img_test_pret=dir_img_test_pret, dir_mask_test_pret=dir_mask_test_pret,
                                   show=show, enhance_contrast=enhance_contrast, zfill=zfill,
                                   threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                                   do_flatten=do_flatten,
                                   do_flatten_border=do_flatten_border, flip=flip, flatten_line_90=flatten_line_90,
                                   keep_name=keep_name,
                                   use_Test=use_Test, use_masks=use_masks, resize_method=resize_method,
                                   skip_all=skip_all)
    Preprocessing.png_folder(npy_res, png_pp)
    Preprocessing.png_folder(os.path.join(outfld, "mask", "npy"), os.path.join(outfld, "mask", "png"), vmax=2)




    get_semantic_seg(ss_model, npy_res, ss_res, oldModel=OLD_MODEL, bilinear=BILINEAR)
    extract_gsc_ss(ss_res, ss_gsc)

    return ss_gsc


def load_maps(tar_fld, pred_fld):
    labels = [os.path.join(tar_fld, x) for x in os.listdir(tar_fld)]
    preds = [os.path.join(pred_fld, x) for x in os.listdir(pred_fld)]
    image_size = 64 if not OLD_MODEL else 50
    assert len(labels) == len(preds)

    tar = np.zeros((len(labels), image_size**2, 3))
    prd = np.zeros((len(labels), image_size**2, 3))
    def idx(n):
        if n == 0:
            return 0
        if 120 < n < 130:
            return 1
        if n == 255:
            return 2

        print("Invalid n: ", n)
        assert 0 == 1
        raise Exception

    for i in tqdm(range(len(labels))):
        lim = Image.open(labels[i])
        larr = np.array(lim)[:, :, 0]
        larr = larr.flatten()
        # print("Larr, ", np.unique(larr))
        for k in range(len(larr)):
            tar[i, k, idx(larr[k])] = 1

        pim = Image.open(preds[i])
        parr = np.array(pim)[:, :, 0]
        # print("Parr, ", np.unique(parr))
        parr = parr.flatten()
        for k in range(len(parr)):
            prd[i, k, idx(parr[k])] = 1

        print()


    return tar, prd


def visualize_pairs(imgs, labels, preds, outfld, titles=None):

    def calcdiff(lbl, pred):
        res = 127 * np.ones((lbl.shape[0], lbl.shape[1], 3), dtype=int)
        for i in range(lbl.shape[0]):
            for j in range(lbl.shape[1]):
                l = int(round(lbl[i, j]/127))
                p = int(round(pred[i, j]/127))
                if l == p:
                    continue
                res[i, j, 2-l] += 63 * (p - l)

        return res

    images = [os.path.join(imgs, x) for x in os.listdir(imgs)]
    labels = [os.path.join(labels, x) for x in os.listdir(labels)]
    preds = [os.path.join(preds, x) for x in os.listdir(preds)]
    if titles is not None:
        assert len(titles) == len(preds)
    resf = os.path.join(outfld, "Comparison")
    os.makedirs(resf, exist_ok=True)
    assert len(images) == len(labels)
    assert len(images) == len(preds)

    for i in tqdm(range(len(images)), desc="Visualize"):
        if random.uniform(0, 1) < 0.82:
            continue
        fix, axes = plt.subplots(2, 2)

        im1 = np.array(Image.open(images[i]))[:, :, 0]
        axes[0, 0].imshow(im1, cmap='gray')
        axes[0, 0].set_title("image")
        im2 = np.array(Image.open(labels[i]))[:, :, 0]
        axes[0, 1].imshow(im2, cmap='gray', vmax=255)
        axes[0, 1].set_title("label")
        im3 = np.array(Image.open(preds[i]))[:, :, 0]
        axes[1, 0].imshow(im3, cmap='gray', vmax=255)
        axes[1, 0].set_title("pred")
        im4 = calcdiff(im2, im3).astype(int)
        axes[1, 1].imshow(im4)
        axes[1, 1].set_title("diff")

        head = titles[i] if titles is not None else ""
        plt.tight_layout()
        plt.suptitle(f"{os.path.basename(images[i])} -> {head}")
        # plt.show(block=True)
        plt.savefig(os.path.join(resf, f"Comp_{str(i).zfill(6)}.png"))
        plt.cla()

def compute_metrics(tar, prd, outf):

    # SKlearn Version:
    #print(tar.shape)
    #print(prd.shape)
    assert tar.shape == prd.shape
    # accessed = []
    sklvTAR = np.zeros(tar.shape[0] * tar.shape[1])
    sklvPRD = np.zeros(tar.shape[0] * tar.shape[1])

    for i in range(tar.shape[0]):
        for j in range(tar.shape[1]):
            #print(i, j, tar[i, j])
            if 1 not in tar[i, j]:
                print(tar)
            # if tar[i, j, 2] == 1:
            #     print(i, j, tar)
            if prd[i, j, 2] == 1:
                # print(i, j, prd)
                both = tar[i, j, 2] == 1
                if both:
                    pass

            # if tar.shape[1] * i + j in accessed:
            #     print(i, j, accessed)
            # accessed.append(tar.shape[1] * i + j)
            sklvTAR[tar.shape[1] * i + j] = np.argwhere(tar[i, j, :] == 1)[0][0]
            sklvPRD[tar.shape[1] * i + j] = np.argwhere(prd[i, j, :] == 1)[0][0]


    # Per Class Confusion:
    with open(outf, 'w') as resf:
        resf.write("Class;TarElems;PrdElems;TP;TN;FP;FN;Acc;P;R;F1;TPR;TNR;FPR;BalAcc;SK-IoU;SK-Rec;SK-Pre;SK-F1\n")



        jac_cls = skm.jaccard_score(sklvTAR, sklvPRD, average=None)
        jac_w = skm.jaccard_score(sklvTAR, sklvPRD, average='weighted')
        rec = skm.recall_score(sklvTAR, sklvPRD, average=None)
        prec = skm.precision_score(sklvTAR, sklvPRD, average=None)
        f1sk = skm.f1_score(sklvTAR, sklvPRD, average=None)

        # print("Jac CLS: ", jac_cls)
        # print("Jac W: ", jac_w)
        # print("Rec: ", rec)


        for cls in range(tar.shape[2]):
            tmat = tar[:, :, cls]
            pmat = prd[:, :, cls]

            tp = tmat * pmat
            tn = (1-tmat) * (1-pmat)
            fp = (1-tmat) * pmat
            fn = tmat * (1-pmat)

            tp = np.sum(tp)
            tn = np.sum(tn)
            fp = np.sum(fp)
            fn = np.sum(fn)

            total = tp + tn + fp + fn
            assert total == tar.shape[0] * tar.shape[1]

            tar_elems = np.sum(tmat)
            prd_elems = np.sum(pmat)
            accuracy = tp / total
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            fpr = fp / (fp + tn)
            bal_acc = (tpr + tnr) / 2

            resf.write(f"{cls};{tar_elems};{prd_elems};{tp};{tn};{fp};{fn};{accuracy};{precision};{recall};{f1};{tpr};{tnr};{fpr};{bal_acc};{jac_cls[cls]};{rec[cls]};{prec[cls]};{f1sk[cls]}\n")





THREADS=20
OLD_MODEL = False
if __name__ == "__main__":
    os.system("wandb disabled")
    infld = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\SS\Synth\Set10'
    # infld = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\SS\Real\Real'
    num = len(os.listdir(r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\SS\Synth\Res\ModelComp'))

    ms = (r'Models/SweepTest/Synth0730_095058_40.pth', False, True)
    mr = (r'Models/SweepTest/Real0731_095142_100.pth', False, True)
    mo = (r'Models/SS/NewModelSmallerMarkers.pth', True, False)
    mf = (r'Models/SweepTest/F1Synth_0804_065010.pth', False, False)
    m2 = (r'Models/SweepTest/SynB2_70EP.pth', False, False)
    m4 = (r'Models/SweepTest/SynB4_70EP.pth', False, False)

    models = [m4]
    pp = [True, False]
    pairs = []
    for x1 in models:
        for x2 in pp:
            pairs.append((x1, x2))
    for pair in tqdm(pairs[::-1], leave=True, position=0, total=len(pairs)):
        elem = pair[0]
        p = pair[1]
        BILINEAR = elem[2]
        pp_suffix = "_pp" if not p else ""
        outfld = os.path.join(r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\SS\Synth\Res\ModelCompB4', f"Run{num}_{os.path.basename(infld)}", f"{os.path.basename(elem[0]).split('.')[0]}_{os.path.basename(infld)}{pp_suffix}")
        OLD_MODEL = elem[1]
        predicitons = predict_ss(infld, outfld, elem[0], skip_pp=p)
        tar, pred = load_maps(os.path.join(outfld, "mask", "png"), predicitons)
        # visualize_pairs(os.path.join(outfld, 'pp', 'png'),
        #                 os.path.join(outfld, 'mask', 'png'),
        #                 predicitons,
        #                 os.path.join(outfld, "Visualize"))
        compute_metrics(tar, pred, os.path.join(outfld, "results.csv"))



