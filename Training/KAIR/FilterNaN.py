import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# t_fld_gt = r"D:/Dateien/KI_Speicher/SwinSTM_Denoise/SynthData/ImagesV4/GT_npy"
# t_fld_ns = r"D:/Dateien/KI_Speicher/SwinSTM_Denoise/SynthData/ImagesV4/NS_npy"
#
# gtf = [os.path.join(t_fld_gt, x) for x in tqdm(os.listdir(t_fld_gt))]
# nsf = [os.path.join(t_fld_ns, x) for x in tqdm(os.listdir(t_fld_ns))]
#
# for gt, ns in tqdm(zip(gtf, nsf), total=len(gtf)):
#     # g = np.load(gt, allow_pickle=True)
#     # n = np.load(ns, allow_pickle=True)
#
#     # fig, axs = plt.subplots(1, 2)
#     # axs[0].imshow(g)
#     # axs[1].imshow(n)
#     # plt.show()
#
#     # isNanG = np.any(np.isnan(g))
#     # isNanN = np.any(np.isnan(g))
#     dn = os.path.dirname(gt)
#     dn2 = os.path.dirname(dn)
#     spec_path = os.path.join(dn2, "Spectra", os.path.basename(gt).split('.')[0] + '.csv')
#     df = pd.read_csv(spec_path)
#     fs = list(df['f'])
#     if len(fs) != 64:
#         print(f"{os.path.basename(spec_path)}, {len(fs)}")
#
#
#     # if isNanG or isNanN:
#         # print("Detected ", os.path.basename(gt))
#
#         # os.remove(gt)
#         # os.remove(ns)
#         # os.remove(spec_path)
#
#
#
#

def show_sizes(inf):
    sizes = {}
    bildf = os.path.join(inf, "NS_npy")
    gtf = os.path.join(inf, "GT_npy")
    spcf = os.path.join(inf, "Spectra")
    spcfF = os.path.join(inf, "Spectra_Full")

    bfs = []
    gfs = []
    sfs = []
    ffs = []
    for elem in os.listdir(os.path.join(inf, bildf)):
        bfs.append(os.path.join(inf, bildf, elem))
    for elem in os.listdir(os.path.join(inf, gtf)):
        gfs.append(os.path.join(inf, gtf, elem))
    for elem in os.listdir(os.path.join(inf, spcf)):
        sfs.append(os.path.join(inf, spcf, elem))
    for elem in os.listdir(os.path.join(inf, spcfF)):
        ffs.append(os.path.join(inf, spcfF, elem))

    for i in tqdm(range(len(bfs))):
        na = np.load(bfs[i], allow_pickle=True)
        ga = np.load(gfs[i], allow_pickle=True)

        valid_size = [256]

        if not np.array_equal(np.array(na.shape), np.array(ga.shape)):
            print(bfs[i], na.shape, ga.shape)

        if str(na.shape[0]) not in sizes.keys():
            sizes[str(na.shape[0])] = 0

        sizes[str(na.shape[0])] += 1

        if np.any(np.isnan(ga)) or np.any(np.isnan(na)):
            print("NaN in ", bfs[i])
            continue

        df = pd.read_csv(sfs[i])
        if len(df['f']) != 64:
            print(f"Spectrum Error in {sfs[i]}: {len(df['f'])}")
            continue

    print(sizes)

def join_NotNaNs(infs, outfs):


    # create_structures
    os.makedirs(outfs, exist_ok=True)
    bildf = os.path.join('bild', 'npy')
    gtf = os.path.join('bild_truth', 'npy')
    spcfF = os.path.join('data', 'spectrum_full')
    spcf = os.path.join('data', 'spectrum')

    obildf = os.path.join(outfs, "NS_npy")
    ogtf = os.path.join(outfs, "GT_npy")
    ospcf = os.path.join(outfs, "Spectra")
    ospcfF = os.path.join(outfs, "Spectra_Full")


    os.makedirs(obildf, exist_ok=True)
    os.makedirs(ogtf, exist_ok=True)
    os.makedirs(ospcf, exist_ok=True)
    os.makedirs(ospcfF, exist_ok=True)

    totals = [len([x for x in tqdm(os.listdir(os.path.join(inf, bildf)), desc='Calc Total')]) for inf in infs]
    total = sum(totals)
    print("Total: ", total)
    idx = 0

    bfs = []
    gfs = []
    sfs = []
    ffs = []
    with tqdm(desc='Gather Files', total=4*total) as pbar:
        for inf in infs:
            for elem in os.listdir(os.path.join(inf, bildf)):
                bfs.append(os.path.join(inf, bildf, elem))
                pbar.update(1)
            for elem in os.listdir(os.path.join(inf, gtf)):
                gfs.append(os.path.join(inf, gtf, elem))
                pbar.update(1)
            for elem in os.listdir(os.path.join(inf, spcf)):
                sfs.append(os.path.join(inf, spcf, elem))
                pbar.update(1)
            for elem in os.listdir(os.path.join(inf, spcfF)):
                ffs.append(os.path.join(inf, spcfF, elem))
                pbar.update(1)


    assert len(bfs) == len(gfs)
    assert len(gfs) == len(sfs)
    assert len(sfs) == len(ffs)

    for i in tqdm(range(len(bfs))):
        na = np.load(bfs[i], allow_pickle=True)
        ga = np.load(gfs[i], allow_pickle=True)

        if np.any(np.isnan(ga)) or np.any(np.isnan(na)):
            print("NaN in ", bfs[i])
            continue



        df = pd.read_csv(sfs[i])
        if len(df['f']) != 64:
            print(f"Spectrum Error in {sfs[i]}: {len(df['f'])}")
            continue

        # df2 = pd.read_csv(ffs[i])
        # if len(df2['f']) != 256:
        #     print(f"Spectrum Error in {ffs[i]}: {len(df2['f'])}")
        #     continue

        shutil.copy(bfs[i], os.path.join(obildf, f"Image{str(idx).zfill(6)}.npy"))
        shutil.copy(gfs[i], os.path.join(ogtf, f"Image{str(idx).zfill(6)}.npy"))
        shutil.copy(sfs[i], os.path.join(ospcf, f"Image{str(idx).zfill(6)}.csv"))
        shutil.copy(ffs[i], os.path.join(ospcfF, f"Image{str(idx).zfill(6)}.csv"))

        idx += 1







if __name__ == "__main__":
    show_sizes(r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\ImagesV5_TF')

    assert 2 == 3

    inflds = [r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\ImagesV5',
              r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\ImagesV5_P2']
    # inflds = [r'D:\Dateien\KI_Speicher\SwinSTM_Denoise\SynthData\ImagesV5']
    inflds = [r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\ImagesV5_test']

    resfld = r'D:\Dateien\KI_Speicher\SwinSTM_Denoise\SynthData\ImagesV5_test_TF'


    join_NotNaNs(inflds, resfld)
