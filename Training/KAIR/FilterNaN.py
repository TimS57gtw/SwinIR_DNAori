import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
t_fld_gt = r"D:/Dateien/KI_Speicher/SwinSTM_Denoise/SynthData/ImagesV4/GT_npy"
t_fld_ns = r"D:/Dateien/KI_Speicher/SwinSTM_Denoise/SynthData/ImagesV4/NS_npy"

gtf = [os.path.join(t_fld_gt, x) for x in os.listdir(t_fld_gt)]
nsf = [os.path.join(t_fld_ns, x) for x in os.listdir(t_fld_ns)]

for gt, ns in tqdm(zip(gtf, nsf), total=len(gtf)):
    g = np.load(gt, allow_pickle=True)
    n = np.load(ns, allow_pickle=True)

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(g)
    # axs[1].imshow(n)
    # plt.show()

    isNanG = np.any(np.isnan(g))
    isNanN = np.any(np.isnan(g))
    dn = os.path.dirname(gt)
    dn2 = os.path.dirname(dn)
    spec_path = os.path.join(dn2, "Spectra", os.path.basename(gt).split('.')[0] + '.csv')

    if isNanG or isNanN:
        print("Deleting ", os.path.basename(gt))
        os.remove(gt)
        os.remove(ns)
        os.remove(spec_path)




