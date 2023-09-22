import os
import shutil

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import ReadWriteSXM
from Training.KAIR.main_test_swinir import main as apply_model

# def apply_model(infld, outfld, model):


def denoise_set(infld, outfld, fraction=1.0, model=r'D:\Dateien\KI_Speicher\SwinDNA\Models\DN_Sample_Long\models\172500_E.pth'):
    pngf = os.path.join(outfld, "PNGs")
    png_ini = os.path.join(pngf, 'initial')
    png_den = os.path.join(pngf, 'denoise')
    png_mix = os.path.join(pngf, 'mixed')
    png_pairs = os.path.join(pngf, 'pairs')

    os.makedirs(pngf, exist_ok=True)
    os.makedirs(png_ini, exist_ok=True)
    os.makedirs(png_den, exist_ok=True)
    os.makedirs(png_mix, exist_ok=True)
    os.makedirs(png_pairs, exist_ok=True)

    if list(os.listdir(infld))[0].split('.')[-1] == 'spm':
        ReadWriteSXM.spm_to_png(infld, png_pairs)
    elif list(os.listdir(infld))[0].split('.')[-1] == 'spm':
        ReadWriteSXM.sxm_to_png(infld, png_pairs)
    else:
        raise Exception("Unknown Filetype " + list(os.listdir(infld))[0].split('.')[-1])

    for elem in os.listdir(png_pairs):
        if elem.split('.')[-1] == 'png':
            shutil.copy(os.path.join(png_pairs, elem), os.path.join(png_ini, elem))

    apply_model(model, png_ini, png_den)

    names = list(os.listdir(png_ini))
    dnnames = list(os.listdir(png_den))

    print(names)
    print(dnnames)

    for i in range(len(names)):
        shutil.move(os.path.join(png_den, dnnames[i]), os.path.join(png_den, names[i]))

    for file in os.listdir(png_den):
        di = np.array(Image.open((os.path.join(png_den, file))))#[:, :, 0]
        ri = np.array(Image.open((os.path.join(png_ini, file))))#[:, :, 0]

        mix = di * (scale) + ri * (1-scale)
        plt.imsave(os.path.join(png_mix, file), mix, vmin=0, vmax=255, cmap='gray')
        rng = []
        with open(os.path.join(png_pairs, file.split('.')[0] + '.txt'), 'r') as f:
            for line in f:
                rng.append(float(line))

        rang = (rng[0], rng[1])

        ReadWriteSXM.arr_2_sxm(mix, rang, os.path.join(outfld, file.split('.')[0] + '.sxm'))




if __name__ == "__main__":
    scale=1.0
    model = r'D:\Dateien\KI_Speicher\SwinDNA\Models\DN_Sample_Long\models\172500_E.pth'
    # infld = r'D:\Dateien\KI_Speicher\SwinDNA\datasets\RectsHisto'
    infld = r'D:\Dateien\KI_Speicher\EvalChainDS\TotalDatasets\INDIV\CF'
    outfld = os.path.join(r'D:\Dateien\KI_Speicher\SwinDNA\Results', os.path.basename(infld) + f"_{int(100*scale)}_{os.path.basename(model).split('.')[0]}")
    os.makedirs(outfld, exist_ok=True)

    denoise_set(infld, outfld, scale, model)


