import os
import shutil

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import ReadWriteSXM
from Training.KAIR.main_test_swinir import main as apply_model
from Preprocessing import pretransform_image
from tqdm import tqdm

# def apply_model(infld, outfld, model):

def preprocess(infld, outfld, npyfld):
    infs = [os.path.join(infld, x) for x in os.listdir(infld)]
    npfs = [os.path.join(npyfld, x.split('.')[0] + ".npy") for x in os.listdir(infld)]
    oufs = [os.path.join(outfld, x) for x in os.listdir(infld)]

    for f in tqdm(range(len(infs)), desc='Preprocess'):
        if SAMPLE:
            pretransform_image(infs[f], npfs[f], img_size=None, show=False, is_mask=False, enhance_contrast=True, line_corr=False,
                           do_flatten=True, do_flatten_border=False, flip=False, mode='', flatten_line_90=True,
                           resize_method="bilinear", skip_all=False)
        else:
            pretransform_image(infs[f], npfs[f], img_size=None, show=False, is_mask=False, enhance_contrast=True,
                               line_corr=False,
                               do_flatten=False, do_flatten_border=True, flip=False, mode='', flatten_line_90=False,
                               resize_method="bilinear", skip_all=False)


        arr = np.load(npfs[f], allow_pickle=True)
        plt.imsave(oufs[f],arr, cmap='gray')





def denoise_set(infld, outfld, fraction=1.0, do_preprocess=True, model=r'D:\Dateien\KI_Speicher\SwinDNA\Models\DN_Sample_Long\models\172500_E.pth', save_sxm=True):
    pngf = os.path.join(outfld, "PNGs")
    png_ini = os.path.join(pngf, 'initial')
    png_pp = os.path.join(pngf, 'preproc')
    npy_pp = os.path.join(pngf, 'npy_pp')
    png_den = os.path.join(pngf, 'denoise')
    png_mix = os.path.join(pngf, 'mixed')
    npy_mix = os.path.join(pngf, 'npy_mix')
    png_pairs = os.path.join(pngf, 'pairs')
    scale=fraction

    os.makedirs(pngf, exist_ok=True)
    os.makedirs(png_ini, exist_ok=True)
    os.makedirs(png_pp, exist_ok=True)
    os.makedirs(npy_pp, exist_ok=True)
    os.makedirs(png_den, exist_ok=True)
    os.makedirs(png_mix, exist_ok=True)
    os.makedirs(png_pairs, exist_ok=True)
    os.makedirs(npy_mix, exist_ok=True)

    if list(os.listdir(infld))[0].split('.')[-1] == 'spm':
        ReadWriteSXM.spm_to_png(infld, png_pairs)
    elif list(os.listdir(infld))[0].split('.')[-1] == 'spm':
        ReadWriteSXM.sxm_to_png(infld, png_pairs)
    elif list(os.listdir(infld))[0].split('.')[-1] == 'png':
        shutil.copytree(infld, png_pairs, dirs_exist_ok=True)
    elif list(os.listdir(infld))[0].split('.')[-1] == 'npy':
        for elem in os.listdir(infld):
            plt.imsave(os.path.join(png_pairs, elem.split('.')[0] + ".png"), np.load(os.path.join(infld, elem), allow_pickle=True), cmap='gray')
    else:
        raise Exception("Unknown Filetype " + list(os.listdir(infld))[0].split('.')[-1])

    for elem in os.listdir(png_pairs):
        if elem.split('.')[-1] == 'png':
            shutil.copy(os.path.join(png_pairs, elem), os.path.join(png_ini, elem))

    if do_preprocess:
        preprocess(png_ini, png_pp, npy_pp)
    else:
        png_pp = png_ini

    apply_model(model, png_pp, png_den)

    names = list(os.listdir(png_ini))
    dnnames = list(os.listdir(png_den))

    print(names)
    print(dnnames)

    for i in range(len(names)):
        shutil.move(os.path.join(png_den, dnnames[i]), os.path.join(png_den, names[i]))

    for file in os.listdir(png_den):
        di = np.array(Image.open((os.path.join(png_den, file)))).astype(float)#[:, :, 0]
        ri = np.array(Image.open((os.path.join(png_ini, file)))).astype(float)[:, :, 0]

        mix = di * (scale) + ri * (1-scale)
        plt.imsave(os.path.join(png_mix, file), mix, vmin=0, vmax=255, cmap='gray')
        np.save(os.path.join(npy_mix, file.split('.')[0] + '.npy'), mix, allow_pickle=True)

        arr = np.load(os.path.join(npy_mix, file.split('.')[0] + '.npy'), allow_pickle=True)
        plt.imshow(arr)
        plt.show()

        if save_sxm:
            rng = []
            try:
                with open(os.path.join(png_pairs, file.split('.')[0] + '.txt'), 'r') as f:
                    for line in f:
                        rng.append(float(line))
            except FileNotFoundError:
                print("No Text File available: " + str(file.split('.')[0] + '.txt'))
                rng.append(1)
                rng.append(1)

            rang = (rng[0], rng[1])

            ReadWriteSXM.arr_2_sxm(mix, rang, os.path.join(outfld, file.split('.')[0] + '.sxm'))



SAMPLE = False
PREPROC=False
if __name__ == "__main__":
    model = r'D:\seifert\PycharmProjects\SwinDNA\swinir_sr_classical_patch64_x1_MoleculeLONG\models\200000_G.pth'
    outfld = os.path.join(r'D:\seifert\PycharmProjects\SwinDNA\Results\ApplyDenoise',"Test")
    os.makedirs(outfld, exist_ok=True)
    denoise_set(infld=r'D:\seifert\PycharmProjects\DNAmeasurement\SR\SwinIR_Include\5_NoBirka_100_70000_G_200000_G\pp_crop_origami_npy',
                outfld=outfld, fraction=1, do_preprocess=PREPROC, model=model)

    if SAMPLE:
        scale=1.0
        model = r'D:\seifert\PycharmProjects\SwinDNA\swinir_sr_classical_patch64_x1_SampleLONG\models\70000_G.pth'
        infld = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\NoBirka\spm\CF'
        outfld = os.path.join(r'D:\seifert\PycharmProjects\SwinDNA\Results\ApplyDenoise', os.path.basename(infld) + f"_{int(100*scale)}_{os.path.basename(model).split('.')[0]}")
        os.makedirs(outfld, exist_ok=True)

        denoise_set(infld=infld, outfld=outfld, fraction=scale, do_preprocess=PREPROC, model=model)

    else:
        scale = 0.5
        model = r'D:\seifert\PycharmProjects\SwinDNA\swinir_sr_classical_patch64_x1_MoleculeLONG\models\200000_G.pth'
        idx = 0
        # print(os.path.isfile(model))
        # with open(model, 'rb') as f:
        #     for line in f:
        #      idx +=1
        #      print(f"{idx} -> {line}")

        infld = r'D:\seifert\PycharmProjects\SwinDNA\Results\MoleculeCrops'
        outfld = os.path.join(r'D:\seifert\PycharmProjects\SwinDNA\Results\ApplyDenoise',
                              os.path.basename(infld) + f"_{int(100 * scale)}_{os.path.basename(model).split('.')[0]}" + ("PP" if PREPROC else ""))
        os.makedirs(outfld, exist_ok=True)

        denoise_set(infld=infld, outfld=outfld, fraction=scale, do_preprocess=PREPROC, model=model)
