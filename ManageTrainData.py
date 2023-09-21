import os
import shutil

import PIL.Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
def combine_SR_set(inflds, outfld, factor=4):
    idx = 0
    resHR = os.path.join(outfld, "HR")
    resLR = os.path.join(outfld, "LR")
    resLRN = os.path.join(outfld, "LR_N")
    resHRN = os.path.join(outfld, "HR_N")

    os.makedirs(resHR, exist_ok=True)
    os.makedirs(resLR, exist_ok=True)
    os.makedirs(resLRN, exist_ok=True)
    os.makedirs(resHRN, exist_ok=True)

    for infld in inflds:
        HR = os.path.join(infld, 'bild')
        HR_N = os.path.join(infld, 'bild_truth')

        for file in tqdm(os.listdir(HR)):
            name = f"Image{str(idx).zfill(6)}.png"
            idx += 1
            locnum = int(file.split(".")[0][5:])
            lbn = f"SR_Label{str(locnum).zfill(6)}.png"

            shutil.copy(os.path.join(HR, file), os.path.join(resHR, name))
            shutil.copy(os.path.join(HR_N, lbn), os.path.join(resHRN, name))

            hri = Image.open(os.path.join(HR, file))
            hape = hri.size[0]
            newsize = int(round(hape / factor))
            lri = hri.resize((newsize, newsize), resample=Image.Resampling.NEAREST)
            lri.save(os.path.join(resLR, name))

            hrn = Image.open(os.path.join(HR_N, lbn))
            lrn = hrn.resize((newsize, newsize), resample=Image.Resampling.NEAREST)
            lrn.save(os.path.join(resLRN, name))



def generate_same_base(noisy_fld, perf_fld, tar_fld):
    files = os.listdir(noisy_fld)
    os.makedirs(tar_fld, exist_ok=True)
    for file in tqdm(files):
        ni = np.array(Image.open(os.path.join(noisy_fld, file)))[:, :, 0].astype(int)
        pi = np.array(Image.open(os.path.join(perf_fld, file)))[:, :, 0].astype(int)
        diff = ni - pi


        vals = diff.flatten()
        # plt.hist(vals, bins = int(np.sqrt(len(vals))))
        # plt.title(np.mean(vals))
        # plt.show()
        # plt.switch_backend('TkAgg')
        # plt.imshow(pi)
        # plt.show()
        # fig, axs = plt.subplots(1, 3)
        pi += int(round(np.mean(vals)))
        diff = ni - pi
        # axs[0].imshow(ni)
        # axs[1].imshow(pi)
        # axs[2].imshow(diff)
        # plt.show()
        pi = np.clip(pi, 0, 255)
        plt.imsave(os.path.join(tar_fld, file), pi, cmap='gray', vmax=255, vmin=0)
        plt.cla()







if __name__ == "__main__":
    generate_same_base(
        r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Sample_HRN\LR',
        r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Sample_HRN\LR_N',
        r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Sample_HRN\LR_NB'
    )

    generate_same_base(
        r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Sample_HRN\LR_Test',
        r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Sample_HRN\LR_N_Test',
        r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Sample_HRN\LR_NB_Test'
    )
    assert 5 == 6

    inflds = [
        r'C:\Users\seifert\PycharmProjects\STM_Simulation\bildordner\DNA_SR2\Set2_HRNoise_HRNoise',
    ]
    outfld = r'C:\Users\seifert\PycharmProjects\SwinIR\TrainData\Sample_HRN'


    combine_SR_set(inflds, outfld, factor=4)
