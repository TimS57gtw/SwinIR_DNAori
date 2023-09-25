import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from ReadWriteSXM import *

def get_modification(w, h, mat, show=False):
    existing = copy.deepcopy(mat)
    molecule = sorted(existing.flatten())[int(0.9 * existing.shape[0] * existing.shape[1])]
    arr = np.zeros((h, w))

    brd = list(mat[0, :]) + list(mat[-1, :]) + list(mat[:, 0]) + list(mat[:, -1])
    newbase = np.median(brd)


    if newbase < 0 or show:
        plt.imshow(mat)
        plt.title(f"New: {newbase}")
        plt.show()

    base = newbase
    back_std = min(3 * np.sqrt(newbase), 5) if newbase > 0 else 5

    modifications = ['noise', 'backgr', 'line', 'circle']
    mod = modifications[3] # random.choice(modifications)

    if mod == 'noise':
        scl = 0.5
        bgr = 0.25
        rd = lambda x, y: np.sqrt( np.square(x - w/2) /(scl*w)**2  + np.square(y - h/2 )/(scl*h)**2 )
        for i in range(w):
            for j in range(h):
                if random.random() < bgr:
                    arr[j, i] = np.random.normal(base, back_std)
                else:
                    arr[j, i] = existing[j, i] + (1-rd(i, j)) * np.random.uniform(-molecule, molecule)

        if show:
            plt.imshow(arr)
            plt.title("rd")
            plt.show()

        return arr, 'set'

    elif mod == 'backgr':
        arr = np.random.normal(base, 10, (h, w))
        return arr, 'set'

    elif mod == "line":

        lw = np.random.uniform(2, min(w, h)/3)

        v = np.array([0, 0])
        while np.linalg.norm(v) < 0.5 * np.sqrt(w**2 + h**2):
            if random.random() < 0.5:
                if random.random() < 0.5:
                    sp1 = np.array([0, np.random.random() * w])

                    r2 = random.random()
                    if r2 < 1/3:
                        sp2 = np.array([h, np.random.random() * w])
                    elif r2 < 2/3:
                        sp2 = np.array([np.random.random() * h, 0])
                    else:
                        sp2 = np.array([np.random.random() * h, w])

                else:
                    sp1 = np.array([h, np.random.random() * w])
                    r2 = random.random()
                    if r2 < 1 / 3:
                        sp2 = np.array([0, np.random.random() * w])
                    elif r2 < 2 / 3:
                        sp2 = np.array([np.random.random() * h, 0])
                    else:
                        sp2 = np.array([np.random.random() * h, w])
            else:
                if random.random() < 0.5:
                    sp1 = np.array([np.random.random() * h, 0])
                    r2 = random.random()
                    if r2 < 1 / 3:
                        sp2 = np.array([h, np.random.random() * w])
                    elif r2 < 2 / 3:
                        sp2 = np.array([0, random.random() * w])
                    else:
                        sp2 = np.array([np.random.random() * h, w])
                else:
                    sp1 = np.array([np.random.random() * h, w])
                    r2 = random.random()
                    if r2 < 1 / 3:
                        sp2 = np.array([h, np.random.random() * w])
                    elif r2 < 2 / 3:
                        sp2 = np.array([np.random.random() * h, 0])
                    else:
                        sp2 = np.array([0, random.random() * w])

            v = sp2 - sp1


        def dist(x, y):
            x = np.array([y, x])
            lbd = np.dot((x - sp1), v) / np.dot(v, v)
            d = np.linalg.norm(x - (sp1 + lbd * v))
            return d


        for i in range(w):
            for j in range(h):
                arr[j, i] = -molecule if dist(i, j) < lw else 0

        ret = existing + arr
        r2 = copy.deepcopy(ret)

        ns = np.random.normal(base, back_std, ret.shape)
        ret = np.maximum(ret, ns)

        if show:
            fig, axs = plt.subplots(1, 3    )
            axs[0].imshow(existing, cmap='gray', vmin=0, vmax=255)
            axs[1].imshow(r2, cmap='gray')
            axs[2].imshow(ret, cmap='gray', vmin=0, vmax=255)
            plt.show()

        return ret, 'set'

    else:
        scl = np.random.uniform(0.2, 0.4)
        cx = np.random.uniform(w/4, 3*w/4)
        cy = np.random.uniform(h/4, 3*h/4)

        rd = lambda x, y: np.sqrt(np.square(x - cx) / (scl * w) ** 2 + np.square(y - cy) / (scl * h) ** 2)
        for i in range(w):
            for j in range(h):
                arr[j, i] = -molecule if rd(i, j) < random.uniform(0.5, 1.5) else 0


        ret = arr + existing
        ns = np.random.normal(base, back_std, existing.shape)
        ret = np.maximum(ret, ns)

        if show:
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(arr)
            axs[1].imshow(existing)
            axs[2].imshow(ret)
            plt.show()

        return ret, 'set'




def modify_images(out_fld, resf, mod_pct=0, show=False):
    tempdict = os.path.join(out_fld, "Eval_Results", "tempdict_yolo.csv")
    pred_fld = os.path.join(out_fld, "yolo_prediction")
    idx = 0

    for arr, boxes, rng, _ in load_labels(pred_fld, tempdict):
        bx_arr = copy.deepcopy(arr)
        backgr = copy.deepcopy(arr)
        mod_arr = copy.deepcopy(arr).astype(float)
        meds = []
        for i in range(backgr.shape[0]):
            med = sorted(backgr[i, :])[int(0.1 * backgr.shape[1])]
            meds.append(med)
            mod_arr[i, :] -= med



        slope = np.median(mod_arr, axis=0)
        # plt.cla()
        # plt.plot(slope)
        # plt.show()
        if show:
            fg, axs = plt.subplots(2, 2)
            axs[0, 0].plot(meds)
            axs[0, 0].plot(slope)
            axs[0, 1].imshow(bx_arr, cmap='gray')
            axs[0, 1].set_title('before')
            axs[1, 0].imshow(mod_arr, cmap='gray')
            axs[1, 0].set_title('line')


        fit = scipy.stats.linregress([x for x in range(len(slope))], slope)

        for i in range(backgr.shape[1]):
            mod_arr[:, i] -= fit.slope * i

        baselift = fit.slope * backgr.shape[1]/2

        mod_arr += baselift


        if show:
            axs[1, 1].imshow(mod_arr, cmap='gray')
            axs[1, 1].set_title('slope')
            plt.show()





        for box in boxes:
            xmin, ymin, xmax, ymax = box
            for i in range(xmin, xmax+1):
                try:
                    bx_arr[ymin, i] = 255
                except IndexError:
                    pass
                try:
                    bx_arr[ymax, i] = 255
                except IndexError:
                    pass
            for i in range(ymin, ymax+1):
                try:
                    bx_arr[i, xmin] = 255
                except IndexError:
                    pass
                try:
                    bx_arr[i, xmax] = 255
                except IndexError:
                    pass
        if show:
            print(boxes)
            plt.imshow(bx_arr)
            plt.title(str(boxes[0]))
            plt.show()


        np.random.shuffle(boxes)
        for box in boxes:
            if random.random() < mod_pct:
                xmin, ymin, xmax, ymax = box
                w = min(xmax, arr.shape[1]) - max(xmin, 0)
                h = min(ymax, arr.shape[0]) - max(ymin, 0)
                modi, mode = get_modification(w, h, mod_arr[ max(ymin, 0):min(ymax, arr.shape[0]), max(xmin, 0):min(xmax, arr.shape[1])])
                # print(f"Modi {modi.shape}, Range: {mod_arr[ max(ymin, 0):min(ymax, arr.shape[0]), max(xmin, 0):min(xmax, arr.shape[1])].shape}, ymin:{ymin}, ymax:{ymax}, shp0:{arr.shape[0]}, xmin:{xmin}, xmax:{xmax}, shp1:{arr.shape[1]}")
                if mode == 'add':
                    mod_arr[ max(ymin, 0):min(ymax, arr.shape[0]), max(xmin, 0):min(xmax, arr.shape[1])] += modi
                else:
                    mod_arr[ max(ymin, 0):min(ymax, arr.shape[0]), max(xmin, 0):min(xmax, arr.shape[1])] = modi

                if show:
                    mart = copy.deepcopy(mod_arr)
                    for i in range(mart.shape[0]):
                        mart[i, :] += meds[i]
                    for i in range(mod_arr.shape[1]):
                        mart[:, i] += i * fit.slope

                    plt.imshow(mart)
                    plt.show()

        for i in range(mod_arr.shape[0]):
            mod_arr[i, :] += meds[i]
        for i in range(mod_arr.shape[1]):
            mod_arr[:, i] += i * fit.slope

        mod_arr -= baselift

        if show:
            plt.imshow(mod_arr)
            plt.show()
            mod_arr = mod_arr.clip(0, 255)

        os.makedirs(resf, exist_ok=True)
        name = f"Image{str(idx).zfill(6)}.sxm"
        name2 = f"Image{str(idx).zfill(6)}.png"
        idx += 1
        arr_2_sxm(mod_arr, range=rng, resfile=os.path.join(resf, name))
        plt.imsave(os.path.join(resf, name2), mod_arr, cmap='gray')



if __name__ == "__main__":

    # arr, _ = spm2arr(r"D:\seifert\PycharmProjects\DNAmeasurement\Output\Try167_NoBirka_FIT_UseU_True_LaterTime_Conf70_SynB4_70EP_XY_\spm\spm\CF\Image0043.spm")
#
    # for i in range(arr.shape[0]):
    #     med = sorted(arr[i, :])[int(0.1 * arr.shape[1])]
    #     arr[i, :] -= med
#
    # arr -= np.amin(arr)
    # arr *= 255 / np.amax(arr)
    # arr = arr[85:203, 85:255]
#
    # plt.imshow(arr)
    # plt.show()
#
#
    # vals = copy.deepcopy(arr).flatten()
    # mu = np.average(vals)
    # sig = np.std(vals)
    # plt.hist(vals, bins=100)
    # plt.title(f"{mu} {sig}")
    # plt.show(
    # assert 1 == 2


    for i in tqdm([0.05, 0.15, 0.25]):
        resf = os.path.join("ModifiedLiq", f"MP_{int(100*i)}")

        modify_images(r'D:\seifert\PycharmProjects\DNAmeasurement\Output\Try167_NoBirka_FIT_UseU_True_LaterTime_Conf70_SynB4_70EP_XY_',
                    resf, mod_pct=i)



