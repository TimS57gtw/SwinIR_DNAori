import copy
import multiprocessing
import os
import time

import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import Evaluate_SS, scipy
import random
import matplotlib
import struct
import seaborn as sns
from multiprocessing import Process
from matplotlib import cm
from scipy.stats import gaussian_kde


def hoshen_koppelmann(arr):
    """
    Implemenierung des Hoshen Koppelmann-Algorithmus
    """

    # Neues Array mit Padding links und oben
    padded_array = np.pad(arr, (1, 0))
    length = arr.shape[0]
    size = length ** 2

    # Erstellung der Matrix fuer labels
    label = np.zeros(np.shape(padded_array), dtype=np.int32)

    # Zaehl-Array. None um Array-Indizes bei 1 zu beginnen
    n = [None]

    # Cluster-Index
    c = 1

    # Setze Label fuer linken und oberen Rand
    for i in range(length + 1):
        label[0, i] = size
        label[i, 0] = size

    # Fuktion zum Finden des "guten" labels
    def good_label(i):

        m = n[i]
        if m < 0:
            r = -m
            assert r >= 0
            m = n[r]

            while m < 0:
                r = -m
                assert r >= 0
                m = n[r]

            n[i] = -r

        else:
            r = i

        return r

    # Iteration über das Array und Fallunterscheidung je nach Besetzung der Plätze
    for i in range(1, length + 1):
        for j in range(1, length + 1):

            # Pixel ist leer
            if padded_array[i, j] == 0:
                label[i, j] = size
                continue

            # Oben und Unten sind nicht besetzt -> Neues Cluster
            if padded_array[i - 1, j] == 0 and padded_array[i, j - 1] == 0:
                label[i, j] = c
                n.append(1)
                c += 1
                continue

            # Nur Links ist besetzt
            if padded_array[i, j - 1] == 1 and padded_array[i - 1, j] == 0:
                l = good_label(label[i, j - 1])
                label[i, j] = l
                n[l] += 1


            # Nur oben ist besetzt
            elif padded_array[i, j - 1] == 0 and padded_array[i - 1, j] == 1:
                l = good_label(label[i - 1, j])
                n[l] += 1
                label[i, j] = l

            # Beide sind besetzt -> Kombiniere Cluster
            else:
                l = good_label(label[i, j - 1])
                u = good_label(label[i - 1, j])

                if u == l:
                    n[l] += 1
                    label[i, j] = l

                else:
                    if l < u:
                        n[l] = n[l] + n[u] + 1
                        n[u] = -l
                        label[i, j] = l
                    else:
                        assert l != u
                        n[u] = n[u] + n[l] + 1
                        n[l] = -u
                        label[i, j] = u

    # Umnummerierung
    cluster_sizes = [None]
    known_good_labels = [None]

    # Echte Label fuer jeden Pixel
    for i in range(np.shape(label)[0]):
        for j in range(np.shape(label)[1]):
            if padded_array[i, j] == 1:
                gl = good_label(label[i, j])
                if gl in known_good_labels:
                    ci = known_good_labels.index(gl)
                    label[i, j] = ci
                else:
                    known_good_labels.append(gl)
                    cluster_sizes.append(n[gl])
                    label[i, j] = len(known_good_labels) - 1
            else:
                label[i, j] = -1

    # Karte Mit Cluster-Groessen
    cs_map = np.zeros(np.shape(label))
    for i in range(np.shape(label)[0]):
        for j in range(np.shape(label)[1]):
            if padded_array[i, j] == 0:
                cs_map[i, j] = 0
            else:
                gl = label[i, j]
                cs_map[i, j] = cluster_sizes[gl]

    # Abschneiden der Array zum Eliminieren des Paddings
    labels = label[1:, 1:]

    return labels


def find_cog(index, arr):
    i_s = []
    j_s = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == index:
                i_s.append(i)
                j_s.append(j)
    if len(i_s) == 0:
        return False
    i_s = sorted(i_s)
    j_s = sorted(j_s)
    return np.average(i_s), np.average(j_s), i_s[0], i_s[-1], j_s[0], j_s[-1]


def filter2markers(folder, resfolder, dist_th=30):
    labelfolder_bin = os.path.join(folder, "data_bin", "numpy")
    labelfolder = os.path.join(folder, "data", "numpy")
    imagefolder = os.path.join(folder, "sxm", "numpy")

    res_labelfolder_bin = os.path.join(resfolder, "data_bin", "numpy")
    res_labelfolder = os.path.join(resfolder, "data", "numpy")
    res_imagefolder = os.path.join(resfolder, "sxm", "numpy")
    os.makedirs(res_labelfolder_bin, exist_ok=True)
    os.makedirs(res_labelfolder, exist_ok=True)
    os.makedirs(res_imagefolder, exist_ok=True)

    indx = 1

    for file in tqdm(os.listdir(imagefolder)):
        bin_label = os.path.join(labelfolder_bin, file.split(".")[0] + "_mask.npy")
        label = os.path.join(labelfolder, file.split(".")[0] + "_mask.npy")
        image = os.path.join(imagefolder, file)

        arr = np.load(bin_label, allow_pickle=True)
        clusters = hoshen_koppelmann(arr)
        # plt.imshow(clusters)
        # plt.show()
        maxi = int(np.amax(clusters))
        if maxi < 2:
            continue
        found_apart = False
        cogslst = []
        for i in range(1, maxi + 1):
            x, y, _, _, _, _ = find_cog(i, clusters)
            cntr = np.array(x, y)
            cogslst.append(cntr)
            lc = len(cogslst)
            for j in range(lc):
                cog = cogslst[j]
                if np.linalg.norm(cog - cntr) > dist_th:
                    found_apart = True
                    break
            if found_apart:
                break

        if not found_apart:
            continue
            # print("ERR")
            # cogslst = []
            # for i in range(1, maxi+1):
            #     x, y, _, _, _, _ = find_cog(i, clusters)
            #     cntr = np.array(x, y)
            #     cogslst.append(cntr)
            #     lc = len(cogslst)
            #     for j in range(lc):
            #         cog = cogslst[j]
            #         print("Dist: ", np.linalg.norm(cog - cntr))
            #         cogslst.append(cntr)

            # plt.imshow(clusters)
            # plt.title("Not Apart")
            # plt.show()

        res_lb = os.path.join(res_labelfolder_bin, f"Image{str(indx).zfill(6)}_mask.npy")
        res_l = os.path.join(res_labelfolder, f"Image{str(indx).zfill(6)}_mask.npy")
        res_i = os.path.join(res_imagefolder, f"Image{str(indx).zfill(6)}.npy")
        shutil.copy(bin_label, res_lb)
        shutil.copy(label, res_l)
        shutil.copy(image, res_i)
        indx += 1


def combine_angles(fld, resf, angle_thrsh=np.pi / 6):
    def in_y(theta):
        for i in range(-10, 10):
            if i * np.pi - angle_thrsh <= theta < i * np.pi + angle_thrsh:
                return True
        return False
        # return -angle_thrsh <= theta < angle_thrsh or np.pi - angle_thrsh <= theta < np.pi + angle_thrsh or \
        #        2 * np.pi - angle_thrsh <= theta < 2 * np.pi + angle_thrsh

    def in_x(theta):
        for j in range(-10, 10):
            i = 2 * j + 1
            if (i * np.pi / 2) - angle_thrsh <= theta < (i * np.pi / 2) + angle_thrsh:
                return True
        return False

    files = [os.path.join(fld, x) for x in os.listdir(fld) if x.endswith("txt")]
    lines = []
    for file in tqdm(files):
        with open(file, "r") as f:
            for line in f:
                if line.startswith("Angle"):
                    continue
                lines.append(line.strip())

    with open(resf, "w") as f:
        f.write("Angle,X,Y,inX,inY\n")
        for line in lines:
            angle = float(line.split(",")[0]) * np.pi / 180
            inx = 1 if in_x(angle) else 0
            iny = 1 if in_y(angle) else 0
            line += f",{inx},{iny}\n"
            f.write(line)


def recalc_effangle(file, resf):
    def trafo_th(theta):
        if theta < 0:
            return theta  # NOQA:E701
        else:
            theta = theta % np.pi
            if theta > np.pi / 2:
                return np.pi - theta
            else:
                return theta  # NOQA:E701

    deg = lambda x: 180 * x / np.pi

    with open(resf, "w") as rf:
        with open(file, "r") as f:
            for line in f:
                if line.startswith("Image,pred"):
                    rf.write(line)
                    continue
                prts = line.split(",")
                th = float(prts[2])
                th = trafo_th(th)
                th = deg(th)
                prts[2] = str(th)

                rf.write(",".join(prts))


def make_angle_list2(folder, resf):
    rf = open(resf, "w")  # file,No,X,Y,Angle
    rf.write(f"File,crpidx,midx,midy,Theta\n")

    binarize = lambda x: 1 if x > 0 else 0
    binarize = np.vectorize(binarize)
    for file in tqdm(os.listdir(folder)):

        labelcrops = []

        img = Image.open(os.path.join(folder, file))
        arr = np.array(img)[:, :]
        if type(arr[0, 0]) is np.ndarray:
            arr = arr[:, :, 0]

        arr_o = binarize(arr)

        hk_o = hoshen_koppelmann(arr_o)

        vals = np.unique(hk_o)
        ori_idcs = []
        for i in vals:
            if i > 0:
                ori_idcs.append(i)

        for oidx in ori_idcs:
            positions = np.argwhere(hk_o == oidx)
            xs = []
            ys = []
            for i in range(len(positions)):
                xs.append(positions[i][0])
                ys.append(positions[i][1])

            midx = np.average(xs)
            midy = np.average(ys)

            left = max(0, min(xs) - 2)
            right = min(arr.shape[0] - 1, max(xs) + 2)
            bot = max(0, min(ys) - 2)
            top = min(arr.shape[1] - 1, max(ys) + 2)

            crp = copy.deepcopy(arr[left:right, bot:top])
            # matplotlib.use('TkAgg')
            # plt.imshow(crp)
            # plt.title("Crop")
            # plt.show()

            labelcrops.append((crp, midx, midy))

        for crpidx, params in enumerate(labelcrops):
            lbl, midx, midy = params
            # plt.imshow(lbl)
            # plt.title(f"Crp No {crpidx}")
            # plt.show()
            lblmax = np.amax(lbl)
            only_ss = lambda x: 1 if x == lblmax else 0
            only_ss = np.vectorize(only_ss)
            lbl = only_ss(lbl)
            fc_ret = Evaluate_SS.find_crops(lbl, show_all=False)
            if fc_ret is None:
                rf.write(f"{file},{crpidx},{midx},{midy},-1\n")
                continue
            crops, cogs, label_mod = fc_ret
            if len(cogs) <= 1:
                rf.write(f"{file},{crpidx},{midx},{midy},-1\n")
                continue
            cog1 = np.array([cogs[0][0], cogs[0][1]])
            cog2 = np.array([cogs[1][0], cogs[1][1]])
            cogdif = cog1 - cog2
            theta = np.arctan2(cogdif[1], cogdif[0])
            if theta < 0:
                cogdif *= -1
                theta = np.arctan2(cogdif[1], cogdif[0])

            thetadeg = 180 * theta / np.pi
            if thetadeg > 90:
                thetadeg = 180 - thetadeg
            # matplotlib.use('TkAgg')
            # print("Theta: ", thetadeg)
            # plt.imshow(lbl)
            # plt.title(thetadeg)
            # plt.show()
            rf.write(f"{file},{crpidx},{midx},{midy},{thetadeg}\n")


def make_angle_list(folder, resf):
    binarize = lambda x: 1 if x > 0 else 0
    binarize = np.vectorize(binarize)
    markering = lambda x: 1 if x == 255 else 0
    markering = np.vectorize(markering)
    rf = open(resf, "w")

    for file in tqdm(os.listdir(folder)):
        img = Image.open(os.path.join(folder, file))
        arr = np.array(img)[:, :]
        arr_o = binarize(arr)

        hk_o = hoshen_koppelmann(arr_o)

        vals = np.unique(hk_o)
        ori_idcs = []
        for i in vals:
            if i > 0:
                ori_idcs.append(i)

        ori2marker = {}
        for elem in ori_idcs:
            ori2marker[elem] = []

        arr_m = markering(arr)
        hk_m = hoshen_koppelmann(arr_m)

        vals = np.unique(hk_m)
        mrk_idcs = []
        for i in vals:
            if i > 0:
                mrk_idcs.append(i)

        for idx in mrk_idcs:
            midx = []
            midy = []
            positions = np.argwhere(hk_m == idx)
            for i in range(len(positions)):
                midx.append(positions[i][0])
                midy.append(positions[i][1])
            cogx = int(round(np.average(midx)))
            cogy = int(round(np.average(midy)))

            if hk_o[cogx, cogy] == -1:
                print("Cog off: ", cogx, cogy)
                plt.imshow(hk_o)
                plt.show()

            ori2marker[hk_o[cogx, cogy]].append((idx, np.average(midx), np.average(midy)))

        for key in ori2marker.keys():
            midx = []
            midy = []
            positions = np.argwhere(hk_o == key)
            for i in range(len(positions)):
                midx.append(positions[i][0])
                midy.append(positions[i][1])
            cogx = np.average(midx)
            cogy = np.average(midy)

            complete = 1 if len(ori2marker[key]) == 2 else 0

            if complete:
                mx1 = ori2marker[key][0][1]
                my1 = ori2marker[key][0][2]
                mx2 = ori2marker[key][1][1]
                my2 = ori2marker[key][1][2]

                dy = my2 - my1
                dx = mx2 - mx1

                theta = np.arctan2(dy, dx)
                thetadeg = 180 * theta / np.pi
                print(thetadeg)
                img.show()
            else:
                thetadeg = -1

            rf.write(f"{file},{complete},{cogx},{cogy},{thetadeg}")

    rf.close()


def smooth_thetas(file, resf, resolution=200, fwhm=3):
    angles = []
    with open(file, "r") as f:
        for line in tqdm(f, desc="Reading..."):
            parts = line.split(",")
            try:
                if len(parts) > 10:
                    ang = float(parts[2])
                else:
                    ang = float(parts[4])
            except ValueError:  # Header
                continue
            angles.append(ang)

    filteredang = [a for a in angles if a >= 0]
    angles = filteredang

    sig = fwhm / (2 * np.sqrt(2 * np.log(2)))

    def gaussian(x, x0):
        return (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(- (x - x0) ** 2 / (2 * sig ** 2))

    def smooth(baseang):
        weights = 0
        for angle in angles:
            w = gaussian(angle, baseang)
            weights += w
        return weights

    testangs = np.linspace(0, 90, resolution)
    ws = []
    for elem in tqdm(testangs, desc='Smoothing...'):
        ws.append(smooth(elem))
    # matplotlib.use("TkAgg")
    #
    # fig, axes = plt.subplots(1, 1, sharey=True)
    #
    # axes.plot(testangs, 10*np.array(ws), c="r", zorder=1)
    # axes.hist(angles, zorder=-1)
    # plt.title("Hist vs smooth")
    # plt.show()

    with open(resf, "w") as f:
        for i in range(len(testangs)):
            f.write(f"{testangs[i]},{ws[i]}\n")


def resize_folder(inpyt, outpt):
    for file in tqdm(os.listdir(inpyt)):
        arr = np.load(os.path.join(inpyt, file))
        zoomfak = 64 / 50
        # plt.imshow(arr)
        # plt.show()
        newmat = np.array(scipy.ndimage.zoom(arr, zoomfak, order=1))
        assert newmat.shape[0] == 64
        # plt.imshow(newmat)
        # plt.show()
        # return
        np.save(os.path.join(outpt, file), newmat, allow_pickle=True)


def resize_mask(inpyt, outpt):
    for file in tqdm(os.listdir(inpyt)):
        arr = np.load(os.path.join(inpyt, file))
        # plt.imshow(arr)
        # plt.show()
        zoomfak = 64 / 50
        newmat = np.array(scipy.ndimage.zoom(arr, zoomfak, order=0))
        assert newmat.shape[0] == 64
        # plt.imshow(newmat)
        # plt.show()
        # return
        np.save(os.path.join(outpt, file), newmat, allow_pickle=True)


def resize_trainset(inpt, outpt, frm=64, to=50):
    os.makedirs(outpt, exist_ok=True)
    train = os.path.join(inpt, "Train")
    test = os.path.join(inpt, "Test")
    trainO = os.path.join(outpt, "Train")
    testO = os.path.join(outpt, "Test")
    os.makedirs(testO, exist_ok=True)
    os.makedirs(testO, exist_ok=True)

    a = [train, test]
    b = [trainO, testO]

    for i in range(len(a)):

        ind = a[i]
        oud = b[i]

        inb = os.path.join(ind, 'bild')
        oub = os.path.join(oud, 'bild')
        os.makedirs(oub, exist_ok=True)

        inl = os.path.join(ind, 'data', 'PNG')
        oul = os.path.join(oud, 'data', 'PNG')
        os.makedirs(oul, exist_ok=True)

        for elem in tqdm(list(os.listdir(inl))):
            img = os.path.join(inl, elem)
            outfile = os.path.join(oul, elem)

            pl = np.array(Image.open(img))[:, :, 0]
            # print(pl)
            # plt.switch_backend('TkAgg')
            # plt.imshow(pl)
            # plt.show()

            zoomfak = to / frm
            newmat = np.array(scipy.ndimage.zoom(pl, zoomfak, order=0))
            # plt.imshow(newmat)
            # plt.show()

            plt.imsave(outfile, newmat, cmap='gray')

        for elem in tqdm(list(os.listdir(inb))):
            img = os.path.join(inb, elem)
            outfile = os.path.join(oub, elem)

            pl = np.array(Image.open(img))[:, :, 0]
            # print(pl)
            # plt.switch_backend('TkAgg')
            # plt.imshow(pl)
            # plt.show()

            zoomfak = to / frm
            newmat = np.array(scipy.ndimage.zoom(pl, zoomfak, order=0))
            # plt.imshow(newmat)
            # plt.show()

            plt.imsave(outfile, newmat, cmap='gray')


def analyze_density(resfim, resfcsv, xs, ys, res):
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xl = xmax - xmin
    yl = ymax - ymin
    xstp = np.linspace(xmin, xmax, res)
    ystp = np.linspace(ymin, ymax, res)

    def density(x, y):
        ret = 0
        for i in range(len(xs)):
            xd = np.abs(x - xs[i]) / (xl / res)
            if xd > 2.5:
                continue
            yd = np.abs(y - ys[i]) / (yl / res)
            if xd < 0.5:
                xf = 4
            elif xd < 1.5:
                xf = 2
            else:
                xf = 1
            if yd < 0.5:
                yf = 4
            elif yd < 1.5:
                yf = 2
            else:
                yf = 1
            ret += xf * yf / 100

        return ret

    # density = np.vectorize(density)
    X, Y = np.meshgrid(xstp, ystp)

    Z = np.zeros_like(X)
    # Z = density(X, Y)
    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[1]):
            Z[i, j] = density(X[i, j], Y[i, j])

    with open(resfcsv, 'w') as f:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]};{Y[i, j]};{Z[i, j]}\n")

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.savefig(resfim)

def combine_markerpos_statistics(folder):
    resf = open(os.path.join(folder, 'stat.csv'), 'w')
    annofld = os.path.join(folder, "data", "MARK")
    firs = True
    for file in tqdm(os.listdir(annofld)):
        with open(os.path.join(annofld, file)) as f:
            for line in f:
                if line.startswith("L"):
                    if not firs:
                        continue
                    firs = False
                resf.write(line)





def measured_markerpos_statistics(folder, outf=None, corr_dist=True):
    distance_folder = os.path.join(folder, "data", "MARK")
    resolution_folder = os.path.join(folder, "data", "OPT")
    files = [os.path.join(distance_folder, x) for x in os.listdir(distance_folder)]

    def find_reso(file):
        num = int(os.path.basename(file).split(".")[0][5:])
        setf = os.path.join(resolution_folder, f"settings{str(num).zfill(6)}.txt")
        # print("Match ", file, setf)
        with open(setf, 'r') as f:
            for line in f:
                # print(line)
                # input()
                if line.startswith('img_width'):
                    pr = line.strip().split("=")
                    len = float(pr[1][:-1])
                    return len

    ds = []
    fileds = {}
    file_reso = {}
    for file in tqdm(files):
        fileds[file] = []

        file_reso[file] = find_reso(file)

        with open(file, 'r') as f:
            for line in f:
                v = float(line.strip())
                if v > 0:
                    ds.append(v)
                    fileds[file].append(v)

    xs = []
    ys = []
    for k in fileds.keys():
        print("d", fileds[k])
        print("reso", file_reso[k])
        for d in fileds[k]:
            xs.append(file_reso[k])
            ys.append(d)

    plt.scatter(xs, ys)
    plt.xlabel("Img_Width (A)")
    plt.ylabel("Distance (nm)")
    plt.show()

    mu = np.average(ds)
    sig = np.std(ds)

    if outf is not None:
        df = os.path.join(os.path.dirname(outf), 'distance_imgw.csv')
        with open(df, 'w') as f:
            f.write("Img_W;Dist\n")
            for i in range(len(xs)):
                f.write(f"{xs[i]};{ys[i]}\n")

        with open(outf, 'w') as f:

            f.write("Total;" + ';'.join(fileds.keys()) + ';\n')
            for i in range(len(ds)):
                f.write(f"{ds[i]};" + ';'.join(
                    [str(x[i]) if i < len(x) else '' for x in [fileds[k] for k in fileds.keys()]]) + ';\n')
                # f.write("Total;" + ";".join([str(x) for x in ds]) + ";\n")
                # for k in fileds.keys():
                #     f.write(f"{k};{';'.join([str(x) for x in fileds[k]])};\n")

    f = lambda x: (1 / np.sqrt(2 * np.pi * sig * sig)) * np.exp(-0.5 * np.power((x - mu) / sig, 2))
    fv = np.vectorize(f)
    x = np.linspace(min(ds) * 0.8, max(ds) * 1.2, 1000)
    fig, ax = plt.subplots()
    ax.hist(ds, bins=int(0.7 * np.sqrt(len(ds))))
    # ax2 = ax.twinx()
    # ax2.plot(x, fv(x))
    # plt.tight_layout()
    plt.title(f"d={mu:.3f}nm, s={sig:.3f}nm")
    # st.pyplot(bbox_inches="tight")
    plt.show()

    if outf is not None:
        res = 100
        while res < 15000:
            start = time.perf_counter()
            folder = os.path.join(os.path.dirname(outf), f"Res{str(res).zfill(6)}")
            os.makedirs(folder, exist_ok=True)
            analyze_density(os.path.join(folder, "Plot.png"), os.path.join(folder, "list.csv"), xs, ys, res)
            print(f"Res: {res} --> {time.perf_counter() - start}")
            res = int(1.5 * res)


def task(runs, qu, num, last, px=None, fld=None):
    std_fak = 0
    spot_brought_mu = 55# 55  # 40 # 40 # 30  # 70 #
    spot_brought_sig = 8.25 * std_fak  # 6 # 6 # 5
    spot_width_mu = 100  # 200 # 200 # 500,  170 # 800
    spot_width_sig = 55 * std_fak  # 40 # 40 # 100, 30 # 150
    aequi_sigma = 0
    aequi_spots_missing = 0

    lss = lambda md : (1/178.44324) * md - 0.00288

    for runID in tqdm(range(runs if px is None else len(px)), disable=not last):
        if runID >= len(px):
            return

        location_ss_u = 30.45 / 87
        location_ss_l = 30.45 / 87

        if px is not None:
            molw, moll, molh, mard, immw = px[runID]

            mini = min(moll, molw)
            maxi = max(moll, molw)
            molw = mini
            moll = maxi
            imgs = 10000
            #px_a = pxl / imgs
            #px_a = 256/15000
            px_a = 64 / immw
            location_ss_u = lss(mard)
            location_ss_l = lss(mard)
            slfheight = molh
            width = molw
            length = moll
        else:
            px_a = np.random.uniform(256 / 18000, 256 / 12000)

        # length = np.random.normal(900, 20)
        # width = np.random.normal(700, 20)  # a


        # location_ss_u = 30.45/87
        # location_ss_l = 30.45/87
        ss_row_up = np.random.normal(location_ss_u * length, 20*std_fak)
        ss_row_low = -np.random.normal(location_ss_l * length, 20*std_fak)
        position_y_row_up = ss_row_up  # Lambda Function
        position_y_row_low = ss_row_low  # Lambda Function
        pos_y_sig = 10  * std_fak # a
        border_abst = 40  # a
        border_sigma = max(1e-2, 40 * std_fak)  # a
        border_abst_y = 30  # a
        border_sigma_y = max(1e-2, 7 * std_fak)  # a
        left_end = -0.45 * width  # a
        right_end = .45 * width  # a
        height_mu = 15  # a
        height_sig = 2 * std_fak  # a

        matrix_shape = (int(np.ceil(width * px_a)), int(np.ceil(length * px_a)))

        position_y = lambda upper: np.random.normal(position_y_row_up, pos_y_sig) if upper else np.random.normal(
            position_y_row_low, pos_y_sig)
        position_x_min = left_end + border_abst
        position_x_max = right_end - border_abst
        threshold = border_abst
        while (1 - (1 / (np.exp((threshold - border_abst) / border_sigma) + 1))) < 0.99:
            threshold += border_abst / 10
        threshold_y = border_abst_y
        while (1 - (1 / (np.exp((threshold_y - border_abst_y) / border_sigma_y) + 1))) < 0.99:
            threshold_y += border_abst_y / 10
        if std_fak == 0:
            no_of_spots = 5
        else:
            no_of_spots = random.randint(4, 6)
        d_position_x = (position_x_max - position_x_min) / (no_of_spots + 1)
        position_x = lambda x: np.random.normal(position_x_min + (x+1) * d_position_x, aequi_sigma)
        height = lambda: np.random.normal(height_mu, height_sig) + slfheight
        spot_brough = lambda: np.random.normal(spot_brought_mu, spot_brought_sig)  # a
        spot_width = lambda: np.random.normal(spot_width_mu, spot_width_sig)  # a # Was Uniform mu, sig
        spot_positions_upper = []
        spot_positions_lower = []

        for i in range(no_of_spots):
            if random.random() > aequi_spots_missing:
                spot_positions_upper.append(
                    (position_x(i), position_y(True), height(), spot_brough(), spot_width()))  #
        for i in range(no_of_spots):
            if random.random() > aequi_spots_missing:
                spot_positions_lower.append(
                    (position_x(i), position_y(False), height(), spot_brough(), spot_width()))  #

        mat_idcs = np.zeros(matrix_shape, dtype="object")
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                mat_idcs[i, j] = np.array([i, j])

        def height_spot(pos_mat, pos_spot, h, b, w):
            # if h < slfheight * px_a:
            #     return 0
            sig_x = -np.square(w) / np.log((h - slfheight) / h)
            sig_y = -np.square(b) / np.log((h - slfheight) / h)
            x_fac = np.exp(-np.square(pos_spot[0] - pos_mat[0]) / sig_x)
            y_fac = np.exp(-np.square(pos_spot[1] - pos_mat[1]) / sig_y)
            ret = h * x_fac * y_fac
            return ret + slfheight

        def border_abfall(x, y):

            x /= px_a
            y /= px_a  # x and y to a

            if x < 0:
                dx = +width / 2 + x
            else:
                dx = width / 2 - x

            if y < 0:
                dy = +length / 2 + y
            else:
                dy = length / 2 - y

            fx = (1 - (1 / (np.exp((dx - border_abst) / border_sigma) + 1)))
            fy = (1 - (1 / (np.exp((dy - border_abst_y) / border_sigma_y) + 1)))
            # print(f"x = {x}, dx={dx} -> f(x)={fx}")
            # print(f"y = {y}, dy={dy} -> f(y)={fy}")
            # print(f"({x, y}) -> f(x)={fx * fy}")

            if dx > threshold and dy > threshold_y:
                # print("---thrsh 1 ")
                return 1
            if dx < 0 or dy < 0:
                # print("---thrsh 0 ")
                return 0

            return fx * fy

        mat_u = np.zeros(matrix_shape)
        offset_x = matrix_shape[0] / 2  # ceil
        offset_y = matrix_shape[1] / 2 #ceil
        measured_yu = 0
        measured_yl = 0
        measured_vu = 0
        measured_vl = 0

        clpsL = np.zeros(mat_u.shape[1])
        clpsU = np.zeros(mat_u.shape[1])

        def visu_spot(shp, spt):
            x, y, h, b, w = spt
            #   print("Spot Positions: ", x, y)
            locmat = np.zeros(shp)
            for i in range(mat_u.shape[0]):
                for j in range(mat_u.shape[1]):
                    # print("Comparing {}, {} with {}, {}, writing to {}, {}".format(i - offset_x, j - offset_y, x, y, i, j))
                    locmat[i, j] += height_spot(np.array([i - offset_x, j - offset_y]) / px_a, np.array([x, y]), h,
                                                b, w) * border_abfall(
                        i - offset_x, j - offset_y)
            return locmat

        def height_spot2(pos_mat, pos_spot, h, b, w):
            # if h < slfheight * px_a:
            #     return 0
            sig_x = -np.square(w) / np.log((h - slfheight) / h)
            sig_y = -np.square(b) / np.log((h - slfheight) / h)
            x_fac = np.exp(-np.square(pos_spot[0] - pos_mat[0]) / sig_x)
            # y_fac = 1 if abs(pos_spot[1] - pos_mat[1]) < np.sqrt(sig_y) else 0  # np.exp(-np.square(pos_spot[1] - pos_mat[1]) / sig_y)
            # print(f"diff: {abs(pos_spot[1] - pos_mat[1])}, sig: {sig_y}")
            y_fac = np.exp(-np.square(pos_spot[1] - pos_mat[1]) / sig_y)
            ret = h * x_fac * y_fac
            return ret + slfheight

        def fkt(y, pos, sig):
            return np.exp(-(y/px_a - pos)**2 / sig**2) #  * border_abfall(y/px_a)
        def visu_spot2(shp, spt):
            x, y, h, b, w = spt
            locmat = np.zeros(shp)
            # print(f"y: {y}, w: {b}")
            sig_x = -np.square(w) / np.log((h - slfheight) / h)

           # x_facs = [np.exp(-np.square(((br- (shp[0] / 2)) / px_a) - x) / sig_x) for br in range(shp[0])]
           # plt.plot(x_facs)
           # plt.title(f"Xfacs x={x} w={w} med={(shp[0] / 2)}")
           # plt.show()
            # brdmat = np.zeros_like(locmat)
            # for br in range(shp[0]):
            #     for wd in range(shp[1]):
            #         brdmat[br, wd] = border_abfall()



            for br in range(shp[0]):
                x_fac = np.exp(-np.square(((br- (shp[0] / 2)) / px_a) - x) / sig_x)

                prof = x_fac * np.array([fkt(i - shp[1] / 2, y, b) for i in
                                  range(shp[1])])
                locmat[br, :] = prof

            # plt.imshow(locmat)
            # plt.title("Locmat")
            # plt.show()

            return locmat

        plt.switch_backend('tkAgg')
        for spot in spot_positions_upper:
            locmat = visu_spot2(mat_u.shape, spot)
            mat_u += locmat

            # plt.imshow(locmat)
            # plt.title("locmat")
            # plt.show()

            mc = locmat #  - slfheight * np.ones_like(locmat)
            mc = np.maximum(np.zeros_like(mc), mc)

            clps = np.sum(mc, axis=0)
            clpsU += clps

            xs = np.array([x for x in range(len(clps))])
            measured_yu += np.dot(xs, clps)
            measured_vu += np.sum(clps)


         #plt.imshow(mat_u)
         #plt.title("MatU")
         #plt.show()



        mat_l = np.zeros(matrix_shape)
        for spot in spot_positions_lower:
            locmat = visu_spot2(mat_l.shape, spot)
            mat_l += locmat

            mc = locmat # - slfheight * np.ones_like(locmat)
            mc = np.maximum(np.zeros_like(mc), mc)
            # mc = locmat
            clps = np.sum(mc, axis=0)
            clpsL += clps

            xs = np.array([x for x in range(len(clps))])

            measured_yl += np.dot(xs, clps)
            measured_vl += np.sum(clps)


        # plt.imshow(np.maximum(mat_l, mat_u))
        # plt.title(f"Mc, px={px[runID]}")
        # plt.show()

       # if fld is not None:# and not os.path.isfile(os.path.join(fld, f"{int(round(px[runID]))}.png")):
       #     plt.plot(clpsU, label="u")
       #     plt.plot(clpsL, label="l")
       #     plt.title(f"Res: {int(round(px[runID]))} IsL: {0.1*spot_positions_lower[0][1]:.3f} IsR: {0.1*spot_positions_upper[0][1]:.3f} L: {0.1*( (measured_yl/measured_vl) - offset_y)/px_a:.4f} -> R: {0.1*( (measured_yu/measured_vu) - offset_y)/px_a:.4f} -> D: {0.1*abs(measured_yu/measured_vu - measured_yl/measured_vl)/px_a:.4f}")
       #     # fld = r'C:\Users\seifert\Documents\MoveToD\DNAMeas\imgProjectionsBrought'
       #     os.makedirs(fld, exist_ok=True)
       #     plt.savefig(os.path.join(fld, f"{int(round(px[runID]))}.png"))
       #     plt.cla()
        # plt.show()



        if measured_vu > 0 and measured_vl > 0:
            measured_marker_upper = measured_yu / measured_vu
            measured_marker_lower = measured_yl / measured_vl
            measured_marker_distance = abs(measured_marker_upper - measured_marker_lower) / px_a
            qu.put((measured_marker_distance / 10, px_a))

            if fld is not None:
                with open(os.path.join(fld, "SpotPosis.csv"), 'a') as g:
                    spots_upper = [x[1] for x in spot_positions_upper]
                    spots_lower = [x[1] for x in spot_positions_lower]
                    diff = np.average(spots_upper) - np.average(spots_lower)
                    g.write(f"{px_a};{diff}\n")

            if fld is not None:
                os.makedirs(fld, exist_ok=True)
                with open(os.path.join(fld, "markDist.csv"), 'a') as g:
                    g.write(f"{molw};{moll};{molh};{mard};{immw};{measured_marker_distance / 10}\n")
    print(f"Process {num} done")


def estimate_pos_distro(savef):
    runs = 100000
    results = []
    pxas = []
    # fil = open(savef, 'w')
    # fil.close()
    os.makedirs(os.path.dirname(savef), exist_ok=True)
    df = os.path.join(os.path.dirname(savef), 'distance_imgw.csv')
    imwf = open(df, 'w')
    imwf.write("Img_W;Dist\n")
    write_buffer = 200
    plot_buffer = 0
    plot_steps = 500
    resolution = 500j
    plot_steps_total = 0

    thrds = 16
    rpt = int(np.ceil(runs / thrds))
    threads = []
    qu = multiprocessing.Queue()

    minpx = 1200
    maxpx = 1800

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # pxsi = np.linspace(maxpx, minpx, runs)
    # pxs = [x for x in pxsi] molw, moll, molh, mard, immw
    pxs = []
    for i in range(runs):
        molw = np.random.uniform(500, 700)
        moll = np.random.uniform(700, 1200)
        molh = np.random.normal(20, 0.75)
        mard = np.random.uniform(45, 75)
        # real_mard = np.random.uniform(45, 75)
        # mard = (real_mard * 700/moll)
        # moll = 700 * real_mard / mard
        immw = np.random.uniform(1000, 2000)
        pxs.append( (molw, moll, molh, mard, immw) )

    # pxs = np.linspace(30/87, 31/87, runs)

    # np.random.shuffle(pxs)
    pxlists = list(chunks(pxs, int(np.ceil(runs/thrds))))

    for i in range(len(pxlists)):
        print(f"Thread {i} --> {len(pxlists[i])}")

    # print(pxlists)
    # print(len(pxlists))
    # print(pxlists[0])
#
    # input()
    skipped  =False
    plot_anyway=False
    sc = 0
    profilef  = os.path.join(os.path.dirname(savef), 'profile')
    # profilef = None
    os.makedirs(os.path.dirname(os.path.dirname(profilef)), exist_ok=True)
    os.makedirs(os.path.dirname(profilef), exist_ok=True)
    os.makedirs(profilef, exist_ok=True)

    with open(os.path.join(profilef, "markDist.csv"), 'w') as g:
        g.write("molw;moll;molh;mard;immw;measured_marker_distance\n")
    for i in range(thrds):

        threads.append(Process(target=task, args=(rpt, qu, i, i == 0, pxlists[i], profilef)))
    for t in threads:
        t.start()
    with tqdm(total=runs, disable=True) as pbar:
        while len(results) < runs-5:

            if qu.qsize() < write_buffer and not skipped:
                skipped = True
                c = qu.qsize()
                print("waiting")
                time.sleep(2)
                continue
            else:
                if skipped:
                    sc += 1
                    print("Skipped")
                    if sc < 3:
                        print("PAW")
                        plot_anyway = True
                    skipped = False
                else:
                    sc = 0

            imwf = open(df, 'a')

            # with open(savef, 'a') as fil:
            # print("Write to ", savef)
            for i in range(write_buffer):
                if qu.qsize() == 0:
                    break
                v, px_a = qu.get()
                results.append(v)
                pxas.append(px_a)
                # fil.write(str(v) + "\n")
                imwf.write(f"{pxas[i]};{results[i]}\n")
            pbar.update(i)
            plot_buffer += i
            plot_steps_total += i

            ys = results
            xs = pxas
            mu = np.average(ys)
            sig = np.std(ys)

            # print("Plot")
            # plt.scatter(xs, ys)
            # plt.xlabel("Img_Width (A)")
            # plt.ylabel("Distance (nm)")
            # plt.show()

            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)

            def gf(x, y, px, py, sig=0.001):
                return np.exp( -np.square((x-px) / (xmax - xmin))/sig) * np.exp(-np.square((y-py) / (ymax - ymin))/sig)

            if plot_buffer > plot_steps or plot_anyway:
                plot_buffer -= plot_steps
                num = plot_steps_total // plot_steps
                fileplt = os.path.join(os.path.dirname(savef), f'Plot_{plot_steps_total}.png')

                # xy = np.vstack([xs, ys])
                # z = gaussian_kde(xy)(xy)
                X, Y = np.mgrid[min(xs):max(xs):resolution, min(ys):max(ys):resolution]
                positions = np.vstack([X.ravel(), Y.ravel()])
                values = np.vstack([xs, ys])

                # resmat = np.zeros((int(abs(resolution)), int(abs(resolution))))
                # for i in tqdm(range(len(xs)), desc="imaging"):
                #     arr = gf(X, Y, xs[i], ys[i])
                #     resmat += arr

                # plt.imshow(resmat)
                # plt.title("Resmat")
                # plt.show()
                fig, ax = plt.subplots()
                try:
                    kernel = gaussian_kde(values)
                    Z = np.reshape(kernel(positions).T, X.shape)
                # plt.imshow(Z)
                # plt.show()
                    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                              extent=[min(xs), max(xs), min(ys), max(ys)], aspect='auto')
                except np.linalg.LinAlgError as e:
                    pass
                ax.set_xlabel("Image Resolution")
                ax.set_ylabel("Marker Distance")
                ax.scatter(xs, ys, s=5)
                plt.savefig(fileplt)

            if plot_anyway:
                break

        imwf.close()

        # print("Saved as ", df)


    res = 100
    while res < 15000:
        start = time.perf_counter()
        folder = os.path.join(os.path.dirname(savef), f"Res{str(res).zfill(6)}")
        os.makedirs(folder, exist_ok=True)
        analyze_density(os.path.join(folder, "Plot.png"), os.path.join(folder, "list.csv"), xs, ys, res)
        print(f"Res: {res} --> {time.perf_counter() - start}")
        res = int(1.5 * res)


    f = lambda x: (1 / np.sqrt(2 * np.pi * sig * sig)) * np.exp(-0.5 * np.power((x - mu) / sig, 2))
    fv = np.vectorize(f)
    x = np.linspace(min(ys) * 0.8, max(ys) * 1.2, 1000)
    fig, ax = plt.subplots()
    ax.hist(ys, bins=int(0.7 * np.sqrt(len(ys))))
    # ax.set_xlim(35, 90)
    # ax2 = ax.twinx()
    # ax2.plot(x, fv(x))
    # plt.tight_layout()
    plt.title(f"d={mu:.3f}nm, s={sig:.3f}nm")
    # st.pyplot(bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plt.switch_backend('TkAgg')

    combine_markerpos_statistics(r'C:\Users\seifert\PycharmProjects\STM_Simulation\bildordner\SingleDNA_V2\Set178_UniRelPosSS')
    assert  'm' == 'n'
    # px = [x for x in range(100, 2000,5)]
    # task(1000, multiprocessing.Queue(), 1, True, px)
    # assert 5 == 9

    # estimate_pos_distro(r"C:\Users\seifert\Documents\MoveToD\DNAMeas\imgWMarkCorr_0505_V94_SameMolW\location_range2.csv")
    # assert 1 == 2
    f = r'C:\Users\seifert\PycharmProjects\STM_Simulation\bildordner\SingleDNA_V2\Set149_EvalYOLO_63'
    measured_markerpos_statistics(f,
                                  os.path.join(f, 'stat.csv'))
    assert 'h' == 8
    resize_trainset(r"C:\Users\seifert\PycharmProjects\DNA_Measurement\SS_DNA_Train\68nm_5k_64px",
                    r"C:\Users\seifert\PycharmProjects\DNA_Measurement\SS_DNA_Train\68nm_5k_64px_RESC50",
                    64, 50)
    assert 4 == 5

    matplotlib.use("TkAgg")
    resize_folder(r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Small50px_5k\Train_pt\sxm\numpy",
                  r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Upscaled64\Train_pt\sxm\numpy")
    resize_folder(r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Small50px_5k\Test_pt\sxm\numpy",
                  r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Upscaled64\Test_pt\sxm\numpy")
    resize_mask(r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Small50px_5k\Train_pt\data\numpy",
                r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Upscaled64\Train_pt\data\numpy")
    resize_mask(r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Small50px_5k\Test_pt\data\numpy",
                r"C:\Users\seife\PycharmProjects\DNA_Measurement3\SS_DNA_Train\Upscaled64\Test_pt\data\numpy")
    exit()
    # make_angle_list2(
    #    "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try613_SynthMix_FIT_UseU_True_TestLabelProviding\\ss_labels",
    #    "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix1\\UnetPredictions.csv")
    # exit(0)

    # make_angle_list2(
    #     "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try616_SynthMix2_FIT_UseU_False_TestLabelProviding\\provided_labels_yolo",
    #     "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\Input.csv")
    # make_angle_list2(
    #     "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try616_SynthMix2_FIT_UseU_False_TestLabelProviding\\provided_labels_resc",
    #     "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\YoloPred.csv")
    # make_angle_list2(
    #     "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try616_SynthMix2_FIT_UseU_False_TestLabelProviding\\ss_labels",
    #     "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\UNetPred.csv")

    fwhm = 10
    resolution = 500
    smooth_thetas("D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\Input.csv",
                  "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\InputSmooth.csv",
                  resolution=resolution,
                  fwhm=fwhm)
    smooth_thetas("D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\YoloPred.csv",
                  "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\YoloPredSmooth.csv",
                  resolution=resolution,
                  fwhm=fwhm)
    smooth_thetas("D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\UNetPred.csv",
                  "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\UNetPredSmooth.csv",
                  resolution=resolution,
                  fwhm=fwhm)
    recalc_effangle(
        "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try616_SynthMix2_FIT_UseU_False_TestLabelProviding\\quality\\quality_indcs.csv",
        "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\QI_temp.csv")
    smooth_thetas("D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\QI_temp.csv",
                  "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix2\\WithUres.csv",
                  resolution=resolution,
                  fwhm=fwhm)
    # smooth_thetas("D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix1\\YoloDetections.csv",
    #               "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix1\\YoloDetectionsSmooth.csv",
    #               resolution=resolution,
    #               fwhm=fwhm)

    exit(0)
    make_angle_list2(
        "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try613_SynthMix_FIT_UseU_False_TestLabelProviding\\ss_labels",
        "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix1\\UnetPredictions.csv")
    exit(0)
    recalc_effangle(
        "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix1\\quality_indcs.csv",
        "D:\\Dateien\\KI_Speicher\\EvalChainDS\\Doku\\AngleLists\\SynthMix1\\WithUres.csv")
    exit()
#
# combine_angles("C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA_T_AngleTest\\Set26\\data\\ANG",
#                "C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA_T_AngleTest\\Set26\\data\\ANG\\Combined.csv")
# combine_angles(
#     "C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA_T_AngleTest\\Set26\\data_flip\\ANG",
#     "C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA_T_AngleTest\\Set26\\data_flip\\ANG\\Combined.csv")
#
# # combine_angles("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\SyntheticY\\data_flip\\ANG",
# #                "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\SyntheticY\\data_flip\\ANG\\Combined.csv")
# exit()
#
# ste = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NewHS_L90_V3\\Test_pt"
# sta = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NewHS_L90_V3\\Train_pt"
# rte = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NewHS_V3_2Markers\\Test_pt"
# rtr = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NewHS_V3_2Markers\\Train_pt"
# filter2markers(sta, rtr)
# filter2markers(ste, rte)
