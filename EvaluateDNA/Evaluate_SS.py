import copy
import multiprocessing
import os
import sys
import time

import cv2
import scipy.signal
from PIL import Image, ImageOps
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import cm  # Colormaps
from PIL import ImageDraw
from multiprocessing import Process
from scipy.optimize import curve_fit

SHOW_ALL=False
matplotlib.use("Agg")
np.seterr(all="ignore")

def extract_arr(fn):
    if os.path.splitext(fn)[1] == ".npy":
        arr = np.load(fn, allow_pickle=True)
        arr = 255 * (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))
        arr = np.transpose(arr)
        if SHOW_ALL:
            plt.imshow(arr)
            plt.title(fn)
            plt.show()
        return arr
    img = Image.open(fn)
    img = ImageOps.grayscale(img)
    arr = np.array(img)
    return arr

def hoshen_koppelmann(arr):
    """
    Implemenierung des Hoshen Koppelmann-Algorithmus
    """
    # Neues Array mit Padding links und oben
    padded_array = np.pad(arr, (1, 0))
    length0 = arr.shape[0]
    length1 = arr.shape[1]
    size = length0 * length1
    # Erstellung der Matrix fuer labels
    label = np.zeros(np.shape(padded_array), dtype=np.int32)
    # Zaehl-Array. None um Array-Indizes bei 1 zu beginnen
    n = [None]
    # Cluster-Index
    c = 1
    # Setze Label fuer linken und oberen Rand
    for i in range(length1 + 1):
        label[0, i] = size
    for i in range(length0+1):
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
    for i in range(1, length0 + 1):
        for j in range(1, length1 + 1):
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

def find_crops(arr, show_all=SHOW_ALL):


    if type(show_all) == str:
        save_all = show_all
        show_all = True
    else:
        save_all = None


    def cog(index, arr):
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

    def clip_numbers(x):
        return min(maxshp, max(0, x))

    def build_clusters(cogs):
        """

        :param cogs: i_cog, j_cog, mini, maxi, minj, maxj = cog
        :return:
        """

        start_cogs = len(cogs)
        distances = np.zeros((start_cogs, start_cogs))
        for i in range(start_cogs):
            distances[i, i] = 0
            pi = np.array([cogs[i][0], cogs[i][1]])
            for j in range(i, start_cogs):
                pj = np.array([cogs[j][0], cogs[j][1]])
                distances[i, j] = np.linalg.norm(pi - pj)

        distances = (distances + np.transpose(distances)) / 2
        if show_all:
            plt.imshow(distances)
            plt.title("Distance matrix")
            if save_all is not None:
                plt.savefig(os.path.join(save_all, "{}_DistanceMat.png".format(len(os.listdir(save_all)))))
            else:
                plt.show()
            plt.close()

        possible_distances = distances.flatten()
        possible_distances = np.unique(possible_distances)
        possible_distances = np.sort(possible_distances)
        for cluster_dist in possible_distances:
            cluster_centers = []
            for i in range(start_cogs):
                if len(cluster_centers) == 0:
                    cluster_centers.append([i])
                #  print(cluster_centers)

                else:
                    appended = False
                    for clu_idx in range(len(cluster_centers)):
                        clu = cluster_centers[clu_idx]
                        for pt in clu:
                            if distances[i, pt] <= cluster_dist:
                                cluster_centers[clu_idx].append(i)
                                # print(cluster_centers)
                                appended = True
                                break
                        if appended:
                            break
                    if not appended:
                        cluster_centers.append([i])
                        # print(cluster_centers)
                if len(cluster_centers) > 2:
                    break

            if len(cluster_centers) == 2:
                break

        return_cogs = []
        for cluster in cluster_centers:
            if len(cluster) == 1:
                return_cogs.append(cogs[cluster[0]])
            else:
                i_cogs = []
                j_cogs = []
                minis = []
                maxis = []
                minjs = []
                maxjs = []
                for idx in cluster:
                    i_cog, j_cog, mini, maxi, minj, maxj = cogs[idx]
                    i_cogs.append(i_cog)
                    j_cogs.append(j_cog)
                    minis.append(mini)
                    maxis.append(maxi)
                    minjs.append(minj)
                    maxjs.append(maxj)

                mini = min(minis)
                maxi = max(maxis)
                minj = min(minjs)
                maxj = max(maxjs)
                i_cog = (maxi + mini) / 2
                j_cog = (maxj + minj) / 2
                return_cogs.append((i_cog, j_cog, mini, maxi, minj, maxj))

        # assert len(return_cogs) == 2
        return return_cogs

    if type(arr) is str:
        arr = extract_arr(arr)

    if np.amax(arr) == np.amin(arr):
        print("No Marker Found")
        return None

    maxshp = arr.shape[0]
    arr = arr / np.amax(arr)
    arr = np.floor(arr)
    arr = arr.astype(int)

    # plt.imshow(arr)
    # plt.title("before HK")
    # plt.show()

    clusters = hoshen_koppelmann(arr)

    classes = np.unique(clusters.flatten())
    labels_ok=False
    while not labels_ok:
        labels_ok = True
        for classi in classes:
            agw = np.argwhere(clusters==classi)
            if len(agw) == 1:
                arr[agw[0][0], agw[0][1]] = 0
                if show_all:
                    plt.imshow(arr)
                    if save_all is not None:
                        plt.savefig(os.path.join(save_all, "{}_FindClusters.png".format(len(os.listdir(save_all)))))
                    else:
                        plt.show()
                    plt.close()

                clusters = hoshen_koppelmann(arr)
                classes = np.unique(clusters.flatten())
                labels_ok = False
                break

    if show_all:
        plt.imshow(arr)
        plt.title("Loaded Label")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_LoadedLabel.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

        plt.imshow(clusters)
        plt.title("Clusters")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_Clusters.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    indx = 1
    # Center of gravity for each cluster
    cogs = []
    while True:
        # if len(np.argwhere(clusters==indx)) == 1:
        #     print("Skipping indx ", indx)
        #     indx += 1
        #     continue
        erg = cog(index=indx, arr=clusters)
        if erg == False:
            break
        cogs.append(erg)
        indx += 1

    # assert len(cogs) == 2

    if len(cogs) > 2:
        cogs = build_clusters(cogs)

    crps = []
    for cog in cogs:
        i_cog, j_cog, mini, maxi, minj, maxj = cog

        width = int(max([abs(np.ceil(i_cog) - mini), abs(maxi - np.floor(i_cog)), abs(np.ceil(j_cog) - minj),
                         abs(maxj - np.floor(j_cog))]))

        crps.append((int(clip_numbers(np.floor(i_cog - width))), int(clip_numbers(np.ceil(i_cog + width))),
                     int(clip_numbers(np.floor(j_cog - width))), int(clip_numbers(np.ceil(j_cog + width)))))

    return crps, cogs,arr


def coarse_crop_img_lbl(image, label, crop, show_all=SHOW_ALL):

    if type(show_all) == str:
        save_all = show_all
        show_all = True
    else:
        save_all = None

    img = extract_arr(image) if type(image) is str else image
    lbl = extract_arr(label) if type(label) is str else label
    lblmax = np.amax(lbl)
    only_ss = lambda x: 1 if x == lblmax else 0
    only_ss = np.vectorize(only_ss)
    lbl = only_ss(lbl)

    a, b, c, d = crop
    img_c = img[a:b+1, c:d+1] # +1
    if show_all:
        plt.imshow(lbl)
        plt.title("Total Label")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_TotalLabel.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    lbl_c = lbl[a:b+1, c:d+1]

    if show_all:
        plt.imshow(lbl_c)
        plt.title("lbl_c")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_Label_c.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

        plt.imshow(img_c)
        plt.title("img_c")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_Image_c.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    return img_c, lbl_c, crop


def find_border(lbl, show_all=SHOW_ALL):

    if type(show_all) == str:
        save_all = show_all
        show_all = True
    else:
        save_all = None


    brd_mat = np.zeros(lbl.shape)
    max_j = lbl.shape[1]
    max_i = lbl.shape[0]

    def nbr_idcs(i, j):
        idxs = []
        for a in [-1, 0, 1]:
            for b in [-1, 0, 1]:
                if 0 <= a + i < max_i:
                    if 0 <= b + j < max_j:
                        idxs.append((a + i, b + j))

        return idxs

    for i in range(lbl.shape[0]):
        for j in range(lbl.shape[1]):
            if lbl[i, j] == 1:
                nis = nbr_idcs(i, j)
                for a, b in nis:
                    if lbl[a, b] == 0:
                        brd_mat[a, b] = 1

    if show_all:
        plt.imshow(lbl)
        plt.title("lbl")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_Label.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()
        plt.imshow(brd_mat)
        plt.title("Border Mat")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_BorderMat.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    return brd_mat


def crop_img(crpd_img, crps_lbl, show_all=SHOW_ALL):

    if type(show_all) == str:
        save_all = show_all
        show_all = True
    else:
        save_all = None


    if show_all:
        plt.imshow(crpd_img)
        plt.title("Before crop_img")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_BeforeCropImg.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()
    try:
        max_label = np.amax(crps_lbl)
    except ValueError:
        return [], 0

    idcs = np.argwhere(crps_lbl != max_label)

    border = find_border(crps_lbl, show_all=show_all if save_all is None else save_all)
    border_idcs = np.argwhere(border == 1)
    brd_values = []
    # print("Crpd img shape: ", crpd_img.shape)
    # plt.imshow(crpd_img)
    # plt.title("Crpd Img")
    # plt.show()
    # plt.imshow(crps_lbl)
    # plt.title("Cdps label")
    # plt.show()
    for a, b in border_idcs:
        try:
            brd_values.append(crpd_img[a, b])
        except IndexError:
            print("Index error in brd_values")

    border_average = np.average(brd_values)

    crpd_img = crpd_img - border_average

    for idx_i, idx_j in idcs:
        crpd_img[idx_i, idx_j] = 0

    if show_all:
        plt.imshow(crpd_img)
        plt.title("After crop_img")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_AfterCropImg.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()
    return crpd_img, border_average


def gauss_fit(crpd_img, accuracy=4000, show_all=SHOW_ALL):

    if type(show_all) == str:
        save_all = show_all
        show_all = True
    else:
        save_all = None

    cpy = copy.deepcopy(crpd_img)

    def multivariate_normal(x, d, mean, covariance):
        """pdf of the multivariate normal distribution."""
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
                np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

    def generate_surface(mean, covariance, d):  # ToDo: Not sure if nbofx und y vertauscht sind
        """Helper function to generate density surface."""
        nb_of_x = crpd_img.shape[1]
        # print("Prev ", mean)
        # print(mean.shape)
        meancp = np.array([mean[1][0], mean[0][0]]).reshape((2, 1))

        # print("After: ", meancp)
        nb_of_y = crpd_img.shape[0]  # grid size
        x1s = np.linspace(0, crpd_img.shape[1], num=nb_of_y)
        x2s = np.linspace(0, crpd_img.shape[0], num=nb_of_x)
        x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
        pdf = np.zeros((nb_of_x, nb_of_y))
        # Fill the cost matrix for each combination of weights
        for i in range(nb_of_x):
            for j in range(nb_of_y):
                try:
                    pdf[i, j] = multivariate_normal(
                        np.array([[x1[i, j]], [x2[i, j]]]),
                        d, meancp, covariance)
                except AttributeError:
                    pass
        return x1, x2, pdf  # x1, x2, pdf(x1,x2)

    if show_all:
        plt.imshow(crpd_img)
        plt.title("Start Gauss fit")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_StartGaussFit.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    mini = np.amin(crpd_img)
    samples = []

    total = np.sum(crpd_img) - (len(np.argwhere(crpd_img != 0))) * mini
    # print(total)
    if total == 0:
        print("Total of zero")
        return -1
    frac = accuracy / total

    # print(frac)
    for i in range(crpd_img.shape[0]):
        for j in range(crpd_img.shape[1]):
            if crpd_img[i, j] == 0:
                continue

            weight = crpd_img[i, j]  # Hier ohne abzug mini
            if weight < 0:
                pass
            try:
                for k in range(int(weight * frac)):
                    samples.append((i, j))
            except ValueError:  # if frac is none
                return None
    # print("Len Samples: ", len(samples))
    samples = np.array(samples)
    xs = [p[0] for p in samples]
    ys = [p[1] for p in samples]
    mn = [[np.average(xs) - 1], [np.average(ys) - 1]]  # War ohne -1
    mn = np.transpose(mn)

    Y = np.transpose(samples)
    # print(Y.shape)
    cov = np.cov(Y).transpose()
    # ("Korrelationskoeffizient: ", (cov[0, 1] / (cov[0, 0] * cov[1, 1])))

    # Plot bivariate distribution
    mn = np.transpose(mn)  # War drin
    # print("Mn", mn)
    # print("cov", cov)
    try:
        x1, x2, p = generate_surface(mn, cov, 2)
    except np.linalg.LinAlgError as e:
        print("Accuracy set too low or array empty")
        if str(e) == "Singular Matrix":
            return -1
        return - 1

    meancp = np.array([mn[1][0] - 1, mn[0][0] - 1]).reshape((2, 1))  # Weils geht

    # print("Mean: ", meancp) # ToDo: Check difference
    # print("Argmax: ", np.unravel_index(np.argmax(p), p.shape)) # ToDo: Check difference

    func_from_covariance(cov, show_all if save_all is None else save_all)


    # Test properties of Korrelationsmatrix Bronstein 854

    # print("Winkel: ", 360 * angle / (2 * np.pi))

    # p = np.fliplr(p)
    sigmax = cov[0, 0]
    sigmaxy = cov[0, 1]
    sigmay = cov[1, 1]
    rhoxy = sigmaxy / np.sqrt(sigmay * sigmax)
    meanx = mn[0][0]
    meany = mn[1][0]
    angle = np.arccos(rhoxy)

    f = lambda x, y: (1 / (2 * np.pi * np.sqrt(sigmax) * np.sqrt(sigmay) * np.sqrt(1 - rhoxy ** 2))) * np.exp(
        -(1 / (2 * (1 - rhoxy ** 2))) * (((x - meanx) ** 2 / sigmax) - 2 * (
                (rhoxy * (x - meanx) * (y - meany)) / (np.sqrt(sigmax) * np.sqrt(sigmay))) + (y - meany) ** 2 / (
                                             sigmay))
    )

    if show_all:

        testmat = np.zeros(crpd_img.shape)
        for i in range(crpd_img.shape[0]):
            for j in range(crpd_img.shape[1]):
                testmat[i, j] = f(i, j)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # temp = p.transpose()
        # temp = np.fliplr(temp)
        ##temp = np.flipud(temp)
        # temp = temp[::-1,::-1]
        # con = ax2.contourf(x1, x2, temp, 33, cmap=cm.YlGnBu)
        # ax2.set_xlim(left=0, right=crpd_img.shape[0])
        # ax2.set_ylim(bottom=0, top=crpd_img.shape[1])
        ax1.imshow(crpd_img)

        # plt.gca().set_aspect('equal', adjustable='box')
        ax2.imshow(testmat)
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_GaussFitTestMat.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    # plt.imshow(p)
    # plt.title("1")
    # plt.show()
    # p = np.fliplr(p)
    # plt.imshow(p)
    # plt.title("2")
    # plt.show()
    # p = np.rot90(p)
    # plt.imshow(p)
    # plt.title("3")
    # plt.show()
    # p = np.rot90(p)
    # plt.imshow(p)
    # plt.title("4")
    # plt.show()
    # p = np.rot90(p)
    # plt.imshow(p)
    # plt.title("5")
    # plt.show()
    # p = np.transpose(p)
    # plt.imshow(p)
    # plt.title("6")
    # plt.show()

    testmat = np.zeros(crpd_img.shape)
    for i in range(crpd_img.shape[0]):
        for j in range(crpd_img.shape[1]):
            testmat[i, j] = f(i, j)

    testmat = (testmat - np.min(testmat)) / (np.max(testmat) - np.min(testmat))
    if show_all:
        plt.imshow(testmat)
        plt.title("Result gauss fit")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_ResGaussFit.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    # print("Got to End with ", meancp, testmat, cov)

    return meancp, testmat, cov


def func_from_covariance(cov, show_all=SHOW_ALL):

    if type(show_all) == str:
        save_all = show_all
        show_all = True
    else:
        save_all = None

    sigmax = cov[0, 0]
    sigmaxy = cov[0, 1]
    sigmay = cov[1, 1]
    rhoxy = sigmaxy / np.sqrt(sigmay * sigmax)
    meanx = 0
    meany = 0

    # print("Angle: ", np.arccos(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])) * 180 / np.pi)
    # mInter = np.tan(np.arccos(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])) - np.pi/2)
    m1 = np.sqrt(cov[0, 0] / cov[1, 1]) * rhoxy
    m2 = np.sqrt(cov[1, 1] / cov[0, 0]) * rhoxy

    if abs(m1) < abs(m2):
        m = np.tan(np.pi / 2 - np.arctan(m1))
        # print("m1")
    else:
        # print("m2")
        m = m2

    if show_all:
        f = lambda x, y: np.exp(
            -(1 / (2 * (1 - rhoxy ** 2))) * (((x - meanx) ** 2 / sigmax) - 2 * (
                    (rhoxy * (x - meanx) * (y - meany)) / (np.sqrt(sigmax) * np.sqrt(sigmay))) + (y - meany) ** 2 / (
                                                 sigmay))
        )
        testmat = np.zeros((500, 500))
        xs = np.linspace(-20, 20, 500)
        ys = np.linspace(-20, 20, 500)
        for i in range(500):
            for j in range(500):
                testmat[i, j] = f(xs[i], ys[j])
                if abs(m * xs[i] - ys[j]) < 0.1:
                    testmat[i, j] *= 4
                elif abs(m1 * xs[-i] - ys[j]) < 0.1:
                    testmat[i, j] *= 2
                elif abs(m2 * xs[i] - ys[j]) < 0.1:
                    testmat[i, j] *= 2

                else:
                    pass

        plt.imshow(testmat)
        plt.title("Func from cov")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_FuncFromCov.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    return m


def get_distance_2(image, label, saveas=None, show_all=SHOW_ALL, transpose_res=True, verbose=True, fit_fp=None,
                   export_hists=None, pp_img_folder=None):
    # Load image and label
    img = extract_arr(image) if type(image) is str else image
    lbl = extract_arr(label) if type(label) is str else label

    if type(show_all) == str:
        save_all = show_all
        show_all = True
    else:
        save_all = None

    if show_all:
       plt.imshow(lbl)
       plt.title("Label")
       if save_all is not None:
           plt.savefig(os.path.join(save_all, "{}_Label.png".format(len(os.listdir(save_all)))))
       else:
           plt.show()
       plt.close()


    if export_hists is not None:
        hist_fp = os.path.join(export_hists, os.path.basename(image).split(".")[0])
        os.makedirs(hist_fp, exist_ok=True)
        plt.imsave(os.path.join(hist_fp, "image.png"), img, cmap="gray")
        plt.close()
        plt.imsave(os.path.join(hist_fp, "label.png"), lbl, cmap="gray")
        plt.close()

        ori = copy.deepcopy(lbl)
        f = lambda x : int(round(x/200))
        f = np.vectorize(f)
        ori = f(ori)
        matti = ori * img
        vals = []
        idcs = np.argwhere(matti > 0)
        for idx in idcs:
            vals.append(matti[idx[0], idx[1]])
        plt.hist(vals, bins = int(np.sqrt(len(vals))))
        plt.title("Not PP Histogram")
        plt.savefig(os.path.join(hist_fp, "hist.png"))
        plt.cla()
        plt.clf()
        plt.close()
        if pp_img_folder is not None:
            fld_pp = os.path.join(hist_fp, "PP")
            os.makedirs(fld_pp, exist_ok=True)
            plt.imsave(os.path.join(fld_pp, "label.png"), lbl, cmap="gray")
            pp_image = extract_arr(
                os.path.join(pp_img_folder, os.path.basename(image).split(".")[0] + ".npy")
                if os.path.isfile(os.path.join(pp_img_folder, os.path.basename(image).split(".")[0] + ".npy"))
                else os.path.join(pp_img_folder, os.path.basename(image).split(".")[0] + ".png"))
            pp_image = pp_image.transpose()
            plt.imsave(os.path.join(fld_pp, "image.png"), pp_image, cmap="gray")
            plt.close()

            matti = ori * pp_image
            vals = []
            for idx in idcs:
                vals.append(matti[idx[0], idx[1]])
            plt.hist(vals, bins=int(np.sqrt(len(vals))))
            plt.title("PP Histogram")
            plt.savefig(os.path.join(fld_pp, "hist.png"))
            plt.cla()
            plt.clf()
            plt.close()
    # Filter only SS label
    lblmax = np.amax(lbl)
    only_ss = lambda x: 1 if x == lblmax else 0
    only_ss = np.vectorize(only_ss)
    lbl = only_ss(lbl)

    # Initialize result
    red_image = np.zeros((img.shape[0], img.shape[1], 3))
    red_image[:, :, 0] = img
    red_image[:, :, 1] = img
    red_image[:, :, 2] = img

    # Split both Markers to obtain cog, weight, boundary

    fc_ret = find_crops(label, show_all=show_all if save_all is None else save_all)
    if fc_ret is None:
        return -1, -1
    crops, cogs, label_mod = fc_ret # Mdeified label without single point markers

    # plt.imshow(label_mod)
    # plt.title("Labelmod")
    # plt.show()


    if len(cogs) == 1:
        if verbose:

            print("Only one Marker found")
        return -1, -1


    # print("Cogs: ", cogs)
    bases = []
    weights = []
    maxima = []
    maskd_markers = []
    for crp in crops:
        img_loc, lbl_loc, crop = coarse_crop_img_lbl(image, label_mod, crp, show_all=show_all if save_all is None else save_all)
        _, base = crop_img(img_loc, lbl_loc, show_all=show_all if save_all is None else save_all)



        masked_makrer = img_loc * lbl_loc
        masked_makrer = masked_makrer.astype(float)
        masked_makrer -= base * lbl_loc

        # plt.imshow(masked_makrer)
        # plt.title("masked Marker")
        # plt.show()

        mm2 = img_loc * lbl_loc
        if show_all:
            prev_avg = np.average(mm2)
            # print("Prev avg", prev_avg)
            plt.imshow(mm2)
            plt.title("MM2")
            if save_all is not None:
                plt.savefig(os.path.join(save_all, "{}_MM2.png".format(len(os.listdir(save_all)))))
            else:
                plt.show()
            plt.close()

        values = sorted(np.unique(mm2.flatten()))
        base = values[int(0.1 * len(values))]
        if show_all:
            mm2 = mm2.astype(float)
            mm2 -= base * lbl_loc
            # print("90pct avg", np.average(mm2))
            plt.imshow(mm2)
            if save_all is not None:
                plt.savefig(os.path.join(save_all, "{}_MM2_minusBase.png".format(len(os.listdir(save_all)))))
            else:
                plt.show()
            plt.close()

        masked_makrer = mm2
        maskd_markers.append(mm2)

        bases.append(base)




        if show_all:
            plt.imshow(masked_makrer)
            plt.title("Masked Marker: " + str(np.average(masked_makrer)))
            if save_all is not None:
                plt.savefig(os.path.join(save_all, "{}_MaskedMarker.png".format(len(os.listdir(save_all)))))
            else:
                plt.show()
            plt.close()

        weight = 1 / np.sum(masked_makrer)
        weights.append(weight)
        maxima.append(np.amax(masked_makrer))

    # Find matrices to subtract bases and reweight
    basemat = np.zeros(img.shape)
    weightmat = np.zeros(img.shape)
    sspossitions = np.argwhere(label_mod != 0)
    if len(sspossitions) <= 1:
        if verbose:
            print("Only single pixel marker")
        return -1, -1

    # # Filter for single point markers
    # start = time.perf_counter()
    # print("Start Filtering: ", len(sspossitions))
    # ok_indcs = []
    # for i in range(len(sspossitions)):
    #     if i in ok_indcs:
    #         continue
    #     for j in range(i+1, len(sspossitions)):
    #         if np.sqrt( (sspossitions[i][0] - sspossitions[j][0])**2 + (sspossitions[i][1] - sspossitions[j][1])**2) < 1.9:
    #             ok_indcs.append(i)
    #             ok_indcs.append(j)
    #             break

    # sspossitions_2 = []
    # for indx in ok_indcs:
    #     sspossitions_2.append(sspossitions[indx])
    # sspossitions = sspossitions_2
    # print("Filtering: ", time.perf_counter() - start, len(sspossitions))

    # print("Cogs: ", cogs)
    cog1 = np.array([cogs[0][0], cogs[0][1]])
    cog2 = np.array([cogs[1][0], cogs[1][1]])

    for pos in sspossitions:
        posi = np.array([pos[0], pos[1]])
        if np.linalg.norm(posi - cog1) < np.linalg.norm(posi - cog2):
            basemat[pos[0], pos[1]] = bases[0]
            weightmat[pos[0], pos[1]] = weights[0]
        else:
            basemat[pos[0], pos[1]] = bases[1]
            weightmat[pos[0], pos[1]] = weights[1]

    if show_all:
        plt.imshow(basemat, cmap="gray")
        plt.title("Basemat")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_Basemat.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


        plt.imshow(weightmat)
        plt.title("weights")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_Weights.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    # Construct total Image with only masked markers
    fit_base = img * label_mod
    fit_base = fit_base.astype(float)
    fit_base -= basemat

    if show_all:
        plt.imshow(fit_base)
        plt.title("Not Weighted Base")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_NotweightedBase.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    weighted_base = fit_base * weightmat
    if show_all:
        plt.imshow(weighted_base, cmap="gray")
        plt.title("Weighted base")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_WeightedBase.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    # testing Convolution of base
    # ks = 9
    # xs = np.linspace(-1, 1, ks)
    # ys = np.linspace(-1, 1, ks)
    # kernel = np.zeros((ks, ks))
    # sig = 0.3
    # for i in range(len(xs)):
    #     for j in range(len(ys)):
    #         kernel[i, j] = np.exp(-(xs[i]**2 / sig) - (ys[j]**2 / sig))
    # kernel /= np.sum(kernel)


    # weighted_base = scipy.signal.convolve2d(weighted_base, kernel, mode='same')



    # Fitting of things
    def gaussian(x, y, th, g, sx, sy, x0, y0):
        return g * np.exp(-(
                ((np.cos(th) ** 2) / (2 * sx) + (np.sin(th) ** 2) / (2 * sy)) * (x - x0) ** 2
                + 2 * ((np.sin(2 * th)) / (4 * sx) - (np.sin(2 * th)) / (4 * sy)) * (x - x0) * (y - y0)
                + ((np.sin(th) ** 2) / (2 * sx) + (np.cos(th) ** 2) / (2 * sy)) * (y - y0) ** 2
        ))


    def _gaussian(M, *args):
        x, y = M
        arr = np.zeros(x.shape)
        for i in range((len(args) - 1) // 5):
            arr += gaussian(x, y, args[0], *args[1 + i * 5: 1 + i * 5 + 5])
        return arr
    if cog2[1] - cog1[1] !=0:
        initital_th = np.arctan((cog2[0] - cog1[0]) / (cog2[1] - cog1[1]))
    else:
        initital_th = np.pi/2
    initial_sig_x = ((img.shape[0] / 10) ** 2) / 10
    initial_sig_y = ((img.shape[0] / 10) ** 2) / 1

    # Initital g, sx, sy, x0, y0
    inital_guesses = [[initital_th],
                      (maxima[0]/30_000, initial_sig_x, initial_sig_y, cog1[1], cog1[0]),
                      (maxima[1]/30_000, initial_sig_x, initial_sig_y, cog2[1], cog2[0])]
    typ_vals = [np.pi/4, maxima[0]/30_000, initial_sig_x, initial_sig_y, img.shape[0]/2, img.shape[0]/2,
                maxima[1]/30_000, initial_sig_x, initial_sig_y, img.shape[0]/2, img.shape[0]/2]
    p0 = []
    for subset in inital_guesses:
        for elem in subset:
            p0.append(elem)

    if show_all and save_all is None:
        print("Guessed Theta: ", 180 * p0[0] / np.pi)


    # Prepare data
    xs = [x for x in range(img.shape[0])]
    ys = [y for y in range(img.shape[1])]

    X, Y = np.meshgrid(xs, ys)
    xdata = np.vstack((X.ravel(), Y.ravel()))
    Z = weighted_base

    # 3D Plot
    if show_all:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z, cmap='plasma')
        ax.set_zlim(0, np.max(Z) * 1.1)
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_3DInput.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    if show_all:
        # print("Initial guesses: ", p0)
        xsp0 = [x for x in range(img.shape[0])]
        ysp0 = [y for y in range(img.shape[1])]
        fit = np.zeros(Z.shape)

        Xp0, Yp0 = np.meshgrid(xsp0, ysp0)
        xdatap0 = np.vstack((Xp0.ravel(), Yp0.ravel()))
        Zp0 = weighted_base
        fitp0 = np.zeros(Zp0.shape)

        for i in range(2):
            fitp0 += gaussian(Xp0, Yp0, p0[0], *p0[1 + i * 5:1 + i * 5 + 5])

        rmsp0 = np.sqrt(np.mean((Zp0 - fitp0) ** 2))
        # print("Initial RMS: ", rmsp0)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(Xp0, Yp0, fitp0, cmap='plasma')
        cset = ax.contourf(Xp0, Yp0, Zp0 - fitp0, zdir='z', offset=-2 * np.amax(fitp0), cmap='plasma')
        ax.set_zlim(-2 * np.amax(fitp0), np.max(fitp0))
        plt.title("Initial Guess, Surface: Fit, Conout: Residual")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_InitialGuessVSresidual.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(Zp0, origin='lower', cmap='plasma',
                  extent=(min(xsp0), max(xsp0), min(ysp0), max(ysp0)))
        ax.contour(Xp0, Yp0, fitp0, colors='w')
        plt.title("Initial Guess, Counour: Fit, image: Data")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_InitialGuessVStarget.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    # Actual fit
    try:
        popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0)
    except RuntimeError:
        if verbose:
            print("Fit did not converge")
        return -1, -1
    if fit_fp is not None:
        relerr = []
        perr = np.sqrt(np.diag(pcov))
        for i in range(len(typ_vals)):
            relerr.append(perr[i]/typ_vals[i])
        with open(fit_fp, "w") as f:
            f.write(f"popt: {popt}\n")
            f.write(f"perr: {perr}\n")
            f.write(f"RelErr: {relerr}\n")
            f.write(f"|er|: {np.linalg.norm(np.array(relerr))}\n")
            f.write(f"pcov: {pcov}\n")

    # if show_all:
    #     print("Actual Theta: ", 180 * popt[0] / np.pi)
    fit = np.zeros(Z.shape)
    # if show_all:
    #     print("popt: ", popt)
    for i in range(2):
        fit += gaussian(X, Y, popt[0], *popt[1 + i * 5:1 + i * 5 + 5])
    #  print('Fitted parameters:')
    # print(popt)

    rms = np.sqrt(np.mean((Z - fit) ** 2))
    if show_all:
        # print('RMS residual =', rms)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, fit, cmap='plasma')
        cset = ax.contourf(X, Y, Z - fit, zdir='z', offset=-2 * np.amax(fit), cmap='plasma')
        ax.set_zlim(-2 * np.amax(fit), np.max(fit))
        plt.title("Surface: Fit, Conout: Residual")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_FitVSresidual.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(Z, origin='lower', cmap='plasma',
                  extent=(min(xs), max(xs), min(ys), max(ys)))
        ax.contour(X, Y, fit, colors='w')
        plt.title("Counour: Fit, image: Data")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_FitVSdata.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    theta, g1, sx1, sy1, x01, y01, g2, sx2, sy2, x02, y02 = popt

    g1 /= weights[0]
    g2 /= weights[1]

    fit = np.zeros(Z.shape)
    fit += gaussian(X, Y, theta, g1, sx1, sy1, x01, y01)
    fit += gaussian(X, Y, theta, g2, sx2, sy2, x02, y02)

    fit *= 255 / np.amax(fit)
    if show_all:
        plt.imshow(fit)
        plt.title("Rescaled fit")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_RescaledFit.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    red_image[:, :, 0] += fit

    red_image /= 510
    if show_all:
        plt.imshow(red_image)
        plt.title("Red image")
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_RedImage.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()


    # Draw Lines

    theta += np.pi/2

    ds = img.shape[0] / 3
    xmax = (ds / 2) * np.cos(theta)
    pt1 = (int(x01 - xmax), int(y01 - xmax * np.tan(theta)))
    pt2 = (int(x01 + xmax), int(y01 + xmax * np.tan(theta)))

    pt3 = (int(x02 - xmax), int(y02 - xmax * np.tan(theta)))
    pt4 = (int(x02 + xmax), int(y02 + xmax * np.tan(theta)))

    red_image *= 255
    img_sv = Image.fromarray(red_image.astype(np.uint8))
    id = ImageDraw.Draw(img_sv)
    id.line([pt1, pt2], width=1)
    id.line([pt3, pt4], width=1)

    # calculate Distance
    p1 = np.array([x01, y01])
    p2 = np.array([x02, y02])
    d = p2 - p1
    l = np.array([1, np.tan(theta)])
    s = np.array([np.tan(theta), -1])
    l /= np.linalg.norm(l)
    s /= np.linalg.norm(s)

    distance = abs(np.dot(d, s))

    # If direction is false:
    direct_distance = np.linalg.norm(p1 - p2)
    tmp = image.split('\\')[-1]
    # print(f"{tmp}: {direct_distance:.2f} vs. {distance:.2f}: {100*abs(direct_distance - distance)/distance:.1f}")
    if direct_distance/2 > distance:
        print("Direct Distance")
        od = distance
        distance = np.sqrt(np.linalg.norm(p1 - p2)**2 - distance**2)
        print(f"Change Dist {image}: {od} --> {distance} ")
    if show_all:
        plt.imshow(img_sv)
        plt.title("D: {:.2f}px, RMS: {:.3e}".format(distance, rms))
        if save_all is not None:
            plt.savefig(os.path.join(save_all, "{}_ImageSV.png".format(len(os.listdir(save_all)))))
        else:
            plt.show()
        plt.close()

    if transpose_res:
        img_sv = img_sv.transpose(Image.TRANSPOSE)
    # Save image with
    # id = ImageDraw.Draw(img_sv)
    # id.text((0, 0), "D: {:.2f}px\nRMS: {:.3e}".format(distance, rms), fill=(255, 255, 255))

    if saveas is not None:
        img_sv.save(saveas[:-3] + "png")



    # Find angles of both spots for
    for i in range(len(maskd_markers)):
        mm = maskd_markers[i]
        # plt.imshow(mm)
        # plt.title("MM0")
        # plt.show()
        if not show_all:
            mm -= bases[i] # K.A. Warum
        f = lambda x : max(0, x)
        f = np.vectorize(f)
        # plt.imshow(mm)
        # plt.title("MM1")
        # plt.show()
        mm = f(mm)
        # plt.imshow(mm)
        # plt.title("Masked Marker")
        # plt.show()

        # plt.imshow(mm)
        # plt.title("MM2")
        # plt.show()

        gf_res = gauss_fit(mm, accuracy=5000, show_all=show_all if save_all is None else save_all)
        if type(gf_res) is int:
            print("Cannot unpack Gauss Fit")
            return -1, -1
        mean, p, cov = gf_res
        sigmax = cov[0, 0]
        sigmaxy = cov[0, 1]
        sigmay = cov[1, 1]
        rhoxy = sigmaxy / np.sqrt(sigmay * sigmax)
        meanx = mean[0]
        meany = mean[1]

        m = func_from_covariance(cov, show_all if save_all is None else save_all)
        dx = mm.shape[0] / (6 * np.sqrt(1 + m ** 2))

        dx /=2

        p1 = (meany - m * dx, meanx - dx)
        p2 = (meany + m * dx, meanx + dx)
        # print(p1)
        h = 2 * np.amax(mm)
        mm[round(p1[1][0]), round(p1[0][0])] = h
        mm[round(p2[1][0]), round(p2[0][0])] = h


        dx = p1[1][0] - p2[1][0]
        dy = p1[0][0] - p2[0][0]

        if fit_fp is not None:
            with open(fit_fp, "a") as f:
                f.write(f"DX{i}: {dx}\n")
                f.write(f"DY{i}: {dy}\n")

        # idcs = np.argwhere(mm > 0)
        # xs = []
        # ys = []
        # for idx in idcs:
        #     for i in range(mm[idx[0], idx[1]]):
        #         xs.append(idx[0])
        #         ys.append(idx[1])
#
        # print("Xs len: ", len(xs))
        # xmed = np.average(xs)
        # ymed = np.average(ys)
        # b = sum([(xs[i] - xmed) * (ys[i] - ymed) for i in range(len(xs))]) / sum( [(xs[i] - xmed)**2 for i in range(len(xs))])
        # a = ymed - b * xmed
#
        # print(a, b)
#
        # for x in range(mm.shape[0]):
        #     y = round(a + b * x)
        #     if 0 <= y < mm.shape[1]:
        #         mm[y, x] = -5
#
        # plt.imshow(mm)
        # plt.show()
    return distance, theta % (2*np.pi)


def evaluate_image(img_fp, lbl_fp, save_fp, fit_fp, accuracy=4000, show_all=SHOW_ALL, verbose=True, transpose=False,
                   export_hists=None, pp_img_folder=None):
    return get_distance_2(img_fp, lbl_fp, saveas=save_fp, show_all=show_all, verbose=verbose, fit_fp=fit_fp,
                              transpose_res=transpose, export_hists=export_hists, pp_img_folder=pp_img_folder)

    # crp, cogs = find_crops(lbl_fp, show_all=show_all)
    # if len(crp) < 2:
    #     print("Len(crp) = ", len(crp))
    #     return -1
    # means = []
    # ps = []
    # crops = []
    # covariances = []
    # bases = []
    # for i in range(2):
    #     img, lbl, crop = coarse_crop_img_lbl(img_fp, lbl_fp, crp[i], show_all=show_all)  # Quatsch
    #     crpd_img, base = crop_img(img, lbl, show_all=show_all)  # Quatsch
    #     bases.append(base)
    #     # mean, p, covariance = gauss_fit(crpd_img, accuracy=accuracy)
    #     mean = [0, 0]
    #     p = np.zeros((100, 100))
    #     covariance = [[1, 0], [0, 1]]
    #     means.append(mean)
    #     ps.append(p)
    #     crops.append(crop)
    #     covariances.append(covariance)
#
    # return get_distance_2(img_fp, lbl_fp, saveas=save_fp, verbose=verbose, fit_fp=fit_fp)

class Distancer(Process):
    def __init__(self, task, dist_dict, theta_dict, log, verbose, transpose=False, export_hists=None, pp_img_folder=None):
        super().__init__()
        self.tasks = task
        self.transpose=transpose
        self.log = log
        self.verbose = verbose
        self.return_dict = dist_dict
        self.theta_dict = theta_dict
        self.export_hists = export_hists
        self.pp_img_folder=pp_img_folder


    def run(self) -> None:
        for task, idx in tqdm(self.tasks, disable=not self.log, desc="Evaluating images parallel"):

            try:
                np.seterr(all="ignore")
                dist, theta = evaluate_image(task[0], task[1], task[2], task[3], show_all=task[4], verbose=True,
                                             transpose=self.transpose, export_hists=self.export_hists,
                                             pp_img_folder=self.pp_img_folder)
                self.return_dict[idx] = dist
                self.theta_dict[idx] = theta
            except Exception as e:
                print(e)
                raise e
                pass



def evaluate_ss(image_folder, label_folder, save_folder, threads=1, accuracy=5000, show_all=SHOW_ALL,
                verbose=True, transpose=False, fit_params_folder=None, export_hists=None, pp_img_folder=None):
    SHOW_ALL = show_all
    if type(show_all) == str:
        show_all = str(show_all)
        os.makedirs(show_all, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    ims = [os.path.join(image_folder, x) for x in os.listdir(image_folder)]
    lbs = [os.path.join(label_folder, x) for x in os.listdir(label_folder)]
    svs = [os.path.join(save_folder, os.path.basename(x)) for x in os.listdir(image_folder)]
    if type(show_all) == str:
        sas = [os.path.join(show_all, x.split(".")[0]) for x in os.listdir(image_folder)]
        for sa in sas:
            os.makedirs(sa, exist_ok=True)
    else:
        sas = [show_all for x in os.listdir(image_folder)]
    if fit_params_folder is not None:
        fps = [os.path.join(fit_params_folder, x.split(".")[0] + ".txt") for x in os.listdir(image_folder)]
    else:
        fps = [None for _ in os.listdir(image_folder)]

    pairs = []
    for i in tqdm(range(len(os.listdir(image_folder))), disable=len(os.listdir(image_folder))<10000, desc="Collecting Eval image Tasks"):
        try:
            pairs.append((ims[i],
                      lbs[i],
                      svs[i],
                      fps[i], sas[i]))
        except IndexError as e:
            print("Index error for i=", i)
            return False
    # for pair in tqdm(pairs):
    #    evaluate_image(*pair, accuracy=5000)


    if threads <= 1:
        ret_dict = {}
        tht_dict = {}
        for i, pair in tqdm(enumerate(pairs), desc="Evaluating Images", total=len(pairs)):
            try:

                dist, theta = evaluate_image(pair[0], pair[1], pair[2], pair[3], show_all=pair[4], verbose=verbose, transpose=transpose,
                                             export_hists=export_hists, pp_img_folder=pp_img_folder)
                ret_dict[i] = dist
                tht_dict[i] = theta
            except Exception as e:
                print(e)
                raise e

    else:
        # ts = np.zeros(threads)
        tls = []
        for i in range(threads):
            tls.append([])
        for i, pair in enumerate(pairs):
            tls[i % threads].append((pair, i))

        manager = multiprocessing.Manager()
        ret_dict = manager.dict()
        tht_dict = manager.dict()
        thrds = []
        pbar = tqdm(total=len(pairs), desc="parallel Evaluations")
        for i in range(threads):
            thrds.append(Distancer(tls[i], ret_dict, tht_dict, log=i == 0, verbose=verbose, transpose=transpose,
                                   export_hists=export_hists, pp_img_folder=pp_img_folder))

        for t in thrds:
            t.start()

        old = 0
        updates = 0
        while True:
            time.sleep(1)
            new = len(ret_dict.keys())
            pbar.update(new - old)
            if old == new:
                updates += 1
            else:
                updates = 0
            if updates >= 20:
                break

            old = new

            if new == len(pairs):
                break

        for t in thrds:
            t.join()

    # print("Ret Dict")
    # for k in ret_dict.keys():
    #     print(f"RD: {k} -> {ret_dict[k]}")
    # for k in tht_dict.keys():
    #     print(f"TD: {k} -> {tht_dict[k]}")


    if len(ret_dict.keys()) == 0:
        return []
    keys = sorted(ret_dict.keys())
    # print(keys)
    # for key in keys:
    #     print(f"{key}: {ret_dict[key]}")
    # print("All equal")
    max_key = max(keys)
    #for i in range(max_key):
    #    assert i in keys
    ## print("All keys found")
    distances = []
    thetas = []
    try:
        distances = [ret_dict[k] for k in range(max_key+1)]
        thetas = [tht_dict[k] for k in range(max_key + 1)]

    except KeyError as e:
        print("Key error")
        for k in range(max_key + 1):
            d = ret_dict[k] if k in ret_dict.keys() else -1
            t = tht_dict[k] if k in tht_dict.keys() else -1
            distances.append(d)
            thetas.append(t)

    return distances, thetas




if __name__ == "__main__":
    # image_folder = "D:\\Dateien\\KI_Speicher\\DNA\\Complete_Eval\\TestNewGauss\\Test1\\image"
    # label_folder = "D:\\Dateien\\KI_Speicher\\DNA\\Complete_Eval\\TestNewGauss\\Test1\\label"
    # save_folder =  "D:\\Dateien\\KI_Speicher\\DNA\\Complete_Eval\\TestNewGauss\\Test1\\Result_Test"

    image_folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\CompleteEvalF\\Pret_Files"
    label_folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\CompleteEvalF\\SS_Labels"
    save_folder =  "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\OnlySS_Bin_5k\\CompleteEvalF\\TestEval"


    threads = 1
    #try:
    evaluate_ss(image_folder, label_folder, save_folder, threads=threads)
    # except Exception as e:
    #     pass
