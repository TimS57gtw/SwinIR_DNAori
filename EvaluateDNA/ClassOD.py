import copy
import os
import random
import time
import imageio
import matplotlib
import matplotlib.pyplot as plt
import struct
import numpy as np
from SPM_Filetype import SPM
from Preprocessing import pretransform_image, flatten_image, corr_lines, linear_norm
from tqdm import tqdm
import scipy
from PIL import Image
from skimage.feature import match_descriptors, plot_matches, SIFT, hog
from skimage.transform import resize, probabilistic_hough_line
from skimage import data, exposure
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from matplotlib import cm
import cv2
from PIL import ImageDraw
from multiprocessing import Process



IGNORE_BORDER = True


def read_sxm(file, reultf):
    assert os.path.exists(file)
    f = open(file, 'rb')
    l = ''
    key = ''
    header = {}
    while l != b':SCANIT_END:':
        l = f.readline().rstrip()
        if l[:1] == b':':
            key = l.split(b':')[1].decode('ascii')
            header[key] = []
        else:
            if l:  # remove empty lines
                header[key].append(l.decode('ascii').split())
    while f.read(1) != b'\x1a':
        pass
    assert f.read(1) == b'\x04'
    assert header['SCANIT_TYPE'][0][0] in ['FLOAT', 'INT', 'UINT', 'DOUBLE']
    data_offset = f.tell()
    size = dict(pixels={
        'x': int(header['SCAN_PIXELS'][0][0]),
        'y': int(header['SCAN_PIXELS'][0][1])
    }, real={
        'x': float(header['SCAN_RANGE'][0][0]),
        'y': float(header['SCAN_RANGE'][0][1]),
        'unit': 'm'
    })
    im_size = size['pixels']['x'] * size['pixels']['y']
    data = np.array(struct.unpack('<>'['MSBFIRST' == header['SCANIT_TYPE'][0][1]] + str(im_size) +
                                  {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[header['SCANIT_TYPE'][0][0]],
                                  f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))

    return data, size['real']['x'] / size['pixels']['x']

def read_spm(file, resultf):
    chID = 0  # 0: Image, 1: DI/DV, 2: ?
    spm = SPM(file)
    dat = spm.get_data(chID=chID)
    reso = spm.get_size_per_pixel()
    dat = dat.astype(float)
    return dat, reso[0]
def read_file(fn, resultf):
    os.makedirs(resultf, exist_ok=True)
    ext = fn.split(".")[-1]
    if ext == "sxm":
        return read_sxm(fn, resultf)
    elif ext == "spm":
        return read_spm(fn, resultf)
    else:
        raise Exception("Unknown raw file {}".format(fn))

def preprocess_image(inarr, flatten_line=True):

    def normalize(image):
        orig_shape = np.shape(image)

        arr = image.flatten()
        amnt = len(arr)
        sorted_arr = np.sort(arr)
        # if False:
        #     plt.hist(sorted_arr, bins=int(len(arr) / 100))
        #     plt.title("Values before normconst")
        #     # plt.show(block=True)

        x80 = sorted_arr[int(0.95 * amnt)]  # 0.8
        x10 = sorted_arr[int(0.1 * amnt)]

        gamma = -(x10 - x80) / np.log(4)

        f_norm = lambda x: 2 / (1 + np.exp(-1 * (x - x80) / (gamma)))

        # if False:
        #     xs = np.linspace(sorted_arr[0], sorted_arr[-1], 1000)
        #     ys = [f_norm(x) for x in xs]
        #     plt.plot(xs, ys)
        #     plt.title("Renorm")
        #     # plt.show(block=True)
        f_norm_vec = np.vectorize(f_norm)

        # if False:
        #     rnm_srted = f_norm_vec(sorted_arr)
        #     plt.hist(rnm_srted, bins=int(len(arr) / 100))
        #     plt.title("Values after normconst")
        #     # plt.show(block=True)

        arr = np.reshape(arr, orig_shape)
        image = f_norm_vec(arr)

        image = linear_norm(image)


        # print("Time: ", -start + time.perf_counter())
        # if False:
        #     plt.imshow(image, cmap='gray')
        #     plt.title("After NormConst")
        #     # plt.show(block=True)

        return image

    # plt.imshow(inarr)
    # plt.title("Before")
    # # plt.show(block=True)

    arr = inarr

    # Flatten line:
    thrsh = 0.5
    for i in range(arr.shape[0]):
        arr[i, :] -= sorted(arr[i, :])[int((1 - thrsh) * arr.shape[1])]

    # plt.imshow(arr)
    # plt.title("After Flatten Line")
    # # plt.show(block=True)

    arr = flatten_image(arr, degree=1)

    # plt.imshow(arr)
    # plt.title("After Flatten Image")
    # # plt.show(block=True)

    arr = normalize(arr)

    # plt.imshow(arr)
    # plt.title("After Norm")
    # # plt.show(block=True)

    return arr


def hoshen_koppelmann(arr, minsize=None):
    """
    Implemenierung des Hoshen Koppelmann-Algorithmus
    """

    # Neues Array mit Padding links und oben
    padded_array = np.pad(arr, (1, 0))
    length_0 = arr.shape[0]
    length_1 = arr.shape[1]

    size = length_0 * length_1

    # Erstellung der Matrix fuer labels
    label = np.zeros(np.shape(padded_array), dtype=np.int32)

    # Zaehl-Array. None um Array-Indizes bei 1 zu beginnen
    n = [None]

    # Cluster-Index
    c = 1

    # Setze Label fuer linken und oberen Rand
    for i in range(length_1 + 1):
        label[0, i] = size
    for i in range(length_0+1):
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
    for i in range(1, length_0 + 1):
        for j in range(1, length_1 + 1):

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

    if minsize is not None:
        # plt.imshow(label)
        # plt.title("labels")
        # # plt.show(block=True)
        # print("cluster_sizes: ", cluster_sizes)
        drop = []
        for i in range(1, len(cluster_sizes)):
            if cluster_sizes[i] < minsize:
                drop.append(i)
        for elem in drop:
            labels[labels == elem] = -1
        # plt.imshow(label)
        # plt.title("After Drop")
        # # plt.show(block=True)

        # Umnummerieren
        vals = np.unique(label)
        # print("Unique: ", vals)
        for new, elem in enumerate(vals):
            if elem < 0:
                continue
            labels[labels == elem] = new

        # print("New unique: ", np.unique(labels))
        # plt.imshow(labels)
        # plt.title("Afer renum")
        # # plt.show(block=True)

    return labels

def watershed(array, resf, minsize=None):
    """
    Find threshold upon which only molecules are visible. Return binary array
    Minsize: Minimal size of clsuter to be considered
    """
    steps = 100
    # plt.imshow(array)
    # plt.title("in")
    # # plt.show(block=True)


    puddles = []
    islands = []
    heights = np.linspace(0, 1, steps+1)
    newarrfld = os.path.join(resf, "newarr")
    lowclst = os.path.join(resf, 'low')
    highclst = os.path.join(resf, 'high')
    os.makedirs(newarrfld, exist_ok=True)
    os.makedirs(lowclst, exist_ok=True)
    os.makedirs(highclst, exist_ok=True)
    for height in tqdm(heights, desc="Raising water level", leave=True, position=0, disable=True):
        newarr = np.zeros(array.shape, dtype=int)
        newarr[array > height] = 1
        if SAVE_HEIGHTS:
            plt.imshow(newarr)
            plt.title(f"NewArr {height}")
            plt.savefig(os.path.join(newarrfld, f'height_{str(int(100 * height)).zfill(4)}.png'))
            plt.clf()

        # print(len(np.argwhere(newarr == 0)))
        # if np.amin(newarr) != np.amax(newarr) and len(np.argwhere(newarr == 0)) >= 2:
        # print(f"{np.amin(newarr)} != {np.amax(newarr)}")
        lbls_heigh = hoshen_koppelmann(newarr, minsize=minsize)
        # plt.imshow(lbls_heigh)
        islands.append(np.amax(lbls_heigh))
        if SAVE_HEIGHTS:
            plt.imshow(lbls_heigh)
            plt.title(f"Labels heigh {height} -> {islands[-1]}")
            plt.savefig(os.path.join(highclst, f'height_{str(int(100 * height)).zfill(4)}.png'))
            plt.clf()

        # plt.imshow(lbls_heigh)
        # plt.title(f"Labels heigh {height} -> {islands[-1]}")
        # # plt.show(block=True)

        newarr = 1-newarr
        lbls_low = hoshen_koppelmann(newarr, minsize=minsize)
        puddles.append(np.amax(lbls_low))
        if SAVE_HEIGHTS:
            plt.imshow(lbls_low)
            plt.title(f"Labels low {height} -> {puddles[-1]}")
            plt.savefig(os.path.join(lowclst, f'height_{str(int(100*height)).zfill(4)}.png'))
            plt.clf()
        # else:
        #     if height < np.amin(array):
        #         puddles.append(0)
        #         islands.append(1)
        #     elif height > np.amax(array):
        #         puddles.append(1)
        #         islands.append(0)
        #     elif len(np.argwhere(newarr == 0)) < 2:
        #         puddles.append(1)
        #         islands.append(1)
        #     else:
        #         raise Exception

    if SAVE_HEIGHTS:
        plt.plot(heights, puddles, label="puddle")
        plt.plot(heights, islands, label="islands")
        plt.title("Number of clusters")
        plt.legend()
        plt.savefig(os.path.join(newarrfld, "plot.png"))
        plt.clf()

    if SAVE_DIA:
        frames = []
        for elem in os.listdir(highclst):
            image = imageio.v2.imread(os.path.join(highclst, elem))
            frames.append(image)

        imageio.mimsave(os.path.join(highclst, '0.gif'),  # output gif
                        frames,  # array of input frames
                        duration=5)  # optional: frames per s

        frames = []
        for elem in os.listdir(lowclst):
            image = imageio.v2.imread(os.path.join(lowclst, elem))
            frames.append(image)

        imageio.mimsave(os.path.join(lowclst, '0.gif'),  # output gif
                        frames,  # array of input frames
                        duration=5)  # optional: frames per s

        frames = []
        for elem in os.listdir(newarrfld):
            image = imageio.v2.imread(os.path.join(newarrfld, elem))
            frames.append(image)

        imageio.mimsave(os.path.join(newarrfld, '0.gif'),  # output gif
                        frames,  # array of input frames
                        duration=5)  # optional: frames per s

    # Find optimal height

    # Find lmax

    # plt.plot(islands)
    # plt.title("heights")
    # # plt.show(block=True)

    thresholds = np.linspace(0, np.log(np.amax(islands)), 100)
    areas = []
    ths = []

    def filter_heigh_areas(points):

        if len(points) == 0:
            return []
        if len(points) == 1:
            return [(points[0], points[0])]

        ranges = []
        inside = False
        start = None
        prev = None
        for i in range(len(points)):
            if not inside:
                start = points[i]
                inside = True
                prev = start
            else:
                if points[i] == prev + 1:
                    prev = points[i]
                    continue
                else:
                    ranges.append((start, points[i-1]))
                    start = points[i]
                    prev = start
        if inside:
            ranges.append((start, points[-1]))


        return ranges



    for th in thresholds:
        th = np.exp(th)
        ths.append(th)
        heightsloc = np.argwhere(islands > th)
        #vplt.plot(islands)
        #vplt.plot([x for x in range(len(islands))], [th for x in range(len(islands))])
        #v# plt.show(block=True)
        ranges = filter_heigh_areas(heightsloc)
        areas.append(len(ranges))


    awas2 = np.argwhere(np.array(areas) == 2)

    if len(awas2) == 0:
        # Too noisy for finding
        das = [0]
        for i in range(1, len(islands)):
            das.append((islands[i] - islands[i-1]) / (heights[i] - heights[i-1]))

        ideal_height = heights[np.argmin(das)]


    else:
        thmax = ths[awas2[-1][0]]

        heights_tar = np.argwhere(islands > thmax)
        heightsrange = filter_heigh_areas(heights_tar)
        lb = heightsrange[0][1]
        rb = heightsrange[1][0]

        # find ideal water height
        ideal_height = 0
        min_islands = np.infty
        for i in range(len(islands)):
            if i < lb:
                continue
            if i > rb:
                continue
            if islands[i] > min_islands:
                continue
            else:
                min_islands = islands[i]
                ideal_height = heights[i]

    watered_arr = np.zeros(array.shape, dtype=int)
    watered_arr[array > ideal_height] = 1



    # Filter small detections
    # plt.imshow(watered_arr)
    # plt.title("before")
    # # plt.show(block=True)
    clusters = hoshen_koppelmann(watered_arr, minsize=minsize)
    watered_arr[clusters == -1] = 0

    # plt.imshow(watered_arr)
    # plt.title("Watered Array")
    # # plt.show(block=True)

    return watered_arr, ideal_height


def locate_molecules(watered_arr, resultf, minsize_mol=50):
    """
    Locate Molecule positions within watered images.
    Returns list of bounding boxes as (xmin, ymin, xmax, ymax)
    """
    clusters = hoshen_koppelmann(watered_arr)
    if SHOW:
        plt.imshow(clusters)
        plt.title("HK")
        plt.savefig(os.path.join(resultf, "watered_clusters.png"))
        plt.clf()
    clusters_dicts = []
    num_clstrs = np.amax(clusters)
    for i in range(1, num_clstrs+1):
        name = i
        aw = np.argwhere(clusters == i)
        length = len(aw)
        d = {}
        d['name'] = name
        d['pos'] = aw
        d['len'] = length
        clusters_dicts.append(d)
        w_t = copy.deepcopy(watered_arr)
        w_t[clusters !=i ] = 0
        # plt.imshow(w_t)
        # plt.title(f"WT {i}, {length}")
        # # plt.show(block=True)

    # Filter sizes
    sizes = []
    for d in clusters_dicts:
        sizes.append(d['len'])

    # plt.hist(sizes)
    # plt.title("Size Distro")
    # # plt.show(block=True)

    mean = np.mean(np.array(sizes)[np.array(sizes) > minsize_mol])
    thrsh = mean /2

    remove_clusters = []
    filtered_dicts = []
    pairs = []
    for d in clusters_dicts:
        pairs.append((d['name'], d['len'], mean, abs(mean - d['len'])/thrsh))
        if abs(d['len']-mean)/thrsh > 1 or d['len'] < minsize_mol:
            remove_clusters.append(d['name'])
        else:
            filtered_dicts.append(d)

    # print(pairs)
    # print("Remove Clusters due to size: ", remove_clusters)
    for rcl in remove_clusters:
        clusters[clusters == rcl] = -1

    if SHOW:
        plt.imshow(clusters)
        plt.title("After Remove outliers")
        plt.savefig(os.path.join(resultf, "filtered_molecules.png"))
        plt.clf()


    # Find bbs
    bbs = []
    for elem in filtered_dicts:
        posis = elem['pos']
        posis = np.array(posis)

        bbs.append((np.amin(posis[: ,1]), np.amin(posis[: ,0]), np.amax(posis[: ,1]), np.amax(posis[:,0])))

    with open(os.path.join(resultf, "BoundingBoxes.csv"), 'w') as f:
        f.write("xmin;ymin;xmax;ymax;conf\n")
        for bb in bbs:
            f.write(f"{bb[0]};{bb[1]};{bb[2]};{bb[3]};1\n")

    return bbs

def draw_bbs(image_arr, bbs, resfile, boxfile):
    image_arr_T = np.array(image_arr).T
    new_image = Image.new('F', (image_arr_T.shape[0], image_arr_T.shape[1]))
    new_image_pixels = new_image.load()
    res_csv = boxfile
    # new_image_pixels = image_arr_T

    for i in range(image_arr_T.shape[0]):
        for j in range(image_arr_T.shape[1]):
            new_image_pixels[i, j] = 255*image_arr_T[i, j]


    new_image = new_image.convert('RGB')
    new_image_pixels = new_image.load()

    with open(res_csv, 'w') as f:
        resw = image_arr.shape[1]
        resh = image_arr.shape[0]
        for bb in bbs:
            f.write(f"0 {(bb[0] + bb[2])/(2*resw)} {(bb[1] + bb[3])/(2*resh)} {(bb[2] - bb[0])/(resw)} {(bb[3] - bb[1])/(resh)} 1\n")



    for bb in bbs:
        xmin = bb[0]
        ymin = bb[1]
        xmax = bb[2]
        ymax = bb[3]
        for x in range(xmin, xmax+1):
            new_image_pixels[x, ymin] = (255, 0, 0)
            new_image_pixels[x, ymax] = (255, 0, 0)
        for y in range(ymin, ymax+1):
            new_image_pixels[xmin, y] = (255, 0, 0)
            new_image_pixels[xmax, y] = (255, 0, 0)


    new_image.save(resfile)

def crop_moldecules(image_arr, bbs, resf, enlarge=1.2, ignore_border=IGNORE_BORDER):
    # plt.imshow(image_arr)
    # plt.title("Total")
    # # plt.show(block=True)
    resf_i = os.path.join(resf, 'images')
    resf_n = os.path.join(resf, 'numpy')
    os.makedirs(resf_i, exist_ok=True)
    os.makedirs(resf_n, exist_ok=True)

    for i, bb in enumerate(bbs):

        xmin = bb[0]
        ymin = bb[1]
        xmax = bb[2]
        ymax = bb[3]
        # print(xmin, xmax, ymin, ymax)

        # plt.imshow(image_arr)
        # plt.title("Total")
        # # plt.show(block=True)

        center = np.array([(xmin + xmax)/2, (ymin+ymax)/2])

        w = xmax - xmin
        h = ymax - ymin
        size = max(w, h)
        # print(size)
        size *= enlarge
        size = int(np.ceil(size))
        # print(size)

        lb = int(np.floor(center[0] -size/2))
        rb = lb + size + 1

        if lb < 0:
            if ignore_border:
                continue
            rb -= lb
            lb = 0

        if rb > image_arr.shape[0]:
            if ignore_border:
                continue
            diff = rb - image_arr.shape[0]+1
            rb -=diff
            lb -= diff


        ub = int(np.floor(center[1] - size/2))
        db = ub + size + 1

        if ub < 0:
            if ignore_border:
                continue
            db -= ub
            ub = 0

        if db > image_arr.shape[1]:
            if ignore_border:
                continue
            diff = db - image_arr.shape[1] + 1
            db -= diff
            ub -= diff

        # print(f'Cropping {lb}--{rb}, {ub}--{db}')
        sumimg = image_arr[ub:db, lb:rb]
        # print(f"Shape: ", sumimg.shape)
        if sumimg.shape[0] != sumimg.shape[1]:
            continue
        plt.imshow(sumimg)
        plt.title(f"crop {i}")
        plt.savefig(os.path.join(resf_i, f"Crop_{i}.png"))
        # # plt.show(block=True)
        plt.clf()

        np.save(os.path.join(resf_n, f'crop{i}.npy'), sumimg, allow_pickle=True)


def apply_HOG(input_fldr, resf):
    for crop in os.listdir(input_fldr):
        img1 = np.load(os.path.join(input_fldr, crop), allow_pickle=True)
        img1 -= np.amin(img1)
        img1 /= np.amax(img1)
        img1 = Image.fromarray(img1)

        train_img = r"D:\seifert\PycharmProjects\DNAmeasurement\datasets\SIFT_Train\Test2.png"
        img1 = Image.open(train_img)


        img1.show()

        fd, hog_image = hog(img1, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img1, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        # plt.show(block=True)
        plt.clf()


def apply_custom(input_fldr, resf, waterheight, imageid=None):



    distance_results = []

    for cropid, crop in enumerate(os.listdir(input_fldr)):
        try:
            img1 = np.load(os.path.join(input_fldr, crop), allow_pickle=True)
            img1 -= np.amin(img1)
            img = img1 / np.amax(img1)

            # print(f"{os.path.join(input_fldr, crop)} ->> {img1.shape} -> {img1.shape[0] == img1.shape[1]}")

            watered_arr = np.zeros(img1.shape, dtype=int)
            watered_arr[img1 > waterheight] = 1

            # Remove boundaries:
            neighbours = []
            for i in range(img1.shape[0]):
                if watered_arr[i, 0] == 1:
                    neighbours.append((i, 0))
                if watered_arr[i, img1.shape[1]-1] == 1:
                    neighbours.append((i, img1.shape[1]-1))

            for i in range(img1.shape[1]):
                if watered_arr[0, i] == 1:
                    neighbours.append((0, i))
                if watered_arr[img1.shape[0]-1, i] == 1:
                    neighbours.append((img1.shape[0]-1, i))

            if SAVE_DIA:
                os.makedirs(os.path.join(resf, f'Crop{cropid}', 'dias'), exist_ok=True)
                dia = 0
                diafld = os.path.join(resf, f'Crop{cropid}', 'dias')



            while len(neighbours) > 0:
                nb = neighbours.pop()
                watered_arr[nb[0], nb[1]] = 0
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if 0 <= i+nb[0] < watered_arr.shape[0]:
                            if 0 <= j+nb[1] < watered_arr.shape[1]:
                                if watered_arr[i+nb[0], j+nb[1]] == 1:
                                    neighbours.append((i+nb[0], j+nb[1]))

                if SAVE_DIA:
                    plt.imsave(os.path.join(diafld, f"dia_{str(dia).zfill(4)}.png"), watered_arr)
                    dia += 1



            center = np.ones_like(watered_arr)
            nbs = [(0, 0)]
            while len(nbs) > 0:
                # print(len(nbs))
                # print(nbs)
                nb = nbs.pop()
                # print(nbs)
                # input()

                center[nb[0], nb[1]] = 0
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if abs(i) + abs(j) != 1:
                            continue
                        if 0 <= i+nb[0] < watered_arr.shape[0]:
                            if 0 <= j+nb[1] < watered_arr.shape[1]:
                                if watered_arr[i+nb[0], j+nb[1]] == 0 and center[i+nb[0], j+nb[1]] == 1:
                                    nbs.append((i+nb[0], j+nb[1]))
                if SAVE_DIA:
                    plt.imsave(os.path.join(diafld, f"dia_{str(dia).zfill(4)}.png"), center)
                    dia += 1

            frames = []

            if SAVE_DIA:
                for elem in os.listdir(diafld):
                    image = imageio.v2.imread(os.path.join(diafld, elem))
                    frames.append(image)

                imageio.mimsave(os.path.join(diafld, '0.gif'),  # output gif
                                frames,  # array of input frames
                                duration=1)  # optional: frames per second


            # Find center
            pos = np.argwhere(center == 1)
            pos = np.array(pos)
            centerpos = np.mean(pos, axis=0)

            # base = np.average(img1[np.argwhere(center == 0)])
            # high = np.average(img1[np.argwhere(center == 1)])


            center = center.astype(np.float32)

            if SAVE_DIA:
                rotfld = os.path.join(resf, f'Crop{cropid}', 'rotation')
                os.makedirs(rotfld, exist_ok=True)


            thetas = np.linspace(0, 90, 91)


            x_scans = np.zeros( (len(thetas), center.shape[0]))
            y_scans = np.zeros( (len(thetas), center.shape[1]))

            # print(f"Center: ", center.shape)

            for i, theta in enumerate(tqdm(thetas, desc="Rotating", disable=True)):
                rot_mat = cv2.getRotationMatrix2D(centerpos, theta, 1.0)
                result = cv2.warpAffine(center, rot_mat, watered_arr.shape[1::-1], flags=ROTATION_INTERPOLATION)


                x_avg = np.average(result, axis=0)
                y_avg = np.average(result, axis=1)

                x_scans[i, :] = x_avg
                y_scans[i, :] = y_avg

                if SAVE_DIA:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 15))

                    axes[0].imshow(result)
                    axes[0].set_title(f"Rotated, Theta = {theta:.2f}°")

                    axes[1].plot(x_avg)
                    axes[1].set_title("X-Axis")
                    asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
                    axes[1].set_aspect(asp)
                    axes[2].plot(y_avg)
                    axes[2].set_title("Y-Axis")

                    asp = np.diff(axes[2].get_xlim())[0] / np.diff(axes[2].get_ylim())[0]
                    axes[2].set_aspect(asp)
                    plt.tight_layout()
                    plt.savefig(os.path.join(rotfld, f"Image{str(int(theta)).zfill(2)}.png"))
                    plt.clf()

            if SAVE_DIA:
                frames = []
                for elem in os.listdir(rotfld):
                    image = imageio.v2.imread(os.path.join(rotfld, elem))
                    frames.append(image)

                imageio.mimsave(os.path.join(rotfld, '0.gif'),  # output gif
                                frames,  # array of input frames
                                duration=10)  # optional: frames per second

            plt.cla()
            if SHOW:
                plt.imshow(x_scans.T)
                plt.title("X-Scans")
                # plt.show(block=True)
                plt.savefig(os.path.join(os.path.dirname(rotfld), "x_scan.png"))
                plt.cla()


                plt.imshow(y_scans.T)
                plt.title("y-Scans")
                # plt.show(block=True)
                plt.savefig(os.path.join(os.path.dirname(rotfld), "y_scan.png"))
                plt.cla()

            x_th = np.amax(x_scans) / 2
            y_th = np.amax(y_scans) / 2

            xws = []
            yws = []
            for th in range(x_scans.shape[0]):
                xl = x_scans[th, :]
                h = np.argwhere(xl > x_th)
                if len(h) == 0:
                    xws.append(0)
                else:
                    lb = np.amin(h)
                    rb = np.amax(h)
                    xws.append(rb - lb)

                yl = y_scans[th, :]
                h = np.argwhere(yl > y_th)
                if len(h) == 0:
                    yws.append(0)
                else:
                    lb = np.amin(h)
                    rb = np.amax(h)
                    yws.append(rb - lb)

            areas = np.array(xws) * np.array(yws)
            if SHOW:
                plt.plot(thetas, xws, label="X")
                plt.plot(thetas, yws, label="Y")
                plt.plot(thetas, areas, label="A")
                plt.legend()
                # plt.show(block=True)
                plt.savefig(os.path.join(os.path.dirname(rotfld), "areaPlot.png"))
                plt.cla()

            points = np.argwhere(areas == np.amax(areas))
            point = np.average(points)
            low = np.ceil(point) - point
            high = 1-low
            thetaOPT = low * thetas[int(np.floor(point))] + high * thetas[int(np.ceil(point))]

            rot_mat = cv2.getRotationMatrix2D(centerpos, thetaOPT, 1.0)
            resultOPT = cv2.warpAffine(center, rot_mat, center.shape[1::-1], flags=ROTATION_INTERPOLATION)

            # print("resultOPT.shape", resultOPT.shape)

            if SHOW:
                plt.imshow(resultOPT)
                plt.title(f"Opt Orientation: {thetaOPT}")
                # plt.show(block=True)
                plt.savefig(os.path.join(os.path.dirname(rotfld), "optimalRot.png"))
                plt.cla()

            binary_rot = np.zeros(result.shape)
            binary_rot[resultOPT > 0.5] = 1
            if SHOW:
                plt.imshow(binary_rot)
                plt.title("Binary Rot")
                # plt.show(block=True)

            pos = np.argwhere(binary_rot == 1)
            pos = np.array(pos)
            rotated_centerpos = np.mean(pos, axis=0)
            w = low * xws[int(np.floor(point))] + high * xws[int(np.ceil(point))]
            h = low * yws[int(np.floor(point))] + high * yws[int(np.ceil(point))]

            img_box = cv2.warpAffine(img1, rot_mat, img1.shape[1::-1], flags=ROTATION_INTERPOLATION)



            if h > w:
                centerposNew = np.array([centerpos[1], centerpos[0]])
                centerpos = centerposNew
                temp = h
                h = w
                w = temp
                resultOPT = resultOPT.T
                img1 = img1.T
                img_box = img_box.T

            img_box3 = np.zeros((img_box.shape[0], img_box.shape[1], 3))

            img_box3[:, :, 0] = 255 * img_box
            img_box3[:, :, 1] = 255 * img_box
            img_box3[:, :, 2] = 255 * img_box

            img_box3 = img_box3.astype(int)
            try:
                for k in range(3):
                    lb = int(np.floor(rotated_centerpos[0] - w / 2))
                    rb = int(np.ceil(rotated_centerpos[0] + w / 2)) + 1
                    lyb = int(np.floor(rotated_centerpos[1] - h / 2))
                    ryb = int(np.ceil(rotated_centerpos[1] + h / 2)) + 1
                    img_box3[ryb, lb:rb,k] = 1 if k == 0 else 0
                    img_box3[lyb,lb:rb,k] = 1 if k == 0 else 0
                    img_box3[lyb:ryb, lb, k] = 1 if k == 0 else 0
                    img_box3[lyb:ryb, rb, k] = 1 if k == 0 else 0
            except ValueError as e:
                print(e)
                print("Skip Crop ", cropid)
                continue
            except IndexError as e:
                print(e)
                print("Skip Crop ", cropid)
                continue

            if SHOW:
                plt.imshow(img_box3)
                # plt.show(block=True)
                plt.savefig(os.path.join(os.path.dirname(rotfld), 'imgBox3.png'))
                plt.cla()

            padding = (-1, 2)
            lstem =  lyb-padding[0] - (ryb+padding[0])
            restem=  lb - padding[1] - (rb + padding[1])
            if not ( lyb-padding[0] - (ryb+padding[0]) < -1 and lb - padding[1] - (rb + padding[1]) < -1):
                print("Zero Size molecule ", cropid)
                continue

            ulb = lyb-padding[0]
            olb = ryb+padding[0]
            urb = lb - padding[1]
            orb = rb + padding[1]

            if ulb < 0 or urb < 0:
                print("Smaller 0")
                continue

            molecule = img_box[ulb:olb, urb:orb]
            mls = molecule.shape
            y_scan = np.amax(molecule, axis=0)
            yss = y_scan.shape
            if len(y_scan) < 2:
                print("Y Scan too short")
                continue
            # print(yss, mls)
            if SHOW:
                fig, axes = plt.subplots(1, 2, figsize=(15, 15))

                axes[0].imshow(molecule)

                axes[1].plot(y_scan)
                axes[1].set_title("Y-scan")
                asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
                axes[1].set_aspect(asp)

                molfld = os.path.join(resf, 'molecule')
                os.makedirs(molfld, exist_ok=True)
                plt.savefig(os.path.join(molfld, f"Crop{str(cropid)}.png"))
                print("Saved as ", os.path.join(molfld, f"Crop{str(cropid)}.png"))
                # if SHOW:
                    # plt.show(block=True)
                plt.clf()


            sigma = 1
            kernf = lambda x : np.exp(-0.5 * (x/sigma)**2)
            kernel = np.array([kernf(x) for x in [-2, -1, 0, 1, 2]])
            kernel /= np.linalg.norm(kernel)

            y_scan_cnv = scipy.ndimage.convolve1d(y_scan, kernel, mode='constant', cval=np.amin(y_scan))

            measured_arr = y_scan_cnv if CONVOLVE else y_scan



            if SHOW:
                plt.plot(y_scan, label="y-scan")
                plt.plot(y_scan_cnv, label="y-scan-convolve")
                plt.legend()
                # plt.show(block=True)
                plt.savefig(os.path.join(molfld, "y_plain_convolve.png"))
                plt.cla()

            def fit_fkt(x, p1, s1, h1, p2, s2, h2, hb):
                g = h1 * np.exp(-0.5 * ((x - p1)/s1)**2) + h2 * np.exp(-0.5 * ((x - p2)/s2)**2)
                p1 = p1 * np.ones_like(x)
                p2 = p2 * np.ones_like(x)
                subf = lambda b : np.maximum(g, hb *np.logical_or(np.logical_and(np.less(p1, b), np.less(b, p1)), np.logical_and(np.less(p1, b), np.less(b, p2))) )
                if type(x) == list:
                    return [subf(k) for k in x]
                else:
                    return subf(x)

            def visualize_initial(f, xmax, params):
                xs = np.linspace(0, xmax, 100)
                x2 = []
                y = []
                b1 = []
                b2 = []
                b3 = []
                y2 = []
                p1, s1, h1, p2, s2, h2, hb = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
                for x in xs:
                    x2.append(x)
                    b1.append(h1 * np.exp(-0.5 * ((x - p1)/s1)**2))
                    b2.append(h2 * np.exp(-0.5 * ((x - p2)/s2)**2))
                    b3.append(hb if p1 < x < p2 or p1 > x > p2 else 0)
                    y.append(fit_fkt(x, p1, s1, h1, p2, s2, h2, hb))
                    y2.append(max(b1[-1] + b2[-1], b3[-1]))

                    plt.scatter(x2, y, label='y')
                    plt.scatter(x2, y2, label='y2')
                    plt.scatter(x2, b1, label='b1')
                    plt.scatter(x2, b2, label='b2')
                    plt.scatter(x2, b3, label='b3')
                    plt.legend()
                    # plt.show(block=True)





            # leftmax
            leftmax = 0
            while leftmax < len(measured_arr):
                if measured_arr[leftmax + 1] > measured_arr[leftmax]:
                    leftmax += 1
                else:
                    break

            # rightmax
            rightmax = len(measured_arr) - 1
            while rightmax >0:
                if measured_arr[rightmax - 1] > measured_arr[rightmax]:
                    rightmax -= 1
                else:
                    break

            # middle = (rightmax + leftmax)/2
            # newx = np.linspace(-100, 100, 801)
            # vals = np.interp(newx, [i - middle for i in range(len(measured_arr))], measured_arr)

            # if not os.path.isfile(f"max_compariosn_{imageid}.csv"):
            #     with open(f"max_compariosn_{imageid}.csv", 'w') as f:
            #         stri2 = []
            #         for i in range(len(newx)):
            #             stri2.append(f"{newx[i]};")
            #         stri2 = "".join(stri2)
            #         f.write(stri2 + "\n")
#
#
#
#
            # stri = []
            # for i in range(len(vals)):
            #     stri.append(f"{vals[i]};")
            # stri = "".join(stri)
            # with open(f"max_compariosn_{imageid}.csv", 'a') as f:
            #     f.write(f"{stri}\n")





            initial = (leftmax, 3, measured_arr[leftmax], rightmax, 3, measured_arr[rightmax], measured_arr[int((leftmax + rightmax)/2)])

            # visualize_initial(_fit_fkt, len(measured_arr) - 1, initial)

            if SHOW:

                fitfld = os.path.join(resf, 'fit')
                os.makedirs(fitfld, exist_ok=True)
                plt.plot(np.linspace(0, len(measured_arr) - 1, 100), [fit_fkt(i, *initial) for i in np.linspace(0, len(measured_arr) - 1, 100)], label="Initial")
                # plt.show(block=True)

                plt.plot(measured_arr, label="target")
                plt.plot(np.linspace(0, len(measured_arr) - 1, 100), [fit_fkt(i, *initial) for i in np.linspace(0, len(measured_arr) - 1, 100)], label="Initial")
                plt.legend()
                # plt.show(block=True)
                plt.savefig(os.path.join(fitfld, 'initialFit.png'))
                plt.cla()

            try:
                popt, pcov = scipy.optimize.curve_fit(fit_fkt, [i for i in range(len(measured_arr))], measured_arr, p0=initial)
            except RuntimeError:
                print("Fit did not converge")
                continue

            if SHOW:
                plt.plot(measured_arr, label="target")
                plt.plot(np.linspace(0, len(measured_arr) - 1, 100), [fit_fkt(i, *initial) for i in np.linspace(0, len(measured_arr) - 1, 100)],
                         label="Initial")
                plt.plot(np.linspace(0, len(measured_arr) - 1, 100), [fit_fkt(i, *popt) for i in np.linspace(0, len(measured_arr) - 1, 100)],
                         label="Result")
                plt.legend()

                plt.savefig(os.path.join(fitfld, f"Crop{str(cropid)}.png"))
                # if SHOW:
                    # plt.show(block=True)
                plt.clf()

            perr = np.linalg.norm(np.sqrt(np.diag(pcov)))
            if np.isinf(perr):
                perr = 1e6

            distance = popt[3] - popt[0]
            init_distance = initial[3] - initial[0]

            # print("Printing")
            # plt.imshow(molecule, cmap='gray')
            # plt.title("molecule")
            # plt.show()

            zoomfak = 20
            width = int(molecule.shape[1] * zoomfak)
            height = int(molecule.shape[0] * zoomfak)
            dim = (width, height)
            molecule_upsc = cv2.resize(molecule, dim, interpolation=cv2.INTER_NEAREST)

            # plt.imshow(molecule_upsc, cmap='gray')
            # plt.title("molecule_upsc")
            # plt.show()

            lpos = int(zoomfak * popt[0])

            rpos = int(zoomfak * popt[3])



            for i in range(molecule_upsc.shape[0]):
                if i % 20 < 3:
                    continue
                if 0 < lpos < molecule_upsc.shape[1]:
                    molecule_upsc[i, lpos] = 0
                if 0 < rpos < molecule_upsc.shape[1]:
                    molecule_upsc[i, rpos] = 0

            # plt.imshow(molecule_upsc, cmap='gray')
            # plt.title("molecule_upsc")
            # plt.show()
            if SAVE_PROFILE:

                fig, axs = plt.subplots(2, 1)
                fitarr = [fit_fkt(i, *popt) for i in np.linspace(0, len(measured_arr) - 1, 100)]
                y_scan_plt = copy.deepcopy(y_scan)
                y_scan_plt = np.array(y_scan_plt)
                oav = np.average(y_scan_plt)
                y_scan_plt += (np.average(fitarr) - oav)
                axs[0].imshow(molecule_upsc, cmap='gray', aspect="auto")
                axs[0].set_axis_off()
                axs[1].plot(y_scan_plt, label="scan")
                axs[1].set_xlim(0, len(measured_arr))
                axs[1].plot(np.linspace(0, len(measured_arr) - 1, 100),
                         fitarr,
                         label="Result")
                axs[1].legend()



                red_img_fld = os.path.join(resf, 'result_vis')
                os.makedirs(red_img_fld, exist_ok=True)
                plt.savefig(os.path.join(red_img_fld, f"Crop{str(cropid)}.png"))
                # print("Saved as ", red_img_fld, f"Crop{str(cropid)}.png")
                # plt.show()



            distance_results.append((cropid, distance, perr, init_distance))
        except Exception as e:
            print(f"Crop ignored due to error: ", e)
            raise e


    return distance_results


def apply_hough(input_fldr, resf):

    train_img = None
    for crop in os.listdir(input_fldr):
        img1 = np.load(os.path.join(input_fldr, crop), allow_pickle=True)
        img1 -= np.amin(img1)
        img1 /= np.amax(img1)

        # img1 = Image.fromarray(img1)

        # train_img = r"D:\seifert\PycharmProjects\DNAmeasurement\datasets\SIFT_Train\Test2.png

        # image = np.array(img1)[:, :, 0]
        image = img1

        if train_img is None:
            train_img = img1

        img1 = cv2.imread(r"D:\seifert\PycharmProjects\DNAmeasurement\datasets\SIFT_Train\Test1.png")

        # convert the input image into grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # modify the data type setting to 32-bit floating point
        img1 = np.float32(img1)
        img1 /= 255


        cv2.imshow('Image with Corners', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        corners = cv2.cornerHarris(img1, 2, 3, 0.05)

        # result is dilated for marking the corners
        corners = cv2.dilate(corners, None)

        # Threshold for an optimal value.
        cv2.imshow('Image with Corners', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.imshow(corners)
        # plt.show(block=True)
        img1[corners > 0.01 * corners.max()][0] = 0
        img1[corners > 0.01 * corners.max()][1] = 0
        img1[corners > 0.01 * corners.max()][2] = 255


        # the window showing output image with corners
        cv2.imshow('Image with Corners', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        edges = canny(image, 2, 1, 25)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                         line_gap=3)

        # Generating figure 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')

        ax[1].imshow(edges, cmap=cm.gray)
        ax[1].set_title('Canny edges')

        ax[2].imshow(edges * 0)
        for line in lines:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_title('Probabilistic Hough')

        for a in ax:
            a.set_axis_off()

        plt.tight_layout()
        # plt.show(block=True)

def apply_SIFT(input_fldr, train_image, resf):

    descriptor_extractor = SIFT()

    for crop in os.listdir(input_fldr):
        img1 = np.load(os.path.join(input_fldr, crop), allow_pickle=True)




        img_train = Image.open(train_image)
        img_train = np.array(img_train)[:, :, 0]
        img_train = img_train.astype(float)
        img_train /= 255

        mins = min(img_train.shape)
        img_train = img_train[:mins, :mins]

         # img1 = img_train.T

        img_train = resize(img_train, (100, 100))
        img1 = resize(img1, (100, 100))

        plt.imshow(img_train)
        plt.title("Train")
        # plt.show(block=True)

        plt.imshow(img1)
        plt.title("Im1")
        # plt.show(block=True)


        descriptor_extractor.detect_and_extract(img_train)
        keypoints1 = descriptor_extractor.keypoints
        descriptors1 = descriptor_extractor.descriptors

        descriptor_extractor.detect_and_extract(img1)
        keypoints2 = descriptor_extractor.keypoints
        descriptors2 = descriptor_extractor.descriptors

        matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,
                                      cross_check=True)



        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 8))

        plt.gray()

        plot_matches(ax[0], img_train, img1, keypoints1, keypoints2, matches12)
        ax[0].axis('off')
        ax[0].set_title("All Keypoints")


        plot_matches(ax[1], img_train, img1, keypoints1, keypoints2, matches12[::15],
                     only_matches=True)
        ax[1].axis('off')
        ax[1].set_title("Subset of Keypoints")

        plt.tight_layout()
        # plt.show(block=True)

def eval_results(result_list, img_name, stat_fld, result_csv, resolution=1):
    for elem in result_list:
        result_csv.write(f"{img_name};{elem[0]};{elem[1] * resolution};{elem[2]};{elem[3] * resolution}\n")

def eval_subset(arr, resf, name, reas=(30, 100)):
    os.makedirs(resf, exist_ok=True)

    # For Peak and Convolve
    # Hist All

    cnv_all = arr[:, 0]
    if len(cnv_all) > 0:
        plt.hist(cnv_all, bins=int(np.ceil(np.sqrt(len(cnv_all)))))
        plt.title("All Distances - CNV")
        plt.savefig(os.path.join(resf, "cnv_all.png"))
        plt.clf()

    pek_all = arr[:, 2]
    if len(pek_all) > 0:
        plt.hist(pek_all, bins=int(np.ceil(np.sqrt(len(pek_all)))))
        plt.title("All Distances - PEAK")
        plt.savefig(os.path.join(resf, "peak_all.png"))
        plt.clf()

        # Hist Reasonable

    cnv_all_reas = []
    for elem in cnv_all:
        if reas[0] < elem < reas[1]:
            cnv_all_reas.append(elem)

    if len(cnv_all_reas) > 0:
        plt.hist(cnv_all_reas, bins=int(np.ceil(np.sqrt(len(cnv_all_reas)))))
        plt.title("Reasonable Distances - CNV")
        plt.savefig(os.path.join(resf, "cnv_reas.png"))
    else:
        cnv_all_reas = [-1]

    pek_all_reas = []
    for elem in pek_all:
        if reas[0] < elem < reas[1]:
            pek_all_reas.append(elem)
    if len(pek_all_reas) > 0:
        plt.hist(pek_all_reas, bins=int(np.ceil(np.sqrt(len(pek_all_reas)))))
        plt.title("Reasonable Distances - PEAK")
        plt.savefig(os.path.join(resf, "peak_reas.png"))



    if arr.shape[0] < 5:
        print("not enough images for quality evaluation: ", name)
        return np.average(cnv_all), np.median(cnv_all), np.std(cnv_all)/np.sqrt(len(cnv_all)), len(cnv_all), np.average(cnv_all_reas), np.median(cnv_all_reas), np.std(cnv_all_reas)/np.sqrt(len(cnv_all_reas)), len(cnv_all_reas), 0, 0, 0, 0
        # Plot Std over Num

    arr = arr[arr[:, 1].argsort()]

    newarr = np.zeros_like(arr)
    pairs = []
    for i in range(arr.shape[0]):
        pairs.append((arr[i, 0], arr[i, 1], arr[i, 2]))
    pairs = sorted(pairs, key = lambda x : x[1])
    for i in range(len(pairs)):
        newarr[i, 0] = pairs[i][0]
        newarr[i, 1] = pairs[i][1]
        newarr[i, 2] = pairs[i][2]


    uncs = []
    nums = []
    uncsP = []
    for i in range(4, arr.shape[0]):
        nums.append(i)
        uncs.append(np.std(arr[:i, 0]) / np.sqrt(i))
        uncsP.append(np.std(arr[:i, 2]) / np.sqrt(i))

    plt.plot(nums, uncs, label="cnv")
    plt.plot(nums, uncsP, label='peak')
    plt.legend()
    plt.savefig(os.path.join(resf, "unc_num.png"))

    opt_idx = np.argmin(uncs)
    arr = arr[:opt_idx]
    with open(os.path.join(resf, 'quality.csv'), 'w') as f:
        f.write("distance;err;dist_peak\n")
        for i in range(arr.shape[0]):
            f.write(f"{arr[i, 0]};{arr[i, 1]};{arr[i,2]}\n")

        # csv Subset

        # Hist Subset
    if arr.shape[0] >=5:
        cnv_qual = arr[:, 0]
        plt.hist(cnv_qual, bins=int(np.ceil(np.sqrt(len(cnv_qual)))))
        plt.title("Quality Distances - CNV")
        plt.savefig(os.path.join(resf, "cnv_qual.png"))

        pek_qual = arr[:, 2]
        plt.hist(pek_qual, bins=int(np.ceil(np.sqrt(len(pek_qual)))))
        plt.title("Quality Distances - PEAK")
        plt.savefig(os.path.join(resf, "peak_qual.png"))
    else:
        cnv_qual = [-1]

    return np.average(cnv_all), np.median(cnv_all), np.std(cnv_all) / np.sqrt(len(cnv_all)), len(cnv_all), np.average(
        cnv_all_reas), np.median(cnv_all_reas), np.std(cnv_all_reas) / np.sqrt(len(cnv_all_reas)), len(
        cnv_all_reas), np.average(cnv_qual), np.median(cnv_qual), np.std(cnv_qual)/np.sqrt(len(cnv_qual)), len(cnv_qual)


def combine_results(res_csv, resultf):


    with open(os.path.join(resultf, 'final_results.csv'), 'w') as final_csv:

        final_csv.write("Image;d_avg;d_medi;d_unc;d_num;d_reas_mean;d_reas_medi;d_reas_unc;d_reas_num;d_qual_mean;d_qual_medi;d_qual_unc;d_qual_num\n")

        lines = 0
        with open(res_csv, 'r') as f:
            for line in f:
                if line.startswith('File'):
                    continue
                lines += 1

        result = np.zeros((lines, 3))
        name_dict = {}

        line_idx = 0
        with open(res_csv, 'r') as f:
            for line in f:
                if line.startswith('File'):
                    continue
                parts = line.split(";")
                name = parts[0]
                if name not in name_dict.keys():
                    name_dict[name] = []
                d =      float(parts[2])
                err =    float(parts[3])
                d_peak = float(parts[4])
                name_dict[name].append((d, err, d_peak))
                result[line_idx][0] = d
                result[line_idx][1] = err
                result[line_idx][2] = d_peak
                line_idx += 1

        totfld = os.path.join(resultf, '0_Total')
        os.makedirs(totfld, exist_ok=True)
        res = eval_subset(result, totfld, "Total")
        txt = f"Total;"
        for elem in res:
            txt += f"{elem};"
        txt += "\n"
        final_csv.write(txt)

        for name in name_dict.keys():
            os.makedirs(os.path.join(resultf, name), exist_ok=True)
            resultname = np.array(name_dict[name])
            res = eval_subset(resultname, os.path.join(resultf, name), name)
            txt = f"{name};"
            for elem in res:
                txt += f"{elem};"
            txt += "\n"
            final_csv.write(txt)



def class_analyze_folder(spm_folder, resultf, abort_od=False):
    spm_files = []
    paths = [spm_folder]
    while len(paths) > 0:
        pth = paths.pop()
        for elem in os.listdir(pth):
            if os.path.isdir(os.path.join(pth, elem)):
                paths.append(os.path.join(pth, elem))
            else:
                spm_files.append(os.path.join(pth, elem))

    # print(spm_files)
    stat_fld = os.path.join(resultf, 'statistics')
    os.makedirs(stat_fld, exist_ok=True)
    result_csv_fp = os.path.join(stat_fld, 'results.csv')
    result_csv = open(result_csv_fp, 'w')
    result_csv.write("File;Crop;Distance;Error;PeakDistance\n")
    start = time.perf_counter()


    with tqdm(total=len(spm_files) * 1122 + 1, desc=f"Evaluating {len(spm_files)} files") as pbar:

        if MULTIPROC:
            threads = 10
            tl = []
            for i in range(max(threads, len(spm_files))):
                tl.append([])
            for i in range(len(spm_files)):
                tl[i % threads].append(spm_files[i])

            thrds = []
            for i in range(len(tl)):
                thrds.append(Analyzer(tl[i], result_folder, stat_fld, result_csv))
                thrds[-1].start()
            for t in thrds:
                t.join()


        else:
            for i, file in enumerate(spm_files):
                try:

                    pbar.update(1)
                    pbar.set_description(f"Reading image {i+1}/{len(spm_files)}")

                    os.makedirs(os.path.join(resultf, 'preprocessed'), exist_ok=True)

                    arr, resolution = read_file(file, resultf)
                    # plt.imshow(arr)
                    # # plt.show(block=True)
                    if resolution < 1:
                        resolution *= 1000  # um to nm
                    # plt.title("Read File")
                    # # plt.show(block=True)

                    img_resf = os.path.join(resultf, str(os.path.basename(file)).split('.')[0])
                    os.makedirs(img_resf, exist_ok=True)

                    pbar.update(5)
                    pbar.set_description(f"Preprocess image {i + 1}/{len(spm_files)}")
                    arr = preprocess_image(arr)

                    plt.imsave(os.path.join(resultf, 'preprocessed', str(os.path.basename(file)).split('.')[0] + "_pp.png"), arr, cmap='gray')


                    pbar.update(5)
                    pbar.set_description(f"Watershed image {i + 1}/{len(spm_files)}")
                    watered_arr, height = watershed(arr, resf=img_resf, minsize=3)

                    pbar.update(1000)
                    pbar.set_description(f"Find Molecules image {i + 1}/{len(spm_files)}")
                    molecule_list = locate_molecules(watered_arr, resultf=img_resf)

                    os.makedirs(os.path.join(resultf, 'boxes'), exist_ok=True)


                    pbar.update(10)
                    pbar.set_description(f"Draw BBs image {i + 1}/{len(spm_files)}")
                    boxfile = os.path.join(resultf, 'boxes', 'csv')
                    os.makedirs(boxfile, exist_ok=True)

                    boxfile = os.path.join(boxfile, f"Image{str(len(os.listdir(boxfile))).zfill(4)}.txt")
                    draw_bbs(arr, molecule_list, os.path.join(resultf, 'boxes', str(os.path.basename(file)).split('.')[0] + "_boxes.png"), boxfile)

                    if abort_od:
                        continue
                    resf_crop = os.path.join(resultf, 'crops', str(os.path.basename(file)).split('.')[0])
                    os.makedirs(resf_crop, exist_ok=True)

                    pbar.update(1)
                    pbar.set_description(f"Crop + Analyze image {i + 1}/{len(spm_files)}")
                    crop_moldecules(arr, molecule_list, resf_crop, enlarge=ENLARGE)


                    result_list = apply_custom(os.path.join(resf_crop, 'numpy'), resf=os.path.join(resf_crop, "analysis"), waterheight=height, imageid=i)


                    pbar.update(100)
                    pbar.set_description(f"Evalaute Results image {i + 1}/{len(spm_files)}")
                    eval_results(result_list, str(os.path.basename(file)).split('.')[0], stat_fld, result_csv, resolution=resolution)
                except Exception as e:
                    print(f"Image execution of file {file} cancelled")
                    print(e)
                    raise e

        pbar.update(1)
        pbar.set_description(f"Combine results")
        result_csv.close()
        combined_fld = os.path.join(resultf, "results")
        os.makedirs(combined_fld, exist_ok=True)
        combine_results(os.path.join(stat_fld, 'results.csv'), combined_fld)



MULTIPROC = False
SHOW = False
SAVE_HEIGHTS = False
SAVE_DIA = False
CONVOLVE = True
SAVE_PROFILE = False
ENLARGE=1.3
ROTATION_INTERPOLATION = cv2.INTER_LINEAR
if __name__ == "__main__":
    IGNORE_BORDER = True

    # dss = [r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\NoBirka',
    #        r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\Set161_EvalYOLO_63\sxm',
    #        r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\AirNewCF']
    # for ds in dss:
    #
    ds = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\AirNewCF'
    res = os.path.join(r'D:\seifert\PycharmProjects\DNAmeasurement\TotalKomp', os.path.basename(ds), "Class")
    os.makedirs(res, exist_ok=True)
    class_analyze_folder(ds, resultf=res)

    assert 1 ==2


    # spm_folder = r"D:\seifert\PycharmProjects\DNAmeasurement\datasets\GOOD"
    spm_folder = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\INDIV'
    spm_folder = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\AirSingle'



    # spm_folder = r'C:\Users\seifert\PycharmProjects\STM_Simulation\bildordner\SingleDNA_V2\Set10\sxm'
    # spm_folder = r'D:\seifert\PycharmProjects\DNAmeasurement\datasets\Best46'
    length = len(os.listdir(r'D:\seifert\PycharmProjects\DNAmeasurement\ClassResults')) if os.path.isdir(r'D:\seifert\PycharmProjects\DNAmeasurement\ClassResults') else 0
    result_folder = os.path.join(r'D:\seifert\PycharmProjects\DNAmeasurement\ClassResults', f"Try_{length}_ClassOD_{os.path.basename(spm_folder)}")
    os.makedirs(result_folder,exist_ok=True)
    class_analyze_folder(spm_folder, resultf=result_folder)
