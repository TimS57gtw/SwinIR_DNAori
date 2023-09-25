# from asyncio import threads
import time

from SPM_Filetype import SPM
import shutil
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image, ImageOps
from multiprocessing import Process
import random
import copy


def pretransform_image(fn, fn_after, img_size=None, show=False, is_mask=False, enhance_contrast=False, line_corr=False,
                       do_flatten=False, do_flatten_border=False, flip=False, mode='', flatten_line_90=True, rot90=False):
    #         task = (fn, fn_after, img_size, show, False, enhance_contrast, lines_corr, do_flatten, flip, flatten_border)

    def flatten_border(arr, thrsh=0.1, degree=np.infty, show=False, equalize="const"):
        arr_width = arr.shape[0]
        assert arr.shape[0] == arr.shape[1], "Non-square matrices not available yet"
        width = max(2, int(np.ceil(thrsh * arr_width)))

        top = arr[:, :width]
        bottom = arr[:, -width:]
        left = arr[:width, :]
        right = arr[-width:, :]

        top = np.average(top, axis=1)
        bottom = np.average(bottom, axis=1)
        left = np.average(left, axis=0)
        right = np.average(right, axis=0)

        # print(top, bottom, left, right)
        # print(top.shape, left.shape)

        hori_list_x = []
        hori_list_y = []

        verti_list_x = []
        verti_list_y = []

        for i in range(len(top)):
            hori_list_x.append(i)
            hori_list_y.append(top[i])
            hori_list_x.append(i)
            hori_list_y.append(bottom[i])

        for i in range(len(left)):
            verti_list_x.append(i)
            verti_list_y.append(left[i])
            verti_list_x.append(i)
            verti_list_y.append(right[i])

        if degree == 1:
            fx_lr = linregress(hori_list_x, hori_list_y)
            fy_lr = linregress(verti_list_x, verti_list_y)

            fx = lambda x: fx_lr.slope * x
            fy = lambda y: fy_lr.slope * y

            if show:
                xs = []
                for i in range(len(top)):
                    xs.append(i)
                    xs.append(i)

                ys = [fy(x) + fy_lr.intercept for x in xs]

                plt.plot(verti_list_x, verti_list_y)
                plt.plot(xs, ys)
                plt.show()

                ys = [fx(x) + fx_lr.intercept for x in xs]

                plt.plot(hori_list_x, hori_list_y)
                plt.plot(xs, ys)

                plt.show()

            center = np.average([fx_lr.intercept, fy_lr.intercept])

            mat = np.zeros(arr.shape)

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    mat[i, j] = center + fx(i) + fy(j)

            if show:
                plt.imshow(arr)
                plt.show()

                plt.imshow(mat)
                plt.show()

            arr = arr - mat
            if show:
                plt.imshow(arr)
                plt.show()

            return arr
        elif 1 < degree < 10:
            fx_lr = np.polyfit(hori_list_x, hori_list_y, deg=degree)[::-1]
            fy_lr = np.polyfit(verti_list_x, verti_list_y, deg=degree)[::-1]

            def fx(x):
                sum = 0
                for i in range(1, len(fx_lr)):
                    sum += fx_lr[i] * x ** i
                return sum

            def fy(x):
                sum = 0
                for i in range(1, len(fy_lr)):
                    sum += fy_lr[i] * x ** i
                return sum

            center = 0.5 * (fx_lr[0] + fy_lr[0])

            if show:
                xs = []
                for i in range(len(top)):
                    xs.append(i)
                    xs.append(i)

                ys = [fy(x) + fy_lr[0] for x in xs]

                plt.plot(verti_list_x, verti_list_y)
                plt.plot(xs, ys)
                plt.show()

                ys = [fx(x) + fx_lr[0] for x in xs]

                plt.plot(hori_list_x, hori_list_y)
                plt.plot(xs, ys)

                plt.show()

            mat = np.zeros(arr.shape)

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    mat[i, j] = center + fx(i) + fy(j)

            if show:
                plt.imshow(arr)
                plt.show()

                plt.imshow(mat)
                plt.show()

            arr = arr - mat
            if show:
                plt.imshow(arr)
                plt.show()

            return arr


        elif degree > 10:
            smoothing = int(arr_width / 10)

            center = np.average([top, bottom, left, right])

            fx_l = [(top[i] + bottom[i]) / 2 - center for i in range(arr_width)]
            fy_l = [(left[i] + right[i]) / 2 - center for i in range(arr_width)]
            if equalize == 'gauss':

                # plt.plot(kernel)
                # plt.show()

                def fx(x):

                    kernel = np.zeros(2 * smoothing + 1)
                    sigma = 2 * smoothing
                    for i in range(2 * smoothing + 1):
                        kernel[i] = np.exp(-(i - smoothing + 1) ** 2 / sigma)

                    left = max(0, x - smoothing)
                    if left == 0:
                        kernel_leftend = max(0, -x + smoothing)
                    else:
                        kernel_leftend = 0
                    right = min(arr_width, x + smoothing)
                    if right == arr_width:
                        kernel_rightend = min(2 * smoothing + 1, smoothing + 1 + arr_width - x)
                    else:
                        kernel_rightend = 2 * smoothing + 1

                    kernel[:kernel_leftend] = 0
                    kernel[kernel_rightend:] = 0

                    kernel = kernel / np.sum(kernel)

                    summe = 0
                    for kernelidx, elem in enumerate([x for x in range(left, right)]):
                        summe += fx_l[elem] * kernel[kernel_leftend + kernelidx]
                    return summe

                def fy(y):

                    kernel = np.zeros(2 * smoothing + 1)
                    sigma = 2 * smoothing
                    for i in range(2 * smoothing + 1):
                        kernel[i] = np.exp(-(i - smoothing + 1) ** 2 / sigma)

                    left = max(0, y - smoothing)
                    if left == 0:
                        kernel_leftend = max(0, -y + smoothing)
                    else:
                        kernel_leftend = 0
                    right = min(arr_width, y + smoothing)
                    if right == arr_width:
                        kernel_rightend = min(2 * smoothing + 1, smoothing + 1 + arr_width - y)
                    else:
                        kernel_rightend = 2 * smoothing + 1

                    kernel[:kernel_leftend] = 0
                    kernel[kernel_rightend:] = 0

                    if sum(kernel) == 0:
                        print("Error")

                    kernel = kernel / np.sum(kernel)

                    summe = 0
                    for kernelidx, elem in enumerate([x for x in range(left, right)]):
                        summe += fy_l[elem] * kernel[kernel_leftend + kernelidx]
                    return summe



            else:
                fx = lambda x: np.average(fx_l[max(0, x - smoothing):min(arr_width, x + smoothing)])
                fy = lambda y: np.average(fy_l[max(0, y - smoothing):min(arr_width, y + smoothing)])

            if show:
                xs = []
                for i in range(len(top)):
                    xs.append(i)
                    xs.append(i)

                ys = [center + fy(x) for x in xs]

                plt.plot(verti_list_x, verti_list_y)
                plt.plot(xs, ys)
                plt.show()

                ys = [center + fx(x) for x in xs]

                plt.plot(hori_list_x, hori_list_y)
                plt.plot(xs, ys)

                plt.show()

            mat = np.zeros(arr.shape)

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    mat[i, j] = center + fx(i) + fy(j)

            if show:
                plt.imshow(arr, cmap='gray')
                plt.title("Prev")
                plt.show()

                plt.imshow(mat)
                plt.show()

            nmat = arr - mat
            if show:
                plt.imshow(nmat, cmap='gray')
                plt.title("After")
                plt.show()

            return nmat

    def bilinear_interpol(mat, pos):

        if not (0 <= pos[0] <= np.shape(mat)[0] - 1 and 0 <= pos[1] <= np.shape(mat)[1] - 1):
            x = min(pos[0], np.shape(mat)[0] - 1)
            y = min(pos[1], np.shape(mat)[1] - 1)
        else:
            x = pos[0]
            y = pos[1]
        xu = int(np.floor(x))
        xfrac = x - xu
        xo = int(np.ceil(x))
        yo = int(np.floor(y))
        yu = int(np.ceil(y))
        yfrac = y - yo

        sum = xfrac * (1 - yfrac) * mat[xo, yo] + (1 - xfrac) * (1 - yfrac) * mat[xu, yo] + xfrac * (yfrac) * mat[
            xo, yu] + (1 - xfrac) * (yfrac) * mat[xu, yu]

        return sum

    def resize_numpy(arr, targetsize):

        newmat = np.zeros((targetsize, targetsize))
        scalfac = np.shape(arr)[0] / targetsize
        for y in range(targetsize):
            for x in range(targetsize):
                newmat[x, y] = bilinear_interpol(arr, np.array([x * scalfac, y * scalfac]))

        return newmat

    def line_90(arr, thrsh=0.9):

        for i in range(arr.shape[0]):
            # if random.random() < 0.1:
            #     plt.plot(arr[i, :])
            #     plt.title(sorted(arr[i, :])[int( (1-thrsh) * arr.shape[0])])
            #     plt.show()
            arr[i, :] -= sorted(arr[i, :])[int( (1-thrsh) * arr.shape[1])]

        return arr


    if os.path.isdir(fn):
        return
    if not is_mask:
        if fn.endswith("npy"):
            arr = np.load(fn, allow_pickle=True)
        elif fn.endswith("png") or fn.endswith("bmp"):
            arr = Image.open(fn)
            arr = np.array(ImageOps.grayscale(arr))
            arr = arr / 255
        elif fn.endswith("spm"):
            spm = SPM(fn)
            dat = spm.get_data()
            if img_size is not None:
                arr = resize_numpy(dat, img_size)
            else:
                arr = dat
        else:
            raise Exception("unknown filetype {}".format(fn))
    else:
        if fn.endswith("npy"):
            arr = np.load(fn, allow_pickle=True)
        elif fn.endswith("png"):
            arr = Image.open(fn)
            arr = np.array(ImageOps.grayscale(arr))
        else:
            raise Exception("unknown filetype {}".format(fn))

        arr = np.round(arr / 127)
        arr = np.array(arr, dtype=int)
    if flip:
        arr *= -1
    if show:
        plt.imshow(arr, cmap='gray')
        plt.title("Prev")
        plt.show()

    arr2 = copy.deepcopy(arr)

    if img_size is not None:
        arr = resize_numpy(arr, img_size)
    if flatten_line_90:
        arr = line_90(arr)
    if do_flatten_border:
        arr = flatten_border(arr)
    if do_flatten and not is_mask:
        arr = flatten_image(arr)
    if line_corr and not is_mask:
        arr = corr_lines(arr)
    # plt.imshow(arr)
    # plt.title("Vor Norm")
    # plt.show()
    if enhance_contrast and not is_mask:
        arr = normalize_soft(arr)
    # plt.imshow(arr)
    # plt.title("nach norm")
    # plt.show()
    if rot90:
        arr = np.rot90(arr)
    if show:
        plt.imshow(arr, cmap='gray')
        plt.title("After")
        plt.show()

   #arr2 = normalize_soft(arr2)
   #plt.imshow(arr2, cmap='gray')
   #plt.title("Ohne Line")
   #plt.show()

    if fn_after is not None:
        np.save(fn_after, arr, allow_pickle=True)
    return arr


def corr_lines(arr, axis=0):
    newarr = np.zeros(arr.shape)
    med = np.average(arr)
    for l in range(arr.shape[axis]):
        if axis == 0:
            avg = np.average(arr[l, :])
            dh = med - avg
            newarr[l, :] = arr[l, :] + dh
        else:
            avg = np.average(arr[:, l])
            dh = med - avg
            newarr[:, l] = arr[:, l] + dh

    # plt.imshow(arr)
    # plt.title("Prev, axis={}".format(axis))
    # plt.show()
    # plt.imshow(newarr)
    # plt.title("after, axis={}".format(axis))
    # plt.show()
    return newarr


class Pretransformer(Process):

    def __init__(self, tasklist, logger=False):
        super().__init__()
        self.tasklist = tasklist
        self.logger = logger

    def run(self) -> None:
        for task in tqdm(self.tasklist, desc="Working on tasks", unit="tasks", disable=not self.logger):
            pretransform_image(*task)


def pretransform_all(dir_img, dir_mask, dir_img_test, dir_mask_test, dir_img_pret, dir_mask_pret,
                     dir_img_test_pret, dir_mask_test_pret, show=False, enhance_contrast=False,
                     zfill=6, threads=16, img_size=None, overwrite=False, lines_corr=False, do_flatten=False,
                     flip=False, flatten_line_90=True,
                     do_flatten_border=False, use_masks=True, use_Test=True, keep_name=False, rot90=False):
    os.makedirs(dir_img_pret, exist_ok=True)
    if use_Test:
        os.makedirs(dir_img_test_pret, exist_ok=True)
    if use_masks:
        os.makedirs(dir_mask_pret, exist_ok=True)
        if use_Test:
            os.makedirs(dir_mask_test_pret, exist_ok=True)

    idx_from_img = lambda fn: int(fn[5:-4])
    idx_from_lbl = lambda fn: int(fn[5:-9]) if "mask" in fn else int(fn[8:-4])
    # idx_from_lbl = lambda fn : int(fn[8:-4])

    normal_fn = lambda idx: "Image{}.npy".format(str(idx).zfill(zfill))
    mask_fn = lambda idx: "Image{}_mask.npy".format(str(idx).zfill(zfill))

    tasks = []

    # Pretransform Train - Img
    imgs = sorted([x for x in os.listdir(dir_img) if not os.path.isdir(os.path.join(dir_img, x))])
    for i, elem in tqdm(enumerate(imgs), disable=True):
        idx = i + 1
        # print("FN: ", elem)
        # assert idx == idx_from_img(elem), "idx={}, lbl={}, orig: {}".format(idx, idx_from_img(elem), elem)
        if not keep_name:
            idx = idx_from_img(elem)
            fn = os.path.join(dir_img, elem)
            fn_after = os.path.join(dir_img_pret, normal_fn(idx))
        else:
            fn = os.path.join(dir_img, elem)
            fn_after = os.path.join(dir_img_pret, elem.split(".")[0] + ".npy")

        if not overwrite and os.path.isfile(fn_after):
            continue

        task = (fn, fn_after, img_size, show, False, enhance_contrast, lines_corr, do_flatten, do_flatten_border, flip, '', flatten_line_90, rot90)
        tasks.append(task)

    # Pretransform Train - Mask
    if use_masks:
        masks = sorted([x for x in os.listdir(dir_mask) if not os.path.isdir(os.path.join(dir_mask, x))])
        for i, elem in enumerate(masks):
            idx = i + 1
            assert idx == idx_from_lbl(elem), "idx={}, lbl={}, orig: {}".format(idx, idx_from_lbl(elem), elem)
            fn = os.path.join(dir_mask, elem)
            fn_after = os.path.join(dir_mask_pret, mask_fn(idx))
            if not overwrite and os.path.isfile(fn_after):
                continue
            task = (fn, fn_after, img_size, show, True, enhance_contrast, False, False, False, flip, "", False, rot90)
            tasks.append(task)

    # pretransform Test- Img
    if use_Test:
        imgs = sorted([x for x in os.listdir(dir_img_test) if not os.path.isdir(os.path.join(dir_img_test, x))])
        for i, elem in enumerate(imgs):
            idx = i + 1
            if not keep_name:
                assert idx == idx_from_img(elem), "idx={}, lbl={}, orig: {}".format(idx, idx_from_img(elem), elem)
                fn = os.path.join(dir_img_test, elem)
                fn_after = os.path.join(dir_img_test_pret, normal_fn(idx))
            else:
                fn = os.path.join(dir_img_test, elem)
                fn_after = os.path.join(dir_img_test_pret, elem.split(".")[0] + ".npy")
            if not overwrite and os.path.isfile(fn_after):
                continue
            task = (
            fn, fn_after, img_size, show, False, enhance_contrast, lines_corr, do_flatten, do_flatten_border, flip, '',
            flatten_line_90, rot90)
            tasks.append(task)

        # Pretransform test - Mask
        if use_masks:
            masks = sorted([x for x in os.listdir(dir_mask_test) if not os.path.isdir(os.path.join(dir_mask_test, x))])
            for i, elem in enumerate(masks):
                idx = i + 1
                assert idx == idx_from_lbl(elem), "idx={}, lbl={}, orig: {}".format(idx, idx_from_lbl(elem), elem)
                fn = os.path.join(dir_mask_test, elem)
                fn_after = os.path.join(dir_mask_test_pret, mask_fn(idx))
                if not overwrite and os.path.isfile(fn_after):
                    continue
                task = (fn, fn_after, img_size, show, True, enhance_contrast, False, False, False, flip, "", False, rot90)
                tasks.append(task)

    if len(tasks) == 0:
        print("Pretransforming not necessary")
        return
    task_sheet = np.zeros(threads, dtype=object)
    for i in range(len(task_sheet)):
        task_sheet[i] = []

    for i, task in enumerate(tasks):
        task_sheet[i % threads].append(task)

    threads_list = []
    for t in range(threads):
        threads_list.append(Pretransformer(task_sheet[t], t == 0))

    for thrd in threads_list:
        thrd.start()

    for thrd in threads_list:
        thrd.join()

    print("Finished pretransforming")
    return


def sort_data_folder(folder):
    os.makedirs(os.path.join(folder, "png"), exist_ok=True)
    os.makedirs(os.path.join(folder, "txt"), exist_ok=True)
    png_folder = os.path.join(folder, "png")
    txt_folder = os.path.join(folder, "txt")

    for elem in tqdm([x for x in os.listdir(folder) if not os.path.isdir(os.path.join(folder, x))]):
        z = os.path.join(folder, elem)
        if elem.endswith("png"):
            os.rename(z, os.path.join(png_folder, elem))
        elif elem.endswith("txt"):
            os.rename(z, os.path.join(txt_folder, elem))
        else:
            raise Exception(" Unknown filetype {}".format(elem))


def zfill_img_folder(folder, zfill=6):
    idx_from_img = lambda fn: int(fn[5:-4])
    normal_fn = lambda idx: "Image{}.".format(str(idx).zfill(zfill))

    for elem in tqdm([x for x in os.listdir(folder) if not os.path.isdir(os.path.join(folder, x))],
                     desc="Z-Filling Image"):
        z = os.path.join(folder, elem)
        idx = idx_from_img(elem)
        fn = normal_fn(idx) + elem[-3:]
        try:
            os.rename(z, os.path.join(folder, fn))
        except FileExistsError:
            pass


def zfill_labelfile(file, zfill=6):
    idx_from_img = lambda fn: int(fn[5:-4])
    normal_fn = lambda idx: "Image{}.".format(str(idx).zfill(zfill))
    txt = ""
    with open(file, "r") as f:
        lines = f.readlines()
        total = len(lines)
    with open(file, "r") as f:
        for line in tqdm(f, desc="Changing Label File", total=total):
            parts = line.strip().split(",")
            try:
                fn = parts[0]
                lbl = parts[1]
            except IndexError:
                break

            dir = os.path.dirname(fn)
            end = os.path.basename(fn)

            idx = idx_from_img(end)
            fn = normal_fn(idx) + end[-3:]

            txt += str(os.path.join(dir, fn)) + "," + lbl + "\n"
    with open(file, "w") as f:
        f.write(txt)


def zfill_lbl_folder(folder, zfill=6):
    idx_from_img = lambda fn: int(fn[8:-4])
    normal_fn = lambda idx: "Label_ss{}.".format(str(idx).zfill(zfill))

    for elem in tqdm([x for x in os.listdir(folder) if not os.path.isdir(os.path.join(folder, x))],
                     desc="Z-Filling Label"):
        if "mask" in elem:
            continue
        z = os.path.join(folder, elem)
        idx = idx_from_img(elem)
        fn = normal_fn(idx) + elem[-3:]
        os.rename(z, os.path.join(folder, fn))


def move_images(fr, to, typ=None):
    for f in tqdm(os.listdir(fr)):
        if not os.path.isdir(f) and (typ is None or f.endswith(typ)):
            os.rename(os.path.join(fr, f), os.path.join(to, f))


def flatten_image(arr, degree=2):
    average_xs = []
    average_ys = []
    xs = []
    ys = []

    for i in range(arr.shape[0]):
        average_xs.append(np.average(arr[i, :]))
        xs.append(i)

    for j in range(arr.shape[1]):
        average_ys.append(np.average(arr[:, j]))
        ys.append(j)

    if degree == 1:
        lr_x = linregress(xs, average_xs)
        lr_y = linregress(ys, average_ys)

        fx = lambda x: lr_x.intercept / 2 + x * lr_x.slope
        fy = lambda y: lr_y.intercept / 2 + y * lr_y.slope
    else:
        fx = np.poly1d(np.polyfit(xs, average_xs, degree))
        fy = np.poly1d(np.polyfit(ys, average_ys, degree))

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] -= fx(i) + fy(j)

    return arr


def flatten_tensor(tens):
    arr = np.squeeze(tens)
    arr = flatten_image(arr)
    return arr[np.newaxis, :, :, np.newaxis]


def compare_normings(img):
    plt.imshow(img, cmap="gray")
    plt.title("orig")
    plt.show()

    arr1 = gamma_correction(img)
    plt.imshow(arr1, cmap="gray")
    plt.title("gamma-corr")
    plt.show()

    arr2 = normalize_sigmoid(img)
    plt.imshow(arr2, cmap="gray")
    plt.title("sigmoid-corr")
    plt.show()

    arr3 = histogramm_equalization(img)
    plt.imshow(arr3, cmap="gray")
    plt.title("histogram-corr")
    plt.show()

    arr4 = normalize_half_sigmoid(img)
    plt.imshow(arr4, cmap="gray")
    plt.title("half-sigmoid-corr")
    plt.show()


def normalize_soft(image, show=False):
    # compare_normings(image)
    return normalize_half_sigmoid(image, show=show)


def linear_norm(image, show=False):
    if show:
        plt.imshow(image)
        plt.title("prev")
        plt.show()

        arr = image.flatten()

        plt.hist(arr, bins=int(len(arr) / 100))
        plt.title("Values before renorm")
        plt.show()

    maxi = np.amax(image)
    mini = np.amin(image)

    fkt = lambda x: (x - mini) / (maxi - mini)
    kf = np.vectorize(fkt)
    ret = kf(image)

    if show:
        plt.imshow(ret)
        plt.title("after")
        plt.show()

        arr = ret.flatten()

        plt.hist(arr, bins=int(len(arr) / 100))
        plt.title("Values after renorm")
        plt.show()
    return ret


def gamma_correction(image, show=False):
    if show:
        plt.imshow(image)
        plt.title("prev")
        plt.show()

        arr = image.flatten()

        plt.hist(arr, bins=int(len(arr) / 100))
        plt.title("Values before renorm")
        plt.show()

    gamma = 0.8

    normalized_image = linear_norm(image=image, show=False)
    enhanced_image = np.power(normalized_image, gamma)
    ret = enhanced_image

    if show:
        plt.imshow(ret)
        plt.title("after")
        plt.show()

        arr = ret.flatten()

        plt.hist(arr, bins=int(len(arr) / 100))
        plt.title("Values after renorm")
        plt.show()
    return ret


def histogramm_equalization(image, show=False):
    if show:
        plt.imshow(image)
        plt.title("prev")
        plt.show()
        arr = image.flatten()
        plt.hist(arr, bins=int(len(arr) / 100))
        plt.title("Values before renorm")
        plt.show()
    orig_shape = image.shape
    arr = image.flatten()
    pairs = []
    for i, x in enumerate(arr):
        pairs.append((i, x))

    pairs = sorted(pairs, key=lambda x: x[1])

    dx = 1 / len(pairs)

    renorm_sorted = np.zeros(len(pairs))
    for i, pair in enumerate(pairs):
        renorm_sorted[pair[0]] = dx * i

    ret = renorm_sorted.reshape(orig_shape)

    if show:
        plt.imshow(ret)
        plt.title("after")
        plt.show()

        arr = ret.flatten()

        plt.hist(arr, bins=int(len(arr) / 100))
        plt.title("Values after renorm")
        plt.show()
    return ret


def normalize_half_sigmoid(image, degree=1, show=False):
    # start = time.perf_counter()
    if show:
        plt.imshow(image, cmap='gray')
        plt.title("Before NormConst")
        plt.show()

    is_tensor = len(image.shape) == 4
    if is_tensor:
        image = image.squeeze()

    orig_shape = np.shape(image)

    arr = image.flatten()
    amnt = len(arr)
    sorted_arr = np.sort(arr)
    if show:
        plt.hist(sorted_arr, bins=int(len(arr) / 100))
        plt.title("Values before normconst")
        plt.show()


    pctl = 20 # Percentile of image height that is mapped to height lv
    pctr = 97 # Percentile of image height that is mapped to height rv
    #plt.hist(sorted_arr)

    xl = sorted_arr[int(pctl * amnt / 100)] # a
    xr = sorted_arr[int(pctr * amnt / 100)] # b
    # print(xl, xr)
    lv = 0.1 # Target Value of pixel with height xl
    rv = 0.6
    alpha = 1

    f1 = np.log(alpha/lv - 1)
    f2 = np.log(alpha/rv - 1)

    gamma = ((xr - xl) / f1) / (1- f2/f1)
    beta = gamma * f1 + xl

    f_norm = lambda x: alpha / (1 + np.exp(-(x-beta)/gamma))

    if show:
        xs = np.linspace(sorted_arr[0], sorted_arr[-1], 1000)
        ys = [f_norm(x) for x in xs]
        plt.plot(xs, ys)
        plt.title("Renorm")
        plt.show()
    f_norm_vec = np.vectorize(f_norm)

    if show:
        rnm_srted = f_norm_vec(sorted_arr)
        plt.hist(rnm_srted, bins=int(len(arr) / 100))
        plt.title("Values after normconst")
        plt.show()

    arr = np.reshape(arr, orig_shape)
    image = f_norm_vec(arr)

    image = linear_norm(image)

    if is_tensor:
        image = image[np.newaxis, :, :, np.newaxis]

    # print("Time: ", -start + time.perf_counter())
    if show:
        plt.imshow(image, cmap='gray')
        plt.title("After NormConst")
        plt.show()

    return image


def normalize_sigmoid(image, degree=1, show=False):
    # start = time.perf_counter()
    if show:
        plt.imshow(image, cmap='gray')
        plt.title("Before NormConst")
        plt.show()

    is_tensor = len(image.shape) == 4
    if is_tensor:
        image = image.squeeze()

    orig_shape = np.shape(image)

    arr = image.flatten()
    amnt = len(arr)
    sorted_arr = np.sort(arr)
    if show:
        plt.hist(sorted_arr, bins=int(len(arr) / 100))
        plt.title("Values before normconst")
        plt.show()

    p10 = sorted_arr[int(0.3 * amnt)]
    p90 = sorted_arr[int(0.7 * amnt)]

    medi = sorted_arr[int(0.5 * amnt)]

    sigma10 = -(1 / 2) * ((p10 - medi) / (np.log((1 / 0.3) - 1)))
    sigma90 = -(1 / 2) * ((p90 - medi) / (np.log((1 / 0.7) - 1)))
    sigma = (sigma10 + sigma90) / 2

    f_norm = lambda x: 1 / (1 + np.exp(-1 * (x - medi) / (2 * sigma)))

    if show:
        xs = np.linspace(sorted_arr[0], sorted_arr[-1], 1000)
        ys = [f_norm(x) for x in xs]
        plt.plot(xs, ys)
        plt.title("Renorm")
        plt.show()
    f_norm_vec = np.vectorize(f_norm)

    if show:
        rnm_srted = f_norm_vec(sorted_arr)
        plt.hist(rnm_srted, bins=int(len(arr) / 100))
        plt.title("Values after normconst")
        plt.show()

    arr = np.reshape(arr, orig_shape)
    image = f_norm_vec(arr)

    if is_tensor:
        image = image[np.newaxis, :, :, np.newaxis]

    # print("Time: ", -start + time.perf_counter())
    if show:
        plt.imshow(image, cmap='gray')
        plt.title("After NormConst")
        plt.show()

    return image


def prepare_spm(fp, flip=False, degree=1):
    def bilinear_interpol(mat, pos):
        if not (0 <= pos[0] <= np.shape(mat)[0] - 1 and 0 <= pos[1] <= np.shape(mat)[1] - 1):
            x = min(pos[0], np.shape(mat)[0] - 1)
            y = min(pos[1], np.shape(mat)[1] - 1)
        else:
            x = pos[0]
            y = pos[1]
        xu = int(np.floor(x))
        xfrac = x - xu
        xo = int(np.ceil(x))
        yo = int(np.floor(y))
        yu = int(np.ceil(y))
        yfrac = y - yo

        sum = xfrac * (1 - yfrac) * mat[xo, yo] + (1 - xfrac) * (1 - yfrac) * mat[xu, yo] + xfrac * (yfrac) * mat[
            xo, yu] + (1 - xfrac) * (yfrac) * mat[xu, yu]

        return sum

    def resize_numpy(arr, targetsize):

        newmat = np.zeros((targetsize, targetsize))
        scalfac = np.shape(arr)[0] / targetsize
        for y in range(targetsize):
            for x in range(targetsize):
                newmat[x, y] = bilinear_interpol(arr, np.array([x * scalfac, y * scalfac]))

        return newmat

    def spm2numpy(spm_fp):
        # print("---Resizing")
        spm = SPM(spm_fp)
        dat = spm.get_data()
        res = resize_numpy(dat)
        return res

    im = spm2numpy(fp)
    if flip:
        im *= -1

    im2 = copy.deepcopy(im)

    im = im[np.newaxis, :, :, np.newaxis]

    return im, im2


def normalization_constants_batch(np_folder, convert_to_np=True):
    fns = []
    if len(os.listdir(np_folder)) < 1000:
        fns = [os.path.join(np_folder, elem) for elem in os.listdir(np_folder) if
               (elem[-3:] == "spm" or elem[-3:] == "npy")]
    else:
        for _ in range(1000):
            fn = np.random.choice(os.listdir(np_folder))
            if fn[:-3] == "npy" or fn[:-3] == "spm":
                fns.append(os.path.join(np_folder, fn))

    pct_10 = []
    pct_90 = []
    avgs = []

    for fn in tqdm(fns):
        if convert_to_np:
            _, arr = prepare_spm(fn, True, False)
        else:
            arr = np.load(fn, allow_pickle=True)

        # plt.imshow(arr)
        # plt.title("Loaded")
        # plt.show()

        # arr = flatten_image(arr)
        arr = arr.flatten()
        amnt = len(arr)
        sorted_arr = np.sort(arr)

        # plt.hist(sorted_arr, bins=50)
        # plt.show()
        p10 = sorted_arr[int(0.1 * amnt)]
        p90 = sorted_arr[int(0.9 * amnt)]

        plt.title("{} , {}".format(p10, p90))
        pct_10.append(p10)
        pct_90.append(p90)
        avgs.append(sorted_arr[int(0.5 * amnt)])
        print("Median: ", sorted_arr[int(0.5 * amnt)], "Average: ", np.average(sorted_arr))

    plt.hist(pct_10)
    plt.show()
    p10 = np.average(pct_10)
    p90 = np.average(pct_90)
    avg = np.average(avgs)

    print("P10: ", p10)
    print("P90: ", p90)
    print("Average: ", avg)

    gamma = 2.4

    sigma10 = -(gamma / 2) * ((p10 - avg) / (np.log((1 / 0.1) - 1)))
    sigma90 = -(gamma / 2) * ((p90 - avg) / (np.log((1 / 0.9) - 1)))
    print("Sigma 10: ", sigma10)
    print("Sigma 90: ", sigma90)
    sigma = (sigma10 + sigma90) / 2

    f_norm = lambda x: 1 / (1 + np.exp(-gamma * (x - avg) / (2 * sigma)))
    f_norm_vec = np.vectorize(f_norm)
    if convert_to_np:
        _, arr = prepare_spm(os.path.join(np_folder, np.random.choice(
            [e for e in os.listdir(np_folder) if not os.path.isdir(os.path.join(np_folder, e))])), True, False)
    else:
        arr = np.load(os.path.join(np_folder, np.random.choice(
            [e for e in os.listdir(np_folder) if not os.path.isdir(os.path.join(np_folder, e))])), allow_pickle=True)

    # arr = flatten_image(arr)

    mini = np.amin(arr)
    maxi = np.amax(arr)

    xs = np.linspace(mini, maxi, 1000)
    ys = f_norm_vec(xs)
    plt.plot(xs, ys)
    plt.title("Conversion")
    plt.show()

    print(mini, maxi)

    plt.imshow(arr)
    plt.title("Prev")
    plt.show()

    arr = f_norm_vec(arr)

    plt.imshow(arr)
    plt.title("After")
    plt.show()


class Woerker(Process):

    def __init__(self, tasklist, first) -> None:
        # id, img, lbl, txt, spm, npy, folder, subfolder, zfil
        self.tl = tasklist
        self.first = first
        # self.id = id
        # self.img = img
        # self.lbl = lbl
        # self.txt = txt
        # self.spm = spm
        # self.npy = npy
        # self.folder = folder
        # self.subfolder = subfolder
        # self.zfil = zfil
        super().__init__()

    def run(self) -> None:
        for task in tqdm(self.tl, disable=not self.first):
            id, img, lbl, txt, spm, npy, folder, subfolder, zfil = task
            shutil.copyfile(img, os.path.join(folder, subfolder, "bild", "Image{}.png".format(str(id).zfill(zfil))))
            shutil.copyfile(lbl, os.path.join(folder, subfolder, "data", "PNG",
                                              "Image{}_mask.png".format(str(id).zfill(zfil))))
            shutil.copyfile(txt, os.path.join(folder, subfolder, "data", "TXT",
                                              "Image{}_lbl.png".format(str(id).zfill(zfil))))
            shutil.copyfile(spm, os.path.join(folder, subfolder, "sxm", "Image{}.spm".format(str(id).zfill(zfil))))
            shutil.copyfile(npy,
                            os.path.join(folder, subfolder, "sxm", "numpy", "Image{}.npy".format(str(id).zfill(zfil))))


def combine_sets(sets, new_folder, shuffle=True):
    subsets = ['Train', 'Test']
    for subset in subsets:
        total = 0
        for set in sets:
            tmx = [os.path.join(set, subset, "bild", x) for x in os.listdir(os.path.join(set, subset, "bild")) if
                   os.path.isfile(os.path.join(set, subset, "bild", x))]
            total += len(tmx)
            del tmx

        zfil = 6
        os.makedirs(os.path.join(new_folder, subset, "bild"), exist_ok=True)
        os.makedirs(os.path.join(new_folder, subset, "data", "PNG"), exist_ok=True)
        os.makedirs(os.path.join(new_folder, subset, "data", "TXT"), exist_ok=True)
        os.makedirs(os.path.join(new_folder, subset, "sxm", "numpy"), exist_ok=True)

        # fn_img = lambda i : os.path.join(new_folder, subset, "bild", "Image{}.png".format(str(i).zfill(zfil)))
        # fn_lbl = lambda i : os.path.join(new_folder, subset,  "data", "PNG","Image{}_mask.png".format(str(i).zfill(zfil)))
        # fn_txt = lambda i : os.path.join(new_folder, subset, "data", "TXT", "Image{}_lbl.png".format(str(i).zfill(zfil)))
        # fn_spm = lambda i : os.path.join(new_folder, subset, "sxm", "Image{}.spm".format(str(i).zfill(zfil)))
        # fn_npy = lambda i : os.path.join(new_folder, subset, "sxm", "numpy","Image{}.npy".format(str(i).zfill(zfil)))

        ids = [x for x in range(1, total + 1)]
        assert len(ids) == total, "Mistake in length"
        idx = 0
        thrds = []

        parallel = os.cpu_count()
        print(parallel)

        task_lists = np.zeros(parallel, dtype=object)
        for i in range(parallel):
            task_lists[i] = []

        if shuffle:
            np.random.shuffle(ids)

        for set in sets:
            imgs = [os.path.join(set, subset, "bild", x) for x in os.listdir(os.path.join(set, subset, "bild")) if
                    os.path.isfile(os.path.join(set, subset, "bild", x))]
            lbls = [os.path.join(set, subset, "data", "PNG", x) for x in
                    os.listdir(os.path.join(set, subset, "data", "PNG")) if
                    os.path.isfile(os.path.join(set, subset, "data", "PNG", x))]
            txts = [os.path.join(set, subset, "data", "TXT", x) for x in
                    os.listdir(os.path.join(set, subset, "data", "TXT")) if
                    os.path.isfile(os.path.join(set, subset, "data", "TXT", x))]
            spms = [os.path.join(set, subset, "sxm", x) for x in os.listdir(os.path.join(set, subset, "sxm")) if
                    os.path.isfile(os.path.join(set, subset, "sxm", x))]
            npys = [os.path.join(set, subset, "sxm", "numpy", x) for x in
                    os.listdir(os.path.join(set, subset, "sxm", "numpy")) if
                    os.path.isfile(os.path.join(set, subset, "sxm", "numpy", x))]

            maxi = min(len(imgs), len(lbls), len(txts), len(spms), len(npys))
            for i in range(maxi):
                task_lists[idx % parallel].append(
                    (ids[idx], imgs[i], lbls[i], txts[i], spms[i], npys[i], new_folder, subset, zfil)
                )
                idx += 1

        print(total, "jobs to do. ", len(task_lists))

        thrds = []
        for i in range(parallel):
            thrds.append(Woerker(task_lists[i], i == 0))
            thrds[-1].start()

        for t in thrds:
            t.join()

        # with tqdm(total=total) as pbar:

        #     for i in range(parallel):
        #         running.append(thrds[idx])
        #         running[-1].start()
        #         idx += 1

        #     while idx < tasks_todo:
        #         for i, t in enumerate(running):
        #             if not t.is_alive():
        #                 running[i] = thrds[idx]
        #                 idx += 1
        #                 running[i].start()
        #                 pbar.update(1)

        #     for i, t in enumerate(running):
        #             t.join()
        #             pbar.update(1)

        print("Combining sets completed")

def preprocess_yolo_test(start, end, args=None):
    show = False
    enhance_contrast = True
    global normalize_soft
    normalize_soft = normalize_half_sigmoid
    zfill = 6
    threads = 12
    img_size = None
    overwrite = False
    lines_corr = False
    do_flatten = False
    flip = False
    flatten_line_90 = True
    do_flatten_border = False
    dir_img = os.path.join(start, "images")
    dir_img_pret = os.path.join(end, "images")
    os.makedirs(dir_img_pret, exist_ok=True)

    if args is None:
        pretransform_all(dir_img=dir_img, dir_mask=None, dir_img_test=None, dir_mask_test=None, dir_img_pret=dir_img_pret,
                     dir_mask_pret=None, dir_img_test_pret=None, dir_mask_test_pret=None, show=show,
                     enhance_contrast=enhance_contrast,
                     zfill=zfill, threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                     do_flatten=do_flatten, flip=flip, flatten_line_90=flatten_line_90,
                     do_flatten_border=do_flatten_border, use_masks=False, use_Test=False, keep_name=True)
    else:
        pretransform_all(*args)

def preprocess_yolo_set(start, end):
    show = False
    enhance_contrast = True
    global normalize_soft
    # normalize_soft = normalize_half_sigmoid
    zfill=6
    threads = 14
    img_size=None
    overwrite=False
    lines_corr=True
    do_flatten=False
    flip = False
    flatten_line_90 = False
    do_flatten_border = False
    dir_img = os.path.join(start, "train", "images")
    dir_img_pret = os.path.join(end, "train", "images")
    dir_lbl = os.path.join(start, "train", "images")
    dir_lbl_pret = os.path.join(end, "train", "images")
    os.makedirs(dir_img_pret, exist_ok=True)
    os.makedirs(dir_lbl_pret, exist_ok=True)

    pretransform_all(dir_img=dir_img, dir_mask=None, dir_img_test=None, dir_mask_test=None, dir_img_pret=dir_img_pret,
                     dir_mask_pret=None, dir_img_test_pret=None, dir_mask_test_pret=None, show=show,
                     enhance_contrast=enhance_contrast,
                     zfill=zfill, threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                     do_flatten=do_flatten, flip=flip, flatten_line_90=flatten_line_90,
                     do_flatten_border=do_flatten_border, use_masks=False, use_Test=False, keep_name=False)
    try:
        shutil.copytree(dir_lbl, dir_lbl_pret)
    except FileExistsError:
        pass
    dir_img = os.path.join(start, "test", "images")
    dir_img_pret = os.path.join(end, "test", "images")
    dir_lbl = os.path.join(start, "test", "images")
    dir_lbl_pret = os.path.join(end, "test", "images")
    os.makedirs(dir_img_pret, exist_ok=True)
    os.makedirs(dir_lbl_pret, exist_ok=True)

    pretransform_all(dir_img=dir_img, dir_mask=None, dir_img_test=None, dir_mask_test=None, dir_img_pret=dir_img_pret,
                     dir_mask_pret=None, dir_img_test_pret=None, dir_mask_test_pret=None, show=show,
                     enhance_contrast=enhance_contrast,
                     zfill=zfill, threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                     do_flatten=do_flatten, flip=flip, flatten_line_90=flatten_line_90,
                     do_flatten_border=do_flatten_border, use_masks=False, use_Test=False, keep_name=False)
    try:
        shutil.copytree(dir_lbl, dir_lbl_pret)
    except FileExistsError:
        pass
    dir_img = os.path.join(start, "val", "images")
    dir_img_pret = os.path.join(end, "val", "images")
    dir_lbl = os.path.join(start, "val", "images")
    dir_lbl_pret = os.path.join(end, "val", "images")
    os.makedirs(dir_img_pret, exist_ok=True)
    os.makedirs(dir_lbl_pret, exist_ok=True)

    pretransform_all(dir_img=dir_img, dir_mask=None, dir_img_test=None, dir_mask_test=None, dir_img_pret=dir_img_pret,
                     dir_mask_pret=None, dir_img_test_pret=None, dir_mask_test_pret=None, show=show,
                     enhance_contrast=enhance_contrast,
                     zfill=zfill, threads=threads, img_size=img_size, overwrite=overwrite, lines_corr=lines_corr,
                     do_flatten=do_flatten, flip=flip, flatten_line_90=flatten_line_90,
                     do_flatten_border=do_flatten_border, use_masks=False, use_Test=False, keep_name=False)
    try:
        shutil.copytree(dir_lbl, dir_lbl_pret)
    except FileExistsError:
        pass


def npy_to_png(folder):
    files = []
    folders = []
    for fld in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, fld)):
            folders.append(os.path.join(folder, fld))
        if os.path.isfile(os.path.join(folder, fld)) and fld.endswith("npy"):
            files.append(os.path.join(folder, fld))

    while len(folders) > 0:
        fld = folders.pop()
        for x in os.listdir(fld):
            if os.path.isdir(os.path.join(fld, x)):
                folders.append(os.path.join(fld, x))
            if os.path.isfile(os.path.join(fld, x)) and x.endswith("npy"):
                files.append(os.path.join(fld, x))


    def transform_file(fn):
        arr = np.load(fn, allow_pickle=True)
        fn2 = fn.split(".")[0] + ".png"
        plt.imsave(fn2, arr, cmap='gray')

    for file in tqdm(files):
        transform_file(file)
        os.remove(file)

if __name__ == "__main__":
    # start = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\RealData"
    # end = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\RealData_pt"
    # preprocess_yolo_test(start, end)
    # npy_to_png(end)
    # exit(0)


    # start = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNA2_2k_256p"
    # end = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNA2_2k_256p_pt"
    # npy_to_png(end)
    #preprocess_yolo_set(start, end)

    start = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNAMix_4k_256p"
    end = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNAMix_4k_256p_pt2"
    preprocess_yolo_set(start, end)
    npy_to_png(end)