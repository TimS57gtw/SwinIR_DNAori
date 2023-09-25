import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import copy
import multiprocessing
from Evaluate_SS import hoshen_koppelmann, find_crops, extract_arr, coarse_crop_img_lbl
# from ErrorsNNFit import Net as ErrorNet
import torch
from matplotlib import cm
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
import matplotlib

MODE="STD"
def rectangularity_file(fn, resolution=2, sparse=0.5, theta_res=18, show=False):
    def find_border(lbl, show_all=False):
        brd_mat = np.zeros(lbl.shape)
        #try:
        max_j = lbl.shape[1]
        max_i = lbl.shape[0]
        #except IndexError as e:
        # plt.imshow(lbl)
        # plt.title("Error in RF")
        # plt.show()
            #raise e


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
            plt.show()
            plt.imshow(brd_mat)
            plt.title("Border Mat")
            plt.show()

        return brd_mat

    def cog(arr):
        center = np.array([0.0, 0.0])
        pts = np.argwhere(arr > 0.5)
        for pt in pts:
            center += np.array([pt[0], pt[1]])
        center /= float(len(pts))
        return center

    def find_area(arr, brd, cog, theta, resolution=4, sparse=1.0):

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        def inside(p, v1, v2, v3):
            d1 = sign(p, v1, v2)
            d2 = sign(p, v2, v3)
            d3 = sign(p, v3, v1)
            hn = d1 < 0 or d2 < 0 or d3 < 0
            hp = d1 > 0 or d2 > 0 or d3 > 0
            return not (hn and hp)

        def in_rect(p, v1, v2, v3, v4):
            return inside(p, v1, v2, v3) or inside(p, v2, v3, v4)

        def rect(brd_indcs, center, theta, w, h):
            if theta == 0:
                theta = 1e-3
            v1 = np.array([1.0, 1 / np.tan(theta)])
            v1 /= np.linalg.norm(v1)
            v2 = np.array([-1 / np.tan(theta), 1])
            v2 /= np.linalg.norm(v2)
            x1 = center + h * v1 + w * v2
            x2 = center - h * v1 + w * v2
            x3 = center + h * v1 - w * v2
            x4 = center - h * v1 - w * v2

            maxi = arr.shape[1]
            if not 0 <= x1[0] < maxi or not 0 <= x1[1] < maxi or not 0 <= x2[0] < maxi or not 0 <= x2[
                1] < maxi or not 0 <= x3[0] < maxi or not 0 <= x3[1] < maxi or not 0 <= x4[0] < maxi or not 0 <= x4[
                1] < maxi:
                return 0, None, None, None, None

            if arr[int(x1[0]), int(x1[1])] == 0:
                return 0, None, None, None, None
            if arr[int(x2[0]), int(x2[1])] == 0:
                return 0, None, None, None, None
            if arr[int(x3[0]), int(x3[1])] == 0:
                return 0, None, None, None, None
            if arr[int(x4[0]), int(x4[1])] == 0:
                return 0, None, None, None, None

            for elem in brd_indcs:
                p = np.array([elem[0], elem[1]])
                if in_rect(p, x1, x2, x3, x4):
                    return 0, None, None, None, None

            return 4 * w * h, x1, x2, x3, x4

        arr_zeros = np.argwhere(brd == 1)
        tples = [(x[0], x[1]) for x in arr_zeros]
        if sparse < 1:
            indxs = [x for x in range(len(arr_zeros))]
            indcs = np.random.permutation(indxs)[:int(len(arr_zeros) * sparse)]
            tples2 = []
            for indx in indcs:
                tples2.append(tples[indx])

            tples = tples2

        areas = []
        ecken = []
        idxs = []
        h_found = [0 for x in range(arr.shape[0])]
        w_found = [0 for x in range(arr.shape[0])]

        areas_loc = np.zeros(arr.shape)
        for w in range(int(arr.shape[0] / 2), 0, -resolution):
            for h in range(int(arr.shape[0] / 2), 0, -resolution):
                a = rect(tples, cog, theta, w, h)[0]
                if a > 0:
                    areas_loc[w, h] = a
                    break
        for h in range(int(arr.shape[0] / 2), 0, -resolution):
            for w in range(int(arr.shape[0] / 2), 0, -resolution):
                a = rect(tples, cog, theta, w, h)[0]
                if a > 0:
                    areas_loc[w, h] = a
                    break

        w, h = np.unravel_index(np.argmax(areas_loc, axis=None), areas_loc.shape)
        area, x1, x2, x3, x4 = rect(arr_zeros, cog, theta, w, h)
        # plt.imshow(areas_loc)
        # plt.show()

        return area, (x1, x2, x3, x4)

    img = Image.open(fn)
    arr = np.array(img, dtype=np.float64)
    if type(arr[0, 0]) is np.ndarray:
        # plt.imshow(arr)
        # plt.title("Rem 3D")
        # plt.show()
        arr = arr[..., 0]
    # else:
    #     plt.imshow(arr)
    #     plt.title("NOT 3D")
    #     plt.show()

    arr /= 255.0
    arr = np.ceil(arr)
    brd = find_border(arr)
    if show:
        plt.imshow(brd)
        plt.title("Border")
        plt.show()

    arr_mrkd = copy.deepcopy(arr)
    if show:
        plt.imshow(arr)
        plt.title(fn)
        plt.show()

    cent = cog(arr)

    # find_d(arr, cent, np.pi/4)
    try:
        cent_idy = int(round(cent[0]))
        cent_idx = int(round(cent[1]))
    except ValueError:
        # NAN
        return 0
    arr_mrkd[cent_idy, cent_idx] = 2

    if show:
        plt.imshow(arr_mrkd)
        plt.title(cent)
        plt.show()

    thetas = np.linspace(0, np.pi / 2, theta_res)
    all_aas = []
    all_corners = []
    for theta in tqdm(thetas, desc="Finding optimum rectangle", disable=True):
        a, ecken = find_area(arr, brd, cent, theta, resolution=resolution, sparse=sparse)
        all_aas.append(a)
        all_corners.append(ecken)

    if show:
        plt.plot(thetas, all_aas)
        plt.title("Area over angle")
        plt.show()

    maxidx = np.argmax(all_aas)
    max_theta = thetas[maxidx]
    max_area = all_aas[maxidx]
    corners = all_corners[maxidx]

    cov_real = len(np.argwhere(arr > 0.5))
    rectangular = max_area / cov_real

    if show:
        pt_x = []
        pt_y = []
        for corner in corners:
            pt_x.append(corner[1])
            pt_y.append(corner[0])
        plt.scatter(pt_x, pt_y)
        plt.imshow(arr_mrkd)
        plt.title("With Corners: {}".format(max_area))
        plt.show()
    return rectangular


class Rectangularity(multiprocessing.Process):
    def __init__(self, task, ret_dict, log, resolution=2, sparse=0.5, theta_res=18, show=False):
        super().__init__()
        self.task = task
        self.ret_dict = ret_dict
        self.disable_tqdm = not log
        self.resolution = resolution
        self.sparse = sparse
        self.theta_res = theta_res
        self.show = show

    def run(self) -> None:
        for fn in tqdm(self.task, desc="Rectangular Parallel", disable=self.disable_tqdm):
            idx = fn.split("\\")[-1].split(".")[0]
            self.ret_dict[idx] = rectangularity_file(fn, self.resolution, self.sparse, self.theta_res, self.show)


def rectangulartiy(fld, threads=4, resolution=2, sparse=0.5, theta_res=18, show=False):
    manager = multiprocessing.Manager()
    quality_dict = manager.dict()

    tasks = []
    files = [os.path.join(fld, x) for x in os.listdir(fld)]
    fns = [x for x in os.listdir(fld)]
    for i in range(len(files)):
        tasks.append(files[i])

    thrds = [[] for i in range(threads)]
    for i, task in enumerate(tasks):
        thrds[i % threads].append(task)

    prcs = []
    for i in range(threads):
        prcs.append(Rectangularity(thrds[i], ret_dict=quality_dict, log=i == 0, resolution=resolution, sparse=sparse,
                                   theta_res=theta_res, show=show))

    for p in prcs:
        p.start()
    for p in prcs:
        p.join()

    # keys = quality_dict.keys()
    # keys = sorted(keys)
    #
    # for key in keys:
    #     print(f"{key}: {quality_dict[key]}")
    return quality_dict


def gauss_accuracy(fit_folder):
    quality_dict = {}
    marker_alignment_dict = {}
    files = [os.path.join(fit_folder, x) for x in os.listdir(fit_folder)]

    for file in tqdm(files, desc="Gauss Accuracy"):
        dx0 = None
        dx1 = None
        dy0 = None
        dy1 = None

        with open(file, "r") as f:
            for line in f:
                parts = line.split(":")
                if parts[0].strip() == "|er|":
                    quality_dict[file.split("\\")[-1].split(".")[0]] = float(parts[1])
                elif parts[0].strip() == "DX0":
                    dx0 = float(parts[1])
                elif parts[0].strip() == "DX1":
                    dx1 = float(parts[1])
                elif parts[0].strip() == "DY0":
                    dy0 = float(parts[1])
                elif parts[0].strip() == "DY1":
                    dy1 = float(parts[1])
                else:
                    #print("Unknown Part", parts[0])
                    pass

        if dx0 is None or dx1 is None or dy0 is None or dy1 is None:
            print("Nones for file ", file)
            marker_alignment_dict[file.split("\\")[-1].split(".")[0]] = -1
        else:
            v1 = np.array([dx0, dy0])
            v2 = np.array([dx1, dy1])
            theta = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if theta > np.pi/2:
                theta = np.pi - theta
            marker_alignment_dict[file.split("\\")[-1].split(".")[0]] = theta

    # keys = quality_dict.keys()
    # keys = sorted(keys)
    #
    # for key in keys:
    #     print(f"GA-{key}: {quality_dict[key]}")
    return quality_dict, marker_alignment_dict


class CandC(multiprocessing.Process):
    def __init__(self, files, size_dict, count_dict, log):
        super().__init__()
        self.files = files
        self.size_dict = size_dict
        self.count_dict = count_dict
        self.disable_tqdm = not log

    def run(self) -> None:
        for file in tqdm(self.files, desc="Count&Compare", disable=self.disable_tqdm):
            idx = file.split("\\")[-1].split(".")[0]
            arr = extract_arr(file).astype(float)
            arr_orig = copy.deepcopy(arr)
            arr *= 1 / 250
            arr = arr.astype(np.uint8)
            # plt.imshow(arr)
            # plt.show()

            lbls = hoshen_koppelmann(arr)
            # plt.imshow(lbls)
            # plt.show()
            self.count_dict[idx] = np.max(lbls)

            fc_ret = find_crops(arr_orig)

            if fc_ret is None:
                continue
            crops, cogs, label_mod = fc_ret  # Mdeified label without single point markers
            sizes = []
            for crp in crops:
                _, lbl_loc, _ = coarse_crop_img_lbl(arr, label_mod, crp, show_all=False)
                sizes.append(len(np.argwhere(lbl_loc > 0)))

            if len(sizes) != 2:
                self.size_dict[idx] = 1
                # print("len of Sizes:", len(sizes))
            else:
                self.size_dict[idx] = np.abs(sizes[0] - sizes[1]) / (sizes[0] / 2 + sizes[1] / 2)


def count_compare_labels(labels, threads=4):
    files = [os.path.join(labels, x) for x in os.listdir(labels)]

    manager = multiprocessing.Manager()
    count_dict = manager.dict()
    size_dict = manager.dict()

    tasks = []

    for i in range(len(files)):
        tasks.append(files[i])

    thrds = [[] for i in range(threads)]
    for i, task in enumerate(tasks):
        thrds[i % threads].append(task)

    prcs = []
    for i in range(threads):
        prcs.append(CandC(thrds[i], size_dict, count_dict, i == 0))

    for p in prcs:
        p.start()
    for p in prcs:
        p.join()

    return size_dict, count_dict


def error_est_rect(rectivity):
    return (0.4401048 - 0.40725 * rectivity) / 0.1855  # Aus Fit von 725 testdaten 0.1855 is avg error


def error_est_gauss(gauss_std):
    return 0.01 * (12.91378 + 5.67717 * gauss_std) / 0.1855


def error_est_counts(markers):
    if markers == 1:
        return 1e8
    return 0.01 * (2.40951 + 5.0377 * markers) / 0.1855


def error_est_size(sizediff):
    return 0.01 * (7.62307 + 0.18131 * 100 * sizediff) / 0.1855


def error_est_conf(confidence):
    return (0.5656 - 0.47732 * confidence) / 0.1855

def error_est_angle(theta):
    if theta > np.pi/2:
        theta = np.pi - theta
    if theta < 0:
        return 10
    return (0.14762 + 0.14352 * theta) / 0.1855

def est_error_fkt(folder):
    raise NotImplementedError
    model = ErrorNet()
    model.load_state_dict(torch.load(os.path.join(folder, "Model.pth")))
    model.eval()
    rf = lambda x: x
    gf = lambda x: x
    sf = lambda x: x
    af = lambda x: x
    cf = lambda x: x

    with open(os.path.join(folder, "constants.csv"), "r") as f:
        for line in f:
            parts = line.split(",")
            if parts[0] == "R":
                rf = lambda x: (x - float(parts[1])) / float(parts[2])
            elif parts[0] == "G":
                gf = lambda x: (x - float(parts[1])) / float(parts[2])
            elif parts[0] == "S":
                sf = lambda x: (x - float(parts[1])) / float(parts[2])
            elif parts[0] == "A":
                af = lambda x: (x - float(parts[1])) / float(parts[2])
            elif parts[0] == "C":
                cf = lambda x: (x - float(parts[1])) / float(parts[2])
            else:
                raise AttributeError("unknown Line start " + parts[0])

    def fkt(rect, gauss, size, count, conf):
        if count == 1:
            return 1
        r = rf(rect)
        g = gf(gauss)
        s = sf(size)
        a = af(count)
        c = cf(conf)
        with torch.autocast(device_type="cpu"):
            inpt = torch.from_numpy(np.array([r, g, s, a, c]))
            pred = model(inpt.float())
        pred = pred.item()
        # print("Pred: ", pred)
        return pred

    return fkt

def est_error_avg(r, g, s, m, c, t):
    return error_est_rect(r) * error_est_gauss(g) * error_est_size(s) * error_est_counts(m) * error_est_conf(
        c) * error_est_angle(t)


def est_error(r, g, s, m, c, t):
    if MODE == "STD":
        return est_std(r, g, s, m, c, t)
    else:
        return est_error_avg(r, g, s, m, c, t)


def visualize_errors(resfolder):
    os.makedirs(resfolder, exist_ok=True)
    cmap = cm.terrain



    Rect = np.arange(0, 1, 0.05)  # Rect
    Gauss = np.arange(0, 0.5, 0.025)
    Size = np.arange(0, 2, 0.1)
    Count = np.arange(1, 7, 0.3)
    Conf = np.arange(0.5, 1, 0.025)
    Angl = np.arange(0, np.pi/2, np.pi/40)#

    r_avg = 0.61
    g_avg = 0.04
    s_avg = 0.6
    a_avg = 2.5
    c_avg = 0.81
    t_avg = 0.341

    cn1 = "Rect"
    cn2 = "Gauss"
    cn3 = "Size"
    cn4 = "Count"
    cn5 = "Conf"
    cn6 = "Angl"

    interp = get_interpolator(file="D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try227_StdPlot\\results.csv",
                              q_indcs="D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try223_DS_StdQ_mitAngle\\quality\\quality_indcs.csv")

    fit = get_fit(file="D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try227_StdPlot\\results.csv",
                              q_indcs="D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try223_DS_StdQ_mitAngle\\quality\\quality_indcs.csv")

    err_fks = [ (np.vectorize(est_std), "STD"),
                (np.vectorize(est_error_avg), "AVG"),
                (interp, "Interpol"),
                (fit, "Fit")]

    conds = [(Rect, cn1), (Gauss, cn2), (Size, cn3), (Count, cn4), (Conf, cn5), (Angl, cn6) ]

    def plot_relation(fkt, fkt_name, cond1, cond1name, cond2, cond2name):

        if cond1name == cond2name:
            p1 = cond1 if cond1name == cn1 else r_avg
            p2 = cond1 if cond1name == cn2 else g_avg
            p3 = cond1 if cond1name == cn3 else s_avg
            p4 = cond1 if cond1name == cn4 else a_avg
            p5 = cond1 if cond1name == cn5 else c_avg
            p6 = cond1 if cond1name == cn6 else t_avg

            ret = fkt(p1, p2, p3, p4, p5, p6)
            # rint(cond1name)
            # rint(cond2name)
            # rint(p1, p2, p3, p4, p5, p6)
            # rint(ret)
            # rint(ret.shape)
            # rint(cond1)
            # rint(cond1.shape)
            plt.scatter(cond1, ret)
            plt.xlabel(cond1name)
            plt.ylabel(fkt_name)
            plt.title(f"{cond1name}-{fkt_name}")
            plt.savefig(os.path.join(resfolder, f"{cond1name}_{fkt_name}.png"))
            plt.cla()
            plt.clf()
            return

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(cond1, cond2)
        if cond1name == cn1:
            p1 = X
        elif cond2name == cn1:
            p1 = Y
        else:
            p1 = r_avg

        if cond1name == cn2:
            p2 = X
        elif cond2name == cn2:
            p2 = Y
        else:
            p2 = g_avg

        if cond1name == cn3:
            p3 = X
        elif cond2name == cn3:
            p3 = Y
        else:
            p3 = s_avg

        if cond1name == cn4:
            p4 = X
        elif cond2name == cn4:
            p4 = Y
        else:
            p4 = a_avg

        if cond1name == cn5:
            p5 = X
        elif cond2name == cn5:
            p5 = Y
        else:
            p5 = c_avg

        if cond1name == cn6:
            p6 = X
        elif cond2name == cn6:
            p6 = Y
        else:
            p6 = t_avg

        Z_class = fkt(p1, p2, p3, p4, p5, p6)

        surf = ax.plot_surface(X, Y, Z_class, cmap=cmap)
        plt.title(f"{cond1name}-{cond2name}-{fkt_name}")
        ax.set_xlabel(cond1name)
        ax.set_ylabel(cond2name)
        ax.set_zlabel(f"{fkt_name} err")
        # plt.show()
        plt.savefig(os.path.join(resfolder, f"{cond1name}_{cond2name}_{fkt_name}.png"))
        plt.cla()
        plt.clf()

    tasks = []

    for i in range(len(err_fks)):
        for j in range(len(conds)):
            for k in range(j, len(conds)):
                tasks.append((err_fks[i][0], err_fks[i][1], conds[j][0], conds[j][1], conds[k][0], conds[k][1]))

    for task in tqdm(tasks, desc="Visualizing Error Fks"):
        plot_relation(*task)




def find_std_funcs(q_indcs_file, resfolder):
    os.makedirs(resfolder, exist_ok=True)
    first = True
    fns = []
    dists = []
    thetas = []
    abws = []
    est_errs = []
    rects = []
    gausss = []
    sizes = []
    counts = []
    confs = []
    angles = []
    err_rects = []
    err_gausss = []
    err_sizes = []
    err_counts = []
    err_confs = []

    width = 100
    with open(q_indcs_file, "r") as f:
        for line in tqdm(f, desc="Reading Quality indcs"):
            if first:
                first = False
                continue
            parts = line.strip().split(",")
            fns.append(parts[0])
            dists.append(float(parts[1]))
            thetas.append(float(parts[2]))
            abws.append(float(parts[3]))
            ee = parts[4]
            if ee == "nan" or float(ee) > 1e8:
                ee = np.infty
            est_errs.append(float(ee))
            rects.append(float(parts[5]))
            gausss.append(float(parts[6]))
            sizes.append(float(parts[7]))
            counts.append(float(parts[8]))
            confs.append(float(parts[9]))
            angles.append(float(parts[10]))
            err_rects.append(float(parts[11]))
            err_gausss.append(float(parts[12]))
            err_sizes.append(float(parts[13]))
            err_counts.append(float(parts[14]))
            err_confs.append(float(parts[15]))


    pairs =[]
    for i in range(len(fns)):
        pairs.append( (dists[i], abws[i], rects[i], gausss[i], sizes[i], counts[i], confs[i], angles[i]) )

    # Rect
    pairs_rect = copy.deepcopy(pairs)
    pairs_rect = sorted(pairs_rect, key=lambda x: x[2])
    rect_x = []
    rect_std = []
    for i in range(width, len(pairs_rect) - width):
        locds = [x[0] for x in pairs_rect[i - width:i + width]]
        rect_x.append(pairs_rect[i][2])
        rect_std.append(np.std(locds))
    plt.plot(rect_x, rect_std)
    plt.title("Rect-Std")
    plt.xlabel("rectangularity")
    plt.ylabel("Std of dists")
    plt.savefig(os.path.join(resfolder, "RectStd.png"))
    plt.cla()
    plt.clf()
    with open(os.path.join(resfolder, "RectStd.csv"), "w") as f:
        f.write("Rect;Std\n")
        for i in range(len(rect_x)):
            f.write(f"{rect_x[i]};{rect_std[i]}\n")

    # Gauss
    pairs_gauss = copy.deepcopy(pairs)
    pairs_gauss = sorted(pairs_gauss, key=lambda x : x[3])



    gauss_x = []
    gauss_std = []
    pg2 = []

    for i in range(len(pairs_gauss)):
        pg2.append( (pairs_gauss[i][0], pairs_gauss[i][3]) )

    pg2 = sorted(pg2, key= lambda x : x[1])

    xs = [x[1] for x in pg2]
    # plt.plot(xs)
    # plt.title("xs")
    # plt.show()

    pg3 = []
    for elem in pg2:
        if str(elem[1]).lower() == "nan" or elem[1] > 3:
            continue
        pg3.append(elem)

    pg3 = sorted(pg3, key=lambda x : x[1])
    xs2 = [x[1] for x in pg3]
    # plt.plot(xs2)
    # plt.title("xs2")
    # plt.show()

    for i in range(width, len(pg3) - width):
        locds = [x[0] for x in pg3[i-width:i+width]]
        gauss_x.append(pg3[i][1])
        gauss_std.append(np.std(locds))

    # plt.plot(gauss_x)
    # plt.show()

    plt.scatter(gauss_x, gauss_std)
    plt.title("Gauss-Std")
    plt.xlabel("gauss std")
    plt.ylabel("Std of dists")
    plt.savefig(os.path.join(resfolder, "GaussStd.png"))
    plt.cla()
    plt.clf()
    with open(os.path.join(resfolder, "GaussStd.csv"), "w") as f:
        f.write("Gauss;Std\n")
        for i in range(len(gauss_x)):
            f.write(f"{gauss_x[i]};{gauss_std[i]}\n")

    # Size
    size_rect = copy.deepcopy(pairs)
    size_rect = sorted(size_rect, key=lambda x: x[4])
    size_x = []
    size_std = []
    for i in range(width, len(size_rect) - width):
        locds = [x[0] for x in size_rect[i - width:i + width]]
        size_x.append(size_rect[i][4])
        size_std.append(np.std(locds))
    plt.plot(size_x, size_std)
    plt.title("Size-Std")
    plt.xlabel("sizediff")
    plt.ylabel("Std of dists")
    plt.savefig(os.path.join(resfolder, "SizeStd.png"))
    plt.cla()
    plt.clf()
    with open(os.path.join(resfolder, "SizeStd.csv"), "w") as f:
        f.write("Size;Std\n")
        for i in range(len(size_x)):
            f.write(f"{size_x[i]};{size_std[i]}\n")

    # Count
    count_rect = copy.deepcopy(pairs)
    count_rect = sorted(count_rect, key=lambda x: x[5])
    min_ct = int(count_rect[0][5])
    max_ct = int(count_rect[-1][5])
    count_x = [x for x in range(min_ct, max_ct+1)]
    count_std = []
    for x in count_x:
        locs = []
        for elem in count_rect:
            if elem[5] == x:
                locs.append(elem[0])
        count_std.append(np.std(locs))

    plt.plot(count_x, count_std)
    plt.title("Count-Std")
    plt.xlabel("counts")
    plt.ylabel("Std of dists")
    plt.savefig(os.path.join(resfolder, "CountStd.png"))
    plt.cla()
    plt.clf()
    with open(os.path.join(resfolder, "CountStd.csv"), "w") as f:
        f.write("Count;Std\n")
        for i in range(len(count_x)):
            f.write(f"{count_x[i]};{count_std[i]}\n")

    # Conf
    conf_rect = copy.deepcopy(pairs)
    conf_rect = sorted(conf_rect, key=lambda x: x[6])
    conf_x = []
    conf_std = []
    for i in range(width, len(conf_rect) - width):
        locds = [x[0] for x in conf_rect[i - width:i + width]]
        conf_x.append(conf_rect[i][6])
        conf_std.append(np.std(locds))
    plt.plot(conf_x, conf_std)
    plt.title("Conf-Std")
    plt.xlabel("confidence")
    plt.ylabel("Std of dists")
    plt.savefig(os.path.join(resfolder, "ConfStd.png"))
    plt.cla()
    plt.clf()
    with open(os.path.join(resfolder, "ConfStd.csv"), "w") as f:
        f.write("Conf;Std\n")
        for i in range(len(conf_x)):
            f.write(f"{conf_x[i]};{conf_std[i]}\n")

    # Angles
    angl_rect = copy.deepcopy(pairs)
    angl_rect = sorted(angl_rect, key=lambda x: x[7])
    angl_rect = list(filter(lambda x : x[7] > 0, angl_rect))
    angl_x = []
    angl_std = []
    for i in range(width, len(angl_rect) - width):
        locds = [x[0] for x in angl_rect[i - width:i + width]]
        angl_x.append(angl_rect[i][7])
        angl_std.append(np.std(locds))
    plt.plot(angl_x, angl_std)
    plt.title("Angle-Std")
    plt.xlabel("angle")
    plt.ylabel("Std of dists")
    plt.savefig(os.path.join(resfolder, "AngleStd.png"))
    plt.cla()
    plt.clf()
    with open(os.path.join(resfolder, "AngleStd.csv"), "w") as f:
        f.write("Angle;Std\n")
        for i in range(len(angl_x)):
            f.write(f"{angl_x[i]};{angl_std[i]}\n")

def est_std_conf(conf):
    return 23.04881 - 5.36682 * conf

def est_std_gauss(gauss):
    return 9.10456 + 51.32145 * gauss

def est_std_count(count):
    if count == 1:
        return 50
    return 8.54324 + 2.80466 * count

def est_std_size(size):
    return 6.143 + 16.4093 * size

def est_std_rect(rect):
    return 51.1157 - 52.70972 * rect

def est_std_angle(theta):
    if theta < 0:
        return 1000
    if theta > np.pi/2:
        theta = np.pi - theta
    return 6.05617 + 19.99393 * theta

def est_std(r, g, s, a, c, t):
    return est_std_rect(r) * est_std_gauss(g) * est_std_count(a) * est_std_size(s) * est_std_conf(c) * est_std_angle(t) / 49000000


def test_std_funcs(q_file, resfolder):
    os.makedirs(resfolder, exist_ok=True)
    first = True
    fns = []
    dists = []
    thetas = []
    abws = []
    est_errs = []
    rects = []
    gausss = []
    sizes = []
    counts = []
    confs = []
    angles = []
    err_rects = []
    err_gausss = []
    err_sizes = []
    err_counts = []
    err_confs = []

    width = 100
    with open(q_file, "r") as f:
        for line in tqdm(f, desc="Reading Quality indcs"):
            if first:
                first = False
                continue
            parts = line.strip().split(",")
            fns.append(parts[0])
            dists.append(float(parts[1]))
            thetas.append(float(parts[2]))
            abws.append(float(parts[3]))
            ee = parts[4]
            if ee == "nan" or float(ee) > 1e8:
                ee = np.infty
            est_errs.append(float(ee))
            rects.append(float(parts[5]))
            gausss.append(float(parts[6]))
            sizes.append(float(parts[7]))
            counts.append(float(parts[8]))
            confs.append(float(parts[9]))
            angles.append(float(parts[10]))
            err_rects.append(float(parts[11]))
            err_gausss.append(float(parts[12]))
            err_sizes.append(float(parts[13]))
            err_counts.append(float(parts[14]))
            err_confs.append(float(parts[15]))

    with open(os.path.join(resfolder, "test_fks.csv"), "w") as f:
        f.write("File;dist;theta;abw;rect;gauss;size;count;conf;angle;err_r;err_g;err_s;err_a;err_c;est_err;std_rect;std_gauss;std_size;std_count;std_conf;std_err\n")

        for i in tqdm(range(len(fns))):
            f.write(f"{fns[i]};{dists[i]};{thetas[i]};{abws[i]};{rects[i]};{gausss[i]};{sizes[i]};{counts[i]};{confs[i]};{angles[i]};{err_rects[i]};{err_gausss[i]};{err_sizes[i]};")
            f.write(f"{err_counts[i]};{err_confs[i]};{est_errs[i]};{est_std_rect(rects[i])};{est_std_gauss(gausss[i])};{est_std_size(sizes[i])};{est_std_count(counts[i])};{est_std_conf(confs[i])};{est_std(rects[i], gausss[i], counts[i], sizes[i], confs[i])}\n")

def multi_std_fit(q_file, resfolder):
    os.makedirs(resfolder, exist_ok=True)
    resf = open(os.path.join(resfolder, "results.csv"), "w")
    first = True
    fns = []
    dists = []
    thetas = []
    abws = []
    est_errs = []
    rects = []
    gausss = []
    sizes = []
    counts = []
    confs = []
    angles = []
    err_rects = []
    err_gausss = []
    err_sizes = []
    err_counts = []
    err_confs = []
    err_angls = []
    std_rects = []
    std_gausss = []
    std_sizes = []
    std_counts = []
    std_confs = []
    std_angls = []

    with open(q_file, "r") as f:
        for line in tqdm(f, desc="Reading Quality indcs"):
            if "nan" in line or "inf" in line:
                continue
            if first:
                first = False
                continue
            parts = line.strip().split(",")
            fns.append(parts[0])
            dists.append(float(parts[1]))
            thetas.append(float(parts[2]))
            abws.append(float(parts[3]))
            ee = parts[4]
            if ee == "nan" or float(ee) > 1e8:
                ee = np.infty
            est_errs.append(float(ee))
            rects.append(float(parts[5]))
            gausss.append(float(parts[6]))
            sizes.append(float(parts[7]))
            counts.append(float(parts[8]))
            confs.append(float(parts[9]))
            angles.append(float(parts[10]))
            err_rects.append(float(parts[11]))
            err_gausss.append(float(parts[12]))
            err_sizes.append(float(parts[13]))
            err_counts.append(float(parts[14]))
            err_confs.append(float(parts[15]))
            err_angls.append(float(parts[16]))
            std_rects.append(float(parts[17]))
            std_gausss.append(float(parts[18]))
            std_sizes.append(float(parts[19]))
            std_counts.append(float(parts[20]))
            std_confs.append(float(parts[21]))
            std_angls.append(float(parts[22]))

            cur_rects = []
            cur_gauss = []
            cur_sizes = []
            cur_count = []
            cur_confs = []
            cur_angls = []
            cur_gesas = []

            cur_rects.append(float(parts[23]))
            cur_gauss.append(float(parts[24]))
            cur_sizes.append(float(parts[25]))
            cur_count.append(float(parts[26]))
            cur_confs.append(float(parts[27]))
            cur_angls.append(float(parts[28]))
            cur_gesas.append(float(parts[29]))

    dists = np.array(dists)
    rects = np.array(rects)
    gausss = np.array(gausss)
    sizes = np.array(sizes)
    counts = np.array(counts)
    confs = np.array(confs)
    angles = np.array(angles)

    sr = sorted(rects)
    rects = (rects - np.median(rects)) / (sr[int(0.95*len(sr))] - sr[int(0.05*len(sr))])+ 0.5
    sg = sorted(gausss)
    gausss = (gausss - np.median(gausss)) / (sg[int(0.95 * len(sg))] - sg[int(0.05 * len(sg))])+ 0.5
    ss = sorted(sizes)
    sizes = (sizes - np.median(sizes)) / (ss[int(0.95 * len(ss))] - ss[int(0.05 * len(ss))]) + 0.5
    sa = sorted(counts)
    counts = (counts - np.median(counts)) / (sa[-1] - sa[0])+ 0.5
    sc = sorted(confs)
    confs = (confs - np.median(confs)) / (sc[int(0.95 * len(sc))] - sc[int(0.05 * len(sc))])+ 0.5
    st = sorted(angles)
    angles = (angles - np.median(angles)) / (st[int(0.95 * len(st))] - st[int(0.05 * len(st))])  + 0.5

    # plt.plot(rects)
    # plt.title("Scaled rects")
    # plt.show()
    vecs = []
    for i in range(len(rects)):
        if str(gausss[i]) != "nan":
            vecs.append(( (np.array([rects[i], gausss[i], sizes[i], counts[i], confs[i], angles[i]])) , dists[i]))
            # print("Appended ", vecs[-1])

    acc = 200
    vals = np.linspace(0, 1, 4)
    pairs = []
    for q1 in vals:
        for q2 in vals:
            for q3 in vals:
                for q4 in vals:
                    for q5 in vals:
                        for q6 in vals:
                            pairs.append(np.array([q1, q2, q3, q4, q5, q6]))

    for vec in vecs:
        print(vec)
    matplotlib.use("TkAgg")
    for pair in tqdm(pairs):
        # print("Pair: ", pair)
        # print("Vec0: ", vecs[0][0])
        locvecs = sorted(vecs, key = lambda x : np.linalg.norm(pair - x[0]))
        dists = [np.linalg.norm(pair - x[0]) for x in locvecs][:-20]
        # plt.plot(dists)
        # plt.show()
        # print("Pair: ", pair)
        # print("Closest: ", locvecs[0])
        # print("Cloest Dists: ", [np.linalg.norm(pair - x[0]) for x in locvecs[:10]])
        ldists = [x[1] for x in locvecs[:acc]]
        # print(" Ldists: ", ldists[:10])
        st = np.std(ldists)
        # print("STD: ", st)
        # input()
        resf.write(f"{pair[0]};{pair[1]};{pair[2]};{pair[3]};{pair[4]};{pair[5]};{st}\n")

    resf.close()



def get_interpolator(file, q_indcs):

    def get_trafos():
        first = True
        fns = []
        dists = []
        thetas = []
        abws = []
        est_errs = []
        rects = []
        gausss = []
        sizes = []
        counts = []
        confs = []
        angles = []
        err_rects = []
        err_gausss = []
        err_sizes = []
        err_counts = []
        err_confs = []
        err_angls = []
        std_rects = []
        std_gausss = []
        std_sizes = []
        std_counts = []
        std_confs = []
        std_angls = []

        with open(q_indcs, "r") as f:
            for line in tqdm(f, desc="Reading Quality indcs"):
                if "nan" in line or "inf" in line:
                    continue
                if first:
                    first = False
                    continue
                parts = line.strip().split(",")
                fns.append(parts[0])
                dists.append(float(parts[1]))
                thetas.append(float(parts[2]))
                abws.append(float(parts[3]))
                ee = parts[4]
                if ee == "nan" or float(ee) > 1e8:
                    ee = np.infty
                est_errs.append(float(ee))
                rects.append(float(parts[5]))
                gausss.append(float(parts[6]))
                sizes.append(float(parts[7]))
                counts.append(float(parts[8]))
                confs.append(float(parts[9]))
                angles.append(float(parts[10]))
                err_rects.append(float(parts[11]))
                err_gausss.append(float(parts[12]))
                err_sizes.append(float(parts[13]))
                err_counts.append(float(parts[14]))
                err_confs.append(float(parts[15]))
                err_angls.append(float(parts[16]))
                std_rects.append(float(parts[17]))
                std_gausss.append(float(parts[18]))
                std_sizes.append(float(parts[19]))
                std_counts.append(float(parts[20]))
                std_confs.append(float(parts[21]))
                std_angls.append(float(parts[22]))

        dists = np.array(dists)
        rects = np.array(rects)
        gausss = np.array(gausss)
        sizes = np.array(sizes)
        counts = np.array(counts)
        confs = np.array(confs)
        angles = np.array(angles)

        sr = sorted(rects)
        rectf = lambda x : (x - np.median(rects)) / (sr[int(0.95 * len(sr))] - sr[int(0.05 * len(sr))]) + 0.5
        sg = sorted(gausss)
        gaussf = lambda x :  (x - np.median(gausss)) / (sg[int(0.95 * len(sg))] - sg[int(0.05 * len(sg))]) + 0.5
        ss = sorted(sizes)
        sizef = lambda x : (x - np.median(sizes)) / (ss[int(0.95 * len(ss))] - ss[int(0.05 * len(ss))]) + 0.5
        sa = sorted(counts)
        countf = lambda x : (x - np.median(counts)) / (sa[int(0.95 * len(sa))] - sa[int(0.05 * len(sa))]) + 0.5
        sc = sorted(confs)
        conff = lambda x : (x - np.median(confs)) / (sc[int(0.95 * len(sc))] - sc[int(0.05 * len(sc))]) + 0.5
        st = sorted(angles)
        anglef = lambda x : (x - np.median(angles)) / (st[int(0.95 * len(st))] - st[int(0.05 * len(st))]) + 0.5

        return rectf, gaussf, sizef, countf, conff, anglef

    rectf, gaussf, sizef, countf, conff, anglef = get_trafos()

    points = []
    vals = []

    with open(file, "r") as f:
        for line in tqdm(f, desc="Reading File"):
            pairs = line.split(";")
            prs = [float(x) for x in pairs]

            points.append(np.array([prs[0], prs[1], prs[2], prs[3], prs[4], prs[5]]))
            vals.append(prs[-1])
    print("Start gen Interp")
    interp = LinearNDInterpolator(points, vals, fill_value=-1)
    print("End gen Interp")

    def fkt(r, g, s, a, c, t):
        q1 = rectf(r)
        q2 = gaussf(g)
        q3 = sizef(s)
        q4 = countf(a)
        q5 = conff(c)
        q6 = anglef(t)

        def crop(num):
            return min(max(0, num), 1)
        crop = np.vectorize(crop)

        q1 = crop(q1)
        q2 = crop(q2)
        q3 = crop(q3)
        q4 = crop(q4)
        q5 = crop(q5)
        q6 = crop(q6)

        return interp(q1, q2, q3, q4, q5, q6)

    return fkt

def get_fit(file, q_indcs):
    """
    q_indcs to optain rescaling functions used for file
    :param file:
    :param q_indcs:
    :return:
    """

    print("FIT FILE: ", q_indcs)

    def get_trafos():
        first = True
        fns = []
        dists = []
        thetas = []
        abws = []
        est_errs = []
        rects = []
        gausss = []
        sizes = []
        counts = []
        confs = []
        angles = []
        err_rects = []
        err_gausss = []
        err_sizes = []
        err_counts = []
        err_confs = []
        err_angls = []
        std_rects = []
        std_gausss = []
        std_sizes = []
        std_counts = []
        std_confs = []
        std_angls = []

        with open(q_indcs, "r") as f:
            for line in tqdm(f, desc="Reading Quality indcs"):
                if "nan" in line or "inf" in line:
                    continue
                if first:
                    first = False
                    continue
                parts = line.strip().split(",")
                fns.append(parts[0])
                dists.append(float(parts[1]))
                thetas.append(float(parts[2]))
                abws.append(float(parts[3]))
                ee = parts[4]
                if ee == "nan" or float(ee) > 1e8:
                    ee = np.infty
                est_errs.append(float(ee))
                rects.append(float(parts[5]))
                gausss.append(float(parts[6]))
                sizes.append(float(parts[7]))
                counts.append(float(parts[8]))
                confs.append(float(parts[9]))
                angles.append(float(parts[10]))
                err_rects.append(float(parts[11]))
                err_gausss.append(float(parts[12]))
                err_sizes.append(float(parts[13]))
                err_counts.append(float(parts[14]))
                err_confs.append(float(parts[15]))
                err_angls.append(float(parts[16]))
                std_rects.append(float(parts[17]))
                std_gausss.append(float(parts[18]))
                std_sizes.append(float(parts[19]))
                std_counts.append(float(parts[20]))
                std_confs.append(float(parts[21]))
                std_angls.append(float(parts[22]))

                # cur_rects = []
                # cur_gauss = []
                # cur_sizes = []
                # cur_count = []
                # cur_confs = []
                # cur_angls = []
                # cur_gesas = []
#
                # cur_rects.append(float(parts[23]))
                # cur_gauss.append(float(parts[24]))
                # cur_sizes.append(float(parts[25]))
                # cur_count.append(float(parts[26]))
                # cur_confs.append(float(parts[27]))
                # cur_angls.append(float(parts[28]))
                # cur_gesas.append(float(parts[29]))



        dists = np.array(dists)
        rects = np.array(rects)
        gausss = np.array(gausss)
        sizes = np.array(sizes)
        counts = np.array(counts)
        confs = np.array(confs)
        angles = np.array(angles)

        sr = sorted(rects)
        rectf = lambda x : (x - np.median(rects)) / (sr[int(0.95 * len(sr))] - sr[int(0.05 * len(sr))]) + 0.5
        sg = sorted(gausss)
        gaussf = lambda x :  (x - np.median(gausss)) / (sg[int(0.95 * len(sg))] - sg[int(0.05 * len(sg))]) + 0.5
        ss = sorted(sizes)
        sizef = lambda x : (x - np.median(sizes)) / (ss[int(0.95 * len(ss))] - ss[int(0.05 * len(ss))]) + 0.5
        sa = sorted(counts)
        countf = lambda x : (x - np.median(counts)) / (sa[-1] - sa[0]) + 0.5
        sc = sorted(confs)
        conff = lambda x : (x - np.median(confs)) / (sc[int(0.95 * len(sc))] - sc[int(0.05 * len(sc))]) + 0.5
        st = sorted(angles)
        anglef = lambda x : (x - np.median(angles)) / (st[int(0.95 * len(st))] - st[int(0.05 * len(st))]) + 0.5

        return rectf, gaussf, sizef, countf, conff, anglef

    rectf, gaussf, sizef, countf, conff, anglef = get_trafos()

    points = []
    vals = []

    with open(file, "r") as f:
        for line in tqdm(f, desc="Reading File"):
            pairs = line.split(";")
            prs = [float(x) for x in pairs]

            points.append(np.array([prs[0], prs[1], prs[2], prs[3], prs[4], prs[5]]))
            vals.append(prs[-1])

    initial = []
    initial.append(0)
    for i in range(6):
        initial.append(0)
    for i in range(6):
        for j in range(6):
            initial.append(0)

    def f(data,
          p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
          p10, p11, p12, p13, p14, p15, p16, p17, p18, p19,
          p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
          p30, p31, p32, p33, p34, p35, p36, p37, p38, p39,
          p40, p41, p42):



        params = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
          p10, p11, p12, p13, p14, p15, p16, p17, p18, p19,
          p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
          p30, p31, p32, p33, p34, p35, p36, p37, p38, p39,
          p40, p41, p42])

        # print("Provided: ", data.shape, params.shape)

        if len(data.shape) == 2:
            retu = []
            for v in range(data.shape[0]):
                ret = params[0]
                for i in range(6):
                    ret += params[1+i] * data[v, i]

                for i in range(6):
                    for j in range(6):
                        ret += params[7 + 6*i + j] * data[v, i] * data[v, j]

                retu.append(ret)
            ret = np.array(retu)
        else:
            ret = params[0]
            for i in range(6):
                ret += params[1 + i] * data[i]

            for i in range(6):
                for j in range(6):
                    ret += params[7 + 6 * i + j] * data[i] * data[j]

        # print("Returning: ", ret)

        return ret

    #for point in points:
    #    print("Pt: ", point)
    points = np.array(points)
    # print("PT0: ", type(points[0]))
    vals = np.array(vals)
    initial = np.array(initial)
    # print("FI ", f(np.array([1, 2, 3, 4, 5, 6]), *initial))
    # print(len(points))
    # print(len(vals))
    # print(f(points[0], *initial))
    # print(len(f(points, *initial)))
    # print(type(points))
    # print(type(vals))
    popt, pcov = curve_fit(f, points, vals)

    def fkt(r, g, s, a, c, t):
        if a == 1:
            return 20
        if t < 0:
            return 20
        q1 = rectf(r)
        q2 = gaussf(g)
        q3 = sizef(s)
        q4 = countf(a)
        q5 = conff(c)
        q6 = anglef(t)

        return f(np.array([q1, q2, q3, q4, q5, q6]), *popt)

    return fkt

if __name__ == "__main__":
    multi_std_fit("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try368_DS_FIT_5050\\quality\\quality_indcs.csv", "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try{}_StdPlot".format(len(os.listdir('D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res'))))
    # visualize_errors("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try{}_Viz_Errfkt".format(len(os.listdir('D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res'))))

    # find_std_funcs("D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try200_StdQ_mitGauss\\quality\\quality_indcs.csv", "D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res\\Try{}_StdPlot".format(len(os.listdir('D:\\Dateien\\KI_Speicher\\EvalChainDS\\TotalDatasets\\Res'))))