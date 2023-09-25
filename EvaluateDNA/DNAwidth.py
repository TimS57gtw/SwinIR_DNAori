import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from ManageTrainData import hoshen_koppelmann
from tqdm import tqdm
from multiprocessing import Process, Manager
def bilinear_interpol(mat, pos):
    if not (0 <= pos[0] <= np.shape(mat)[0] - 1 and 0 <= pos[1] <= np.shape(mat)[1] - 1):
        return 0
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
def calc_width(file):
    img = Image.open(file)
    arr = np.array(img)[:, :, 0]


    find_mark = lambda x : 1 if x == 255 else 0
    find_mark = np.vectorize(find_mark)

    mark = find_mark(arr).astype(int)

    clusers = hoshen_koppelmann(mark)


    if np.amax(clusers) != 2:
        return -1, -1, -1

    cls = np.argwhere(clusers == 1)
    cls2 = np.argwhere(clusers == 2)

    x0 = 0
    y0 = 0
    for i in range(len(cls)):
        x0 += cls[i][1]
        y0 += cls[i][0]

    x1 = 0
    y1 = 0
    for i in range(len(cls2)):
        x1 += cls2[i][1]
        y1 += cls2[i][0]

    p0 = np.array([y0, x0]) / len(cls)
    p1 = np.array([y1, x1]) / len(cls2)

    # print(p0)
    # print(p1)
    markerv = p1 - p0
    lenfkt = lambda x : p0 + x * markerv

    orthovec = np.array([1, -markerv[0] / markerv[1]])
    orthovec /= np.linalg.norm(orthovec)

    # print(np.dot(orthovec, markerv))
#
    # print("Ortho: ", orthovec)
    # print("Len: ", markerv)

    ll = 501# 251
    wl = 201# 51
    scan_martix = np.zeros((ll, wl)) #len, wid
    lenspts = np.linspace(-0.75, 1.75, ll)
    len_unit = np.linalg.norm(markerv) * (np.amax(lenspts) - np.amin(lenspts)) / ll
    widspts = np.linspace(-25, 25, wl)
    wid_unit = (np.amax(widspts) - np.amin(widspts)) / wl

    # with open(r'C:\Users\seifert\Documents\MoveToD\DNA_wid\scanning.csv', 'w') as f:

    for i in range(len(lenspts)):
        for j in range(len(widspts)):
            posi = lenfkt(lenspts[i]) + orthovec * widspts[j]
            val = bilinear_interpol(arr, posi)
            # f.write(f"{posi[0]};{posi[1]};{val}\n")
            scan_martix[i, j] = val


    # plt.imshow(clusers)
    # plt.show()
    # plt.switch_backend('TkAgg')
    # plt.imshow(scan_martix)
    # plt.show()

    # Get widths

    cw = int(np.floor(wl/2))
    leftedge = []
    rightedge = []
    th = 100
    for j in tqdm(range(ll), disable=True):
        rl = 0
        while scan_martix[j, cw - rl] > th and cw - rl > 0:
            rl += 1
        leftedge.append(-rl)
        rr = 0
        while cw + rr < scan_martix.shape[1] and scan_martix[j, cw + rr] > th:
            rr += 1
        rightedge.append(rr)



    ws = []
    for i in range(len(leftedge)):
        if leftedge[i] < 0 and rightedge[i] > 0:
            ws.append(rightedge[i] - leftedge[i])

    mollen = len(ws)
    molw = np.median(ws)


    # fig, axs = plt.subplots(1,2)
    # axs[0].plot(leftedge)
    # axs[0].plot(rightedge)
    # axs[1].imshow(scan_martix.T)
    # plt.gca().set_aspect('auto')
    # plt.title(f"L={mollen}x{len_unit}, W={molw}x{wid_unit}")
    # plt.show()

    return molw * wid_unit, mollen * len_unit, arr.shape[0]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def task(files, dict):
    for file in files:
        dict[os.path.basename(file)] = calc_width(file)

def analyze_run(fld):
    threads = 16
    resfile = os.path.join(fld, "width_measurements.csv")
    ss_fld = os.path.join(fld, "ss_labels")

    tempdict = {}
    tempdict_distances = {}
    tempdict_file = os.path.join(fld, "Eval_Results", "tempdict.csv")
    with open(tempdict_file, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            parts = line.strip().split(";")
            tempdict[parts[0]] = parts[1]
            tempdict_distances[parts[0]] = float(parts[2])

    resdict = {}

    files = [os.path.join(ss_fld, x) for x in os.listdir(ss_fld)]
    filelists = list(chunks(files, int(np.ceil(len(files)/threads))))
    man = Manager()
    retd = man.dict()
    ts = []
    for i in range(len(filelists)):
            ts.append(Process(target=task, args=(filelists[i], retd)))
    pbar = tqdm(total=len(files), desc="Calc Dims")
    old = 0
    for t in ts:
        t.start()
    while len(retd.keys()) < len(files):
        diff = len(retd.keys()) - old
        old += diff
        pbar.update(diff)
        time.sleep(1)
    for t in ts:
        t.join()


    for file in tqdm(os.listdir(ss_fld), total=len(os.listdir(ss_fld))):
        bn = file.split(".")[0]
        resdict[bn] = {}
        w, l, ims = retd[file]
        if w < 0:
            continue
        resdict[bn]['ims'] = ims
        resdict[bn]['w_px'] = w
        resdict[bn]['l_px'] = l
        resdict[bn]['dist'] = tempdict_distances[bn]

        crp_fld = tempdict[bn]
        with open(os.path.join(crp_fld, "Image.txt"), 'r') as f:
            for line in f:
                if line.startswith("Crop_nMpPX_X"):
                    parts = line.strip().split(":")
                    nmppx = float(parts[1])
                    if not 0.1 < nmppx < 20:
                        print(line, nmppx)
                    resdict[bn]["nm_p_px"] = nmppx
                if line.startswith("Crop_Px_X"):
                    parts = line.strip().split(":")
                    cropsz = float(parts[1])
                    resdict[bn]["cropsz"] = cropsz
        resdict[bn]['w_nm'] = resdict[bn]['w_px'] * resdict[bn]["cropsz"] * resdict[bn]["nm_p_px"] / resdict[bn]['ims']
        resdict[bn]['l_nm'] = resdict[bn]['l_px'] * resdict[bn]["cropsz"] * resdict[bn]["nm_p_px"] / resdict[bn]['ims']

    with open(resfile, 'w') as f:
        f.write("name;w_px;l_px;nm_p_px;w_nm;l_nm;dist\n")
        for elem in resdict.keys():
            try:
                f.write(f"{elem};{resdict[elem]['w_px']};{resdict[elem]['l_px']};{resdict[elem]['nm_p_px']};{resdict[elem]['w_nm']};{resdict[elem]['l_nm']};{resdict[elem]['dist']}\n")
            except KeyError:
                continue

if __name__ == "__main__":
    plt.switch_backend('TkAgg')
    # fld = r'D:\seifert\PycharmProjects\DNAmeasurement\Output\Try144_AirNewCF_FIT_UseU_True_LaterTime_Conf70_SynB4_70EP_SaveSXM'
    fld = r'D:\seifert\PycharmProjects\DNAmeasurement\Output\Try167_NoBirka_FIT_UseU_True_LaterTime_Conf70_SynB4_70EP_XY_'
    analyze_run(fld)

    assert 1 == 2





