from multiprocessing import Process

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
from ManageTrainData import analyze_density

def task(runs, qu, num, last, px=None, fld=None):
    std_fak = 0
    spot_brought_mu = 55# 55  # 40 # 40 # 30  # 70 #
    spot_brought_sig = 8.25 * std_fak  # 6 # 6 # 5

    lss = lambda md: (1 / 178.9467) * md - 0.00118674

    for runID in tqdm(range(runs if px is None else len(px)), disable=not last):
        if runID >= len(px):
            return

        # location_ss_u = 30.45 / 87
        # location_ss_l = 30.45 / 87

        if px is not None:
            molw, moll, molh, mard, immw = px[runID]

            maxi = max(molw, moll)
            mini = min(molw, moll)

            molw = mini
            moll = maxi

            px_a = 64/immw
            location_ss_u = lss(mard)
            location_ss_l = lss(mard)
        else:
            px_a = np.random.uniform(256 / 18000, 256 / 12000)

        # length = np.random.normal(900, 20)
        # width = np.random.normal(700, 20)  # a
        if std_fak == 0:
            length = 900

        else:
            length = np.random.uniform(600, 1200)
            location_ss_u = np.random.uniform(20, 43) / 87
            location_ss_l = np.random.uniform(20, 43) / 87

        # location_ss_u = 30.45/87
        # location_ss_l = 30.45/87
        # print("pxl", pxl, "lss", location_ss_l * 87)

        ss_row_up = np.random.normal(location_ss_u * length, 20*std_fak)
        ss_row_low = -np.random.normal(location_ss_l * length, 20*std_fak)
        position_y_row_up = ss_row_up  # Lambda Function
        position_y_row_low = ss_row_low  # Lambda Function
        pos_y_sig = 10  * std_fak # a
        height_mu = 15  # a
        height_sig = 2 * std_fak  # a

        matrix_shape = int(np.ceil(length * px_a))

        position_y = lambda upper: np.random.normal(position_y_row_up, pos_y_sig) if upper else np.random.normal(
            position_y_row_low, pos_y_sig)
        if std_fak == 0:
            no_of_spots = 5
        else:
            no_of_spots = random.randint(4, 6)

        spot_brough = lambda: np.random.normal(spot_brought_mu, spot_brought_sig)  # a
        spot_positions_upper = (position_y(True), spot_brough())
        spot_positions_lower = (position_y(False), spot_brough())

        mat_idcs = np.zeros(matrix_shape, dtype="object")
        for i in range(matrix_shape):
                mat_idcs[i] = i


        def border_abfall(y):
            border_abst_y = 30  # a
            border_sigma_y = max(1e-2, 7 * std_fak)

            if y < 0:
                dy = +length / 2 + y
            else:
                dy = length / 2 - y

            fy = (1 - (1 / (np.exp((dy - border_abst_y) / border_sigma_y) + 1)))
            # print(f"x = {x}, dx={dx} -> f(x)={fx}")
            # print(f"y = {y}, dy={dy} -> f(y)={fy}")
            # print(f"({x, y}) -> f(x)={fx * fy}")



            return  fy

        def fkt(y, pos, sig):
            return np.exp(-(y/px_a - pos)**2 / sig**2) * border_abfall(y/px_a)

        measured_yu = 0
        measured_yl = 0
        measured_vu = 0
        measured_vl = 0
         #print(spot_positions_upper)
         #print(spot_positions_lower)
        mat_u = np.array([fkt(i - matrix_shape/2, spot_positions_upper[0], spot_positions_upper[1]) for i in range(matrix_shape)])

        plt.switch_backend('tkAgg')



        xs = np.array([x for x in range(len(mat_u))])
        measured_yu += np.dot(xs, mat_u)
        measured_vu += np.sum(mat_u)

        mat_l = np.array([fkt(i - matrix_shape/2, spot_positions_lower[0], spot_positions_lower[1]) for i in range(matrix_shape)])


        plt.switch_backend('tkAgg')

        xs = np.array([x for x in range(len(mat_l))])
        measured_yl += np.dot(xs, mat_l)
        measured_vl += np.sum(mat_l)

        # print("PosU: ", spot_positions_upper[0])
        # print("PosL: ", spot_positions_lower[0])
        # plt.plot(mat_u, label='U')
        # plt.plot(mat_l, label='L')
        # plt.legend()
        # plt.title("Matu")
        # plt.show()




       # plt.imshow(np.maximum(mat_l, mat_u))
       # plt.title(f"Mc, px={px[runID]}")
       # plt.show()
       # if fld is not None and not os.path.isfile(os.path.join(fld, f"{int(round(px[runID]))}.png")):
       #     plt.plot(mat_u, label="u")
       #     plt.plot(mat_l, label="l")
       #     plt.title(f"Res: {int(round(px[runID]))} IsL: {0.1*spot_positions_lower[1]:.3f} IsR: {0.1*spot_positions_upper[0]:.3f}  -> D: {0.1*abs(measured_yu/measured_vu - measured_yl/measured_vl)/px_a:.4f}")
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
                os.makedirs(fld, exist_ok=True)
                with open(os.path.join(fld, "markDist.csv"), 'a') as g:
                    g.write(f"{molw};{moll};{molh};{mard};{immw};{measured_marker_distance/10}\n")
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
    write_buffer = 1000
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

    pxsi = np.linspace(maxpx, minpx, runs)
    pxs = [x for x in pxsi]

    pxs = np.linspace(50, 70, runs) # molw, moll, molh, mard, immw

    np.random.shuffle(pxs)
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
    # px = [x for x in range(100, 2000,5)]
    # task(1000, multiprocessing.Queue(), 1, True, px)
    # assert 5 == 9

    estimate_pos_distro(r"C:\Users\seifert\Documents\MoveToD\DNAMeas\imgWMarkCorr_0505_V93_line_MolWSmallerMolL\location_range2.csv")
