# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:09:09 2022

@author: seife
"""


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def measurementError(meas):
    if meas is None:
        meas = [101.44, 0, 0, 104.48, 98.34,100.15, 96.30, 100.97,
               108.15, 94.06, 92.13, 0, 100.81, 105.41, 0, 0,
               0, 93.34, 101.33, 0, 0, 0]



    def streuung(l):
        j = len(l)
        m = np.average(l)
        return (1/(j-1)) * sum([(x - m)**2 for x in l])


    vmeas = [x for x in meas if x > 0]
    if len(vmeas) <= 1:
        return 0, 0, 0
    zero_pct = 1 - len(vmeas)/len(meas)



    total= len(vmeas)

    meas = vmeas

    sig = np.sqrt(streuung(meas))
    mu = np.average(meas)


    vmeas = [x for x in meas if x != 0]


    norm = lambda x : np.exp(-((x - mu)**2)/(2 * sig**2))/(sig * np.sqrt(2*np.pi))


    real_med = np.average(meas)
    real_streu = streuung(meas)




    # fehleranalyse 16.4.1.3

    vi = [real_med - x for x in meas]


    n = len(meas)
    stabw = np.sqrt(sum([v**2 for v in vi]) / (n * (n-1)))
    # print("Stabw: ", stabw)
    wahrsch_f = 0.6745 * stabw
    # print("Wahrscheinlicher Fehler: ", wahrsch_f)
    mittl_f = 0.7979 * stabw
    # print("Mittlerer Fehler: ", mittl_f

    return stabw, wahrsch_f, mittl_f



def analyzeError(meas):
    if meas is None:
        meas = [101.44, 0, 0, 104.48, 98.34,100.15, 96.30, 100.97,
               108.15, 94.06, 92.13, 0, 100.81, 105.41, 0, 0, 
               0, 93.34, 101.33, 0, 0, 0]
    def streuung(l):
        j = len(l)
        m = np.average(l)
        return (1/(j-1)) * sum([(x - m)**2 for x in l])


    vmeas = [x for x in meas if x != 0]
    zero_pct = 1 - len(vmeas)/len(meas)



    total= len(vmeas)

    meas = vmeas

    sig = np.sqrt(streuung(meas))
    mu = np.average(meas)

    def gen_samples(n):
        zeros = int(n * np.random.normal((zero_pct), 0.1))
        smples = np.random.normal(mu, sig, n)
        idx = [x for x in range(n)]
        np.random.shuffle(idx)
        for j in range(zeros):
            smples[idx[j]] = 0


        return smples

    meas = gen_samples(22)
    plt.hist(meas, bins=15)
    plt.title("Gen Samples")
    plt.show()
    vmeas = [x for x in meas if x != 0]
    plt.hist(vmeas, bins=15)
    plt.title("Valid Gen Samples")
    plt.show()
    meas = vmeas


    norm = lambda x : np.exp(-((x - mu)**2)/(2 * sig**2))/(sig * np.sqrt(2*np.pi))
    xs = np.linspace(min(meas), max(meas), 200)
    ys = norm(xs)
    plt.plot(xs, ys)
    plt.title("PDF")
    plt.show()

    real_med = np.average(meas)
    real_streu = streuung(meas)

    real_med = np.average(meas)
    real_streu = streuung(meas)


    plt.hist(meas, bins=15)
    plt.title("Valid Measurements")
    plt.show()

    print("Valid percentage: ", 100 * len(vmeas)/len(meas))
    print("Mean_real: {:.2f}".format(real_med))
    print("Sreu_real: {:.2f}".format(real_streu))



    # fehleranalyse 16.4.1.3
    test_samples = 1000
    meds = []
    oris = []
    stabws = []
    wfs = []
    mfs = []
    for meas_ori in tqdm(range(1, 200)):
        s1 = []
        w1 = []
        m1 = []
        d1 = []
        for k in range(test_samples):
            meas = gen_samples(meas_ori)

            meas = [x for x in meas if x != 0]
            #plt.hist(meas, bins=15)
            #plt.xlim([90, 110])
            #plt.title("Meas")
            #plt.show()
            real_med = np.average(meas)
            vi = [real_med - x for x in meas]


            n = len(meas)
            if n == 0:
                continue
            stabw = np.sqrt(sum([v**2 for v in vi]) / (n * (n-1)))
            # print("Stabw: ", stabw)
            wahrsch_f = 0.6745 * stabw
            # print("Wahrscheinlicher Fehler: ", wahrsch_f)
            mittl_f = 0.7979 * stabw
            # print("Mittlerer Fehler: ", mittl_f)

            s1.append(stabw)
            w1.append(wahrsch_f)
            m1.append(mittl_f)
            d1.append(real_med)

        oris.append(meas_ori)
        meds.append(np.average(d1))
        stabws.append(np.average(s1))
        wfs.append(np.average(w1))
        mfs.append(np.average(m1))

    plt.plot(oris, meds)
    plt.title("Medium")
    plt.show()

    plt.plot(oris, stabws)
    plt.title("Stabw")
    plt.show()

    plt.plot(oris, wfs)
    plt.title("Wahrsch Fehler")
    plt.show()
    plt.plot(oris, mfs)
    plt.title("Mittl Fehler")
    plt.show()

    p = 9
    print("{}M: {}".format(oris[p], stabws[p]))

    p = 19
    print("{}M: {}".format(oris[p], stabws[p]))

    p = 39
    print("{}M: {}".format(oris[p], stabws[p]))

    tar = 3
    k = 3
    while stabws[k] > tar:
        k+= 1
        if k == len(stabws) - 1:
            break

    print("{}M: {}".format(oris[k], stabws[k]))

    tar = 2
    k = 3
    while stabws[k] > tar:
        k+= 1
        if k == len(stabws) - 1:
            break

    print("{}M: {}".format(oris[k], stabws[k]))

    tar = 1
    k = 3
    while stabws[k] > tar:
        k+= 1
        if k == len(stabws) - 1:
            break

    print("{}M: {}".format(oris[k], stabws[k]))


def get_q_errors(meas=None, sig=None, n=None):
    ALPHA = 0.68895
    BETA = 0.28881
    if meas is None:
        assert sig is not None and n is not None, "Either Meas or Sig and N required"
        sindv = sig
        ntot = n
    else:
        ntot = len(meas)
        sindv = np.std(meas)

    nopt = int(round(0.4192 * ntot))
    sigopt = 2 * np.sqrt(ALPHA * BETA) * sindv / np.sqrt(ntot)

    # print("Estimated Achievable Error: {:.3f}nm with {} Images".format(sigopt, nopt))
    return sigopt, nopt


