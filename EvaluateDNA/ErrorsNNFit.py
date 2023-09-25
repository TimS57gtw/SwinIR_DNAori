import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
DEVICE = torch.device("cuda:0")


def prepare_dataset(file, delim=",", v_split=0.1, t_split=0.1, apply_fkt=False, savefld=None):
    rs = []
    gs = []
    ss = []
    ass = []
    cs = []
    ds = []
    cerrs = []
    abws = []
    with open(file, "r") as f:
        # rectang, gauss, size, count, conf, abw, est_err
        for line in f:
            parts = line.split(delim)
            rs.append(float(parts[0].strip()))
            gs.append(min(0.85, float(parts[1].strip())))
            ss.append(float(parts[2].strip()))
            ass.append(float(parts[3].strip()))
            cs.append(float(parts[4].strip()))
            ds.append(float(parts[5].strip()))
            cerrs.append(min(2.5, float(parts[6].strip())))

    tr_imgs = int(len(rs) * (1 - v_split - t_split))
    v_imgs = int(len(rs) * v_split)
    r_avg = np.average(rs)
    g_avg = np.average(gs)
    s_avg = np.average(ss)
    a_avg = np.average(ass)
    c_avg = np.average(cs)
    r_srt = sorted(rs)
    g_srt = sorted(gs)
    s_srt = sorted(ss)
    a_srt = sorted(ass)
    c_srt = sorted(cs)
    r_spn = r_srt[int(0.9 * len(r_srt))] - r_srt[int(0.1 * len(r_srt))]
    g_spn = g_srt[int(0.9 * len(g_srt))] - g_srt[int(0.1 * len(g_srt))]
    s_spn = s_srt[int(0.9 * len(s_srt))] - s_srt[int(0.1 * len(s_srt))]
    a_spn = a_srt[int(0.9 * len(a_srt))] - a_srt[int(0.1 * len(a_srt))]
    c_spn = c_srt[int(0.9 * len(c_srt))] - c_srt[int(0.1 * len(c_srt))]

    r_fkt = lambda x: (x - r_avg) / r_spn
    g_fkt = lambda x: (x - g_avg) / g_spn
    s_fkt = lambda x: (x - s_avg) / s_spn
    a_fkt = lambda x: (x - a_avg) / a_spn
    c_fkt = lambda x: (x - c_avg) / c_spn


    if savefld is not None:
        with open(os.path.join(savefld, "constants.csv"), "w") as f:
            f.write(f"R,{r_avg},{r_spn}\n")
            f.write(f"G,{g_avg},{g_spn}\n")
            f.write(f"S,{s_avg},{s_spn}\n")
            f.write(f"A,{a_avg},{a_spn}\n")
            f.write(f"C,{c_avg},{c_spn}\n")


    pairs = []
    for i in range(len(rs)):
        if apply_fkt:
            pairs.append((r_fkt(rs[i]),
                          g_fkt(gs[i]),
                          s_fkt(ss[i]),
                          a_fkt(ass[i]),
                          c_fkt(cs[i]),
                          ds[i]))
        else:
            pairs.append((rs[i], gs[i], ss[i], ass[i], cs[i], ds[i]))

    np.random.shuffle(pairs)
    tr_pairs = pairs[:tr_imgs]
    v_pairs = pairs[tr_imgs:tr_imgs + v_imgs]
    t_pairs = pairs[tr_imgs + v_imgs:]

    return tr_pairs, v_pairs, t_pairs, r_fkt, g_fkt, s_fkt, a_fkt, c_fkt


class Dataset:

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inpt = torch.from_numpy(np.array(self.pairs[idx][:5]))
        tar = torch.from_numpy(np.array([self.pairs[idx][5]]))
        return inpt, tar


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.l1 = nn.Linear(5, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 5)
        self.l4 = nn.Linear(5, 1)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn3 = nn.BatchNorm1d(5)


    def forward(self, x):
        x = self.l1(x)
        x = F.tanh(x)
        # x = self.l2(x)
        # x = F.tanh(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = F.sigmoid(x)
        # x = 2.5 * F.sigmoid(x)
        return x


tls = []
vls = []
iters = []


def train(filepath, savefld):
    tr_pairs, v_pairs, t_pairs, rf, gf, sf, af, cf = prepare_dataset(filepath, apply_fkt=True, savefld=savefld)
    train_set = Dataset(tr_pairs)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=2, drop_last=True)
    net = Net()
    net = net.to(device=DEVICE)
    net.train()
    crit = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1)
    epochs = 3
    loss = 1
    step = 0
    with torch.autocast(enabled=False, device_type='cuda'):
        for ep in range(epochs):
            for inpt, tar in tqdm(train_loader, desc=f"Training EP {ep + 1}"):
                inpt = inpt.to(device=DEVICE, dtype=torch.float)
                tar = tar.to(device=DEVICE, dtype=torch.float)
                optimizer.zero_grad()
                step += 1
                pred = net(inpt)
                loss = crit(pred, tar)
                tls.append(loss.item())
                iters.append(step)

                loss.backward()
                optimizer.step()

    plt.plot(iters, tls)
    plt.title("Train Loss")
    plt.show()

    test_set = Dataset(t_pairs)
    dists = []
    try:
        for i in range(10000):
            inpt, tar = test_set[i]
            inpt = inpt.to(device=DEVICE, dtype=torch.float)
            pred = net(inpt).item()

            print("Test: {} -> {}".format(tar.item(), pred))
            dists.append(abs(tar - pred))
    except IndexError:
        pass

    plt.hist(dists, bins=20)
    plt.xlim([0, 1])
    plt.title("Test Results")
    plt.show()

    torch.save(net.state_dict(), os.path.join(savefld, "Model.pth"))
    return net

def r_squared(inpt, outpt):
    mean_in = np.average(inpt)
    ssres = sum([(inpt[i] - outpt[i])**2 for i in range(len(inpt))])
    sstot = sum([(inpt[i] - mean_in)**2 for i in range(len(inpt))])
    return 1 - (ssres/sstot)


def test_dependencies(net, filepath, resfile):
    net.eval()
    tr_pairs, _, _, rf, gf, sf, af, cf = prepare_dataset(filepath, t_split=0, v_split=0, apply_fkt=False)
    train_set = Dataset(tr_pairs)
    test_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=16)

    resf = open(resfile, "w")
    rs = []
    gs = []
    ss = []
    ass = []
    cs = []
    ds = []
    cerrs = []  # class errs
    res = []
    resf.write("rect,gauss,size,count,conf,class_err,pred,abw\n")
    with open(filepath, "r") as f:
        for line in tqdm(f):
            parts = line.split(",")
            r = float(parts[0].strip())
            g = min(0.85, float(parts[1].strip()))
            s = float(parts[2].strip())
            a = float(parts[3].strip())
            c = float(parts[4].strip())
            d = float(parts[5].strip())
            cerr = min(2.5, float(parts[6].strip()))
            rs.append(r)
            gs.append(g)
            ss.append(s)
            ass.append(a)
            cs.append(c)
            ds.append(d)
            cerrs.append(cerr)

            inpt = torch.from_numpy(np.array([rf(r), gf(g), sf(s), af(a), cf(c)])).to(device=DEVICE, dtype=torch.float)

            pred = net(inpt).detach().cpu().numpy()
            res.append(pred)

            resf.write(f"{r},{g},{s},{a},{c},{cerr},{pred},{d}\n")


    slope, intercept, r_value, p_value, std_err = linregress(ds, cerrs)
    cerrs = np.array(cerrs)

    cerrs -= intercept
    cerrs /= slope
    xs = np.linspace(0, max(ds), 100)
    ys = np.vectorize(lambda x: x)(xs)
    plt.scatter(ds, cerrs)
    plt.plot(xs, ys, color="red")
    plt.title("Class vs act. diff, r2: {}".format(r_squared(ds, cerrs)))
    plt.xlabel("Abw")
    plt.ylim([0, max(ds)])
    plt.ylabel("Prediction")
    plt.show()

    xs = np.linspace(0, max(ds), 100)
    ys = np.vectorize(lambda x : x)(xs)
    plt.scatter(ds, res)
    plt.plot(xs, ys, color="red")
    plt.title("Pred vs act. diff, r2: {}".format(r_squared(ds, res)))
    plt.xlabel("Abw")
    plt.ylim([0, max(ds)])
    plt.ylabel("Prediction")
    plt.show()

    xs = np.linspace(0, max(ds), 100)
    ys = np.vectorize(lambda x: x)(xs)
    plt.scatter(ds, res)
    plt.plot(xs, ys)
    plt.title("Pred vs act. diff, r2: {}".format(r_squared(ds, res)))
    plt.xlabel("Abw")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.ylabel("Prediction")
    plt.show()

    resf.close()


if __name__ == '__main__':
    savefolder = os.path.join("D:\\Dateien\\KI_Speicher\\EvalChainDS\\NNFit\\V{}".format(len(os.listdir("D:\\Dateien\\KI_Speicher\\EvalChainDS\\NNFit"))))
    os.makedirs(savefolder, exist_ok=True)
    net = train(filepath=os.path.join("datasets", "NNFit", "TestResults.txt"), savefld=savefolder)
    test_dependencies(net=net, filepath=os.path.join("datasets", "NNFit", "TestResults.txt"),
                      resfile=os.path.join(savefolder, "results.csv"))
