import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from tqdm import tqdm
from matplotlib import ticker
from sklearn import manifold, datasets
from numpy.random import RandomState
from PIL import Image
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def save_2d(points, points_color, title, fn):
    xs = []
    ys = []
    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    xs = sorted(xs)
    ys = sorted(ys)
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    ax.set_xlim(left=xs[int(0.05*len(xs))], right=xs[int(0.95*len(xs))])
    ax.set_ylim(bottom=ys[int(0.05*len(ys))], top=ys[int(0.95*len(ys))])

    plt.savefig(fn)
    plt.close()

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()

def stack_imgs(imgs, limit_num=None):
    x, y = imgs[0].shape
    num = min(len(imgs), limit_num) if limit_num is not None else len(imgs)
    #print(num, w, h)
    newmat = np.zeros(shape=(num, x*y))

    for i in range(num):
        newmat[i, :] = imgs[i].flatten()
    # newmat = newmat.T
    plt.imshow(newmat)
    plt.show()
    print("Newmat: ", newmat.shape)
    return newmat

if __name__ == "__main__":

    resfld = "E:\\11_22\\SURF_TSNE\\Try{}_TSNE".format(len(os.listdir("E:\\11_22\\SURF_TSNE")))

    os.makedirs(resfld, exist_ok=True)
    # rng = RandomState(0)
    mode="ABW"

   # t_sne = manifold.TSNE(
   #     n_components=2,
   #     perplexity=30,
   #     n_iter=250,
   #     init="random",
   #     random_state=rng,
   # )
   # S_t_sne = t_sne.fit_transform(S_points)
    # plot_3d(S_points, S_color, "Original S-curve samples")
#
    # plot_2d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")


    images = "E:\\11_22\\Try402_DS_INTER_Norm_RO005_E4X1\\pp_crop_origami_npy"
    labels = "E:\\11_22\\Try402_DS_INTER_Norm_RO005_E4X1\\ss_labels"
    quality = "E:\\11_22\\Try402_DS_INTER_Norm_RO005_E4X1\\quality\\quality_indcs.csv"

    t_sne = TSNE(
        n_components=2,
        perplexity=40,
        n_iter=500,
        init="random"
    )



    q_dict = {}
    with open(quality, "r") as f:
        for line in f:
            parts = line.split(",")
            try:
                if mode == "ABW":
                    q_dict[parts[0]] = np.log(float(parts[3])) # 3: Abw, 4: EE
                else:
                    q_dict[parts[0]] = np.log(min(15.15, float(parts[4]))) # 3: Abw, 4: EE
            except ValueError:
                q_dict[parts[0]] = np.inf
    im_files = [os.path.join(images, x) for x in os.listdir(images)]
    lbl_files = [os.path.join(labels, x) for x in os.listdir(labels)]

    im_fn = [os.path.basename(x).split(".")[0] for x in im_files]
    quals = []
    for x in im_fn:

        quals.append(q_dict[x])

    plt.scatter([x for x in range(len(quals))], quals)
    plt.title(mode)
    plt.show()

    centers = []
    spreads = []
    abws = []

    im_arrs = []
    lb_arrs = []

    for idx, f in enumerate(tqdm(im_files)):
        if f.endswith("npy"):
            img = np.load(f, allow_pickle=True)
        else:
            img = Image.open(f)
            img = np.array(img)
            img = img[:, :, 0]

        g = lbl_files[idx]
        if g.endswith("npy"):
            lbl = np.load(g, allow_pickle=True)
        else:
            lbl = Image.open(g)
            lbl = np.array(lbl).astype(float)
            lbl = lbl[:, :, 0]
        lbl /= 255.0


        im_arrs.append(img)
        lb_arrs.append(lbl)
        # if idx == 50:
        #     break

    imgs = stack_imgs(im_arrs, limit_num=None)
    # imgs = np.stack(im_arrs)
    lbls = stack_imgs(lb_arrs, limit_num=None)
    # lbls = np.stack(lb_arrs)




    S_TSNE = t_sne.fit_transform(imgs)
    print("N Features: ", t_sne.n_features_in_)
    print("Img Shape: ", S_TSNE.shape)
    with open(os.path.join(resfld, "img.csv"), "w") as f:
        for i in range(len(S_TSNE)):
            f.write(f"{im_fn[i]};{quals[i]};{S_TSNE[i][0]};{S_TSNE[i][1]}\n")
    save_2d(S_TSNE, quals, "Images", os.path.join(resfld, "ImageRes.png"))


    S_TSNE = t_sne.fit_transform(lbls)
    print("N Features: ", t_sne.n_features_in_)
    print("Lbl Shape: ", S_TSNE.shape)
    with open(os.path.join(resfld, "lbl.csv"), "w") as f:
        for i in range(len(S_TSNE)):
            f.write(f"{im_fn[i]};{quals[i]};{S_TSNE[i][0]};{S_TSNE[i][1]}\n")
    save_2d(S_TSNE, quals, "Labels", os.path.join(resfld, "LabelRes.png"))


    dotp = imgs * lbls
    S_TSNE = t_sne.fit_transform(dotp)
    print("N Features: ", t_sne.n_features_in_)
    print("Dot Shape: ", S_TSNE.shape)
    with open(os.path.join(resfld, "dot.csv"), "w") as f:
        for i in range(len(S_TSNE)):
            f.write(f"{im_fn[i]};{quals[i]};{S_TSNE[i][0]};{S_TSNE[i][1]}\n")

    save_2d(S_TSNE, quals, "Dotp", os.path.join(resfld, "DotpRes.png"))






