from PIL import Image, ImageOps
import numpy as np
import torch
import matplotlib.pyplot as plt



def ssMask2img(arr : torch.Tensor, colormat=False):
    assert len(arr.shape) == 3
    channels = arr.shape[0]
    col = 255 / channels
    # mat2 = arr.argmax(dim=0)
    # plt.imshow(mat2.cpu())
    # plt.title("Argmac")
    # plt.show()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    min = np.amin(arr, axis=0)
    max = np.amax(arr, axis=0)
    #print("Min: , ", min)
    arr = arr - min
    arr = arr / (max - min + 1e-6)
    mat = np.zeros(arr.shape[:-1], dtype=int)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # print(arr[i, j])
            # print("argmax: ", np.argmax(arr[i, j]), "col ", col)
            mat[i, j] = np.argmax(arr[i, j])

    if colormat:
        cmap = np.zeros(np.shape(arr))

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(3):
                    cmap[i, j, k] = arr[i, j][k]

        cmap *= 255
        cmap = cmap.astype(np.uint8)
        return mat, Image.fromarray(cmap)

    return mat

