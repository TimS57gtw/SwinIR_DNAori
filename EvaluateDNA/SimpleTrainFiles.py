import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import os


def simple_grid(resolution=100, cols=2, rows=2, classes=3, show=False):

    mat = np.zeros((resolution, resolution))
    lbl = np.zeros((resolution, resolution))

    assert classes > 1

    grid = np.random.randint(0, classes, size=(cols, rows))

    colsize = resolution / cols
    rows = resolution / rows

    g2col = lambda l : l * 1/(classes-1)

    for i in range(resolution):
        for j in range(resolution):
            lbl[i, j] = grid[int(np.floor(i/colsize)), int(np.floor(j/rows))]
            mat[i, j] = g2col(grid[int(np.floor(i/colsize)), int(np.floor(j/rows))])

    if show:
        plt.imshow(mat)
        plt.title("Mat")
        plt.show()


        plt.imshow(lbl)
        plt.title("lbl")
        plt.show()

    return mat, lbl

class Counter:

    i = 0

    def inc_and_get(self):
        self.i+= 1
        return self.i

    def fns(self):
        self.i+= 1
        fn1 = os.path.join("SS_DNA_Train", "Simple1Test", "sxm", "numpy", "Image{}.npy".format(str(self.i).zfill(6)))
        fn2 = os.path.join("SS_DNA_Train", "Simple1Test", "data", "numpy", "Image{}_mask.npy".format(str(self.i).zfill(6)))
        return fn1, fn2, self.i

class SG(Process):

    def __init__(self, ct, nums):

        super().__init__()
        self.name = "P-" + str(np.random.randint(0, 10000))
        self.ct = ct
        self.nums = nums

    def run(self) -> None:
        for j in range(self.nums):
            mat, lbl = simple_grid(100, 3, 3, 3, False)
            fnm, fnl, idx = self.ct.fns()
            try:
                np.save(fnm, mat, allow_pickle=True)
                np.save(fnl, lbl, allow_pickle=True)
                #plt.imsave(fnm[:-3] + ".png", mat)
                #plt.imsave(fnl[:-3] + ".png", lbl)
            except FileNotFoundError:
                os.makedirs("\\".join(fnm.split("\\")[:-1]))
                os.makedirs("\\".join(fnl.split("\\")[:-1]))
                np.save(fnm, mat, allow_pickle=True)
                np.save(fnl, lbl, allow_pickle=True)
            print(self.name, idx)


def parallel_grid():

    imgs = 1000


    thrds = 12
    nums = np.zeros((thrds, ), dtype=int)
    j = 0

    for i in range(imgs):
        nums[j] += 1
        j+= 1
        if j == thrds:
            j = 0

    print(nums)




    BaseManager.register('Counter', Counter)
    filemanager = BaseManager()
    filemanager.start()
    ct = filemanager.Counter()
    t_list = []

    for i in range(thrds):
        t_list.append(SG(ct, nums[i]))
        t_list[-1].start()

    for t in t_list:
        t.join()



if __name__ == "__main__":

    parallel_grid()