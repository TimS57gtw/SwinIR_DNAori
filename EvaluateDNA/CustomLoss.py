# Implementation f Focal tversky loss after
# https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
# https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np

epsilon = 1e-5
smooth = 1

def tversky(y_pred, y_true):

    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1-y_pred_pos))
    false_pos = torch.sum((1-y_true_pos)*y_pred_pos)
    #print("\nFP: ", false_pos.item(), "FN: ", false_neg.item())
    alpha = 0.1
    # return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75 # 0.75
    return torch.pow((1-pt_1), gamma)




def my_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += my_dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def my_multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += my_dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def my_dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, classes=2):
    # Dice loss (objective to minimize) between 0 and 1
    ytrue = F.softmax(input, dim=1).float()
    # print(ypred)
    ypred = F.one_hot(target, classes).permute(0, 3, 1, 2).float()
    assert ytrue.size() == ypred.size()
    fn = my_multiclass_dice_coeff if True else my_dice_coeff
    return 1 - fn(ypred, ytrue, reduce_batch_first=True)

class MyWeightedLoss():
    def __init__(self, classes, weights=None, smth=1, eps=1e-8):
        if weights is not None:
            assert classes == len(weights)
        else:
            weights = []
            for i in range(classes):
                weights.append(1/classes)

        self.classes = classes
        self.weights = weights
        self.smth = smth
        self.eps = eps
        self.c0ls = []
        self.c1ls = []
        self.c2ls = []
        self.gamma = 1.2
        self.lbs = np.zeros((3, 3))
        self.lbs[1, 2] = -0.2 * self.weights[1]
        self.lbs[2, 1] = 0.1 * self.weights[1]
        self.lbs[0, 0] = self.weights[0]
        self.lbs[1, 1] = self.weights[1]
        self.lbs[2, 2] = self.weights[2]


    def __call__(self, input: Tensor, target: Tensor):

        # if random.random() < 0.05:
        #     pr = input.detach().cpu().permute(0, 2, 3, 1).numpy()
        #     print(pr.shape)
        #     ta = target.detach().cpu().permute(0, 2, 3, 1).numpy()
        #     fix, axs = plt.subplots(1, 2)
        #     axs[0].imshow(pr[0])
        #     axs[0].set_title("Pred")
        #     axs[1].imshow(ta[0])
        #     axs[1].set_title("target")
        #     plt.show()
        draw = random.random() < 0.0
        total = 0
        loss = 0
        for i in range(self.classes):
            ypredi = input[:, i, ...].flatten()
            cl_ls = 0
            if draw:
                for j in range(0, 32, 7):
                    interesting = input[j, i, ...].detach().cpu().numpy()
                    targetsss = target[j, i, ...].detach().cpu().numpy()
                    fix, axs = plt.subplots(1, 2)
                    axs[0].imshow(interesting)
                    axs[0].set_title("Pred")
                    axs[1].imshow(targetsss)
                    axs[1].set_title("target")
                    plt.title(f"Class {i} w {self.weights[i]}")
                    plt.show()
            for j in range(self.classes):
                if self.lbs[i, j] == 0:
                    continue
                ytruej = target[:, j, ...].flatten()
                cl_ls += self.lbs[i, j] * torch.sum(torch.dot(ypredi, ytruej))
                total += self.lbs[i, j] * torch.sum(torch.dot(ytruej, ytruej))  /input.shape[0] # Shape 0 is BS, not Shape

            if i == 0:
                self.c0ls.append(cl_ls.cpu().detach().numpy() / input.shape[0])
            if i == 1:
                self.c1ls.append(cl_ls.cpu().detach().numpy() / input.shape[0])
            if i == 2:
                self.c2ls.append(cl_ls.cpu().detach().numpy() / input.shape[0])
            loss += cl_ls/input.shape[0]
        # c3 = torch.sum(torch.dot(target[:, 2, ...].flatten(), input[:, 2, ...].flatten()))/16384
        return torch.pow(1 - ((loss + self.eps)/(total + self.eps)), 1/self.gamma)


    def plot_ws(self):
        xs = [x for x in range(len(self.c0ls))]
        # c0s = [x.cpu().detach().numpy() for x in self.c0ls]
        # c1s = [x.cpu().detach().numpy() for x in self.c1ls]
        # c2s = [x.cpu().detach().numpy() for x in self.c2ls]

        c0s = self.c0ls
        c1s = self.c1ls
        c2s = self.c2ls

        plt.plot(xs, c0s)
        plt.title("C0")
        plt.show()
        plt.plot(xs, c1s)
        plt.title("C1")
        plt.show()
        plt.plot(xs, c2s)
        plt.title("C2")
        plt.show()


class WDL:
    def __init__(self, classes, weights=None, smth=1, eps=1e-8):
        if weights is not None:
            assert classes == len(weights)
        else:
            weights = []
            for i in range(classes):
                weights.append(1 / classes)

        self.classes = classes
        self.weights = weights
        self.smth = smth
        self.eps = eps
        self.c0ls = []
        self.c1ls = []
        self.c2ls = []
        self.lbs = np.zeros((classes, classes))

    def __call__(self, input: Tensor, target: Tensor):
        sup = 0
        low = 0

        draw = random.random() < 0.00
        for i in range(self.classes):
            ytrue = target[:, i, ...].flatten()
            ypred = input[:, i, ...].flatten()
            if draw:
                for j in range(0, 32, 10):
                    interesting = input[j, i, ...].detach().cpu().numpy()
                    targetsss = target[j, i, ...].detach().cpu().numpy()
                    fix, axs = plt.subplots(1, 2)
                    axs[0].imshow(interesting)
                    axs[0].set_title("Pred")
                    axs[1].imshow(targetsss)
                    axs[1].set_title("target")
                    plt.title(f"Class {i} w {self.weights[i]}")
                    plt.show()
            sup += self.weights[i] * torch.sum(torch.dot(ypred, ytrue))
            low += self.weights[i] * torch.sum(torch.dot(ypred, ypred)) + torch.sum(torch.dot(ytrue, ytrue))
            if i == 0:
                self.c0ls.append((sup / low).cpu().detach().numpy())
            if i == 1:
                self.c1ls.append((sup / low).cpu().detach().numpy())
            if i == 2:
                self.c2ls.append((sup / low).cpu().detach().numpy())
        # c3 = torch.sum(torch.dot(target[:, 2, ...].flatten(), input[:, 2, ...].flatten()))/16384
        return 1 - 2 * (sup + self.smth) / (low + self.smth + self.eps)

    def plot_ws(self):
        xs = [x for x in range(len(self.c0ls))]
        # c0s = [x.cpu().detach().numpy() for x in self.c0ls]
        # c1s = [x.cpu().detach().numpy() for x in self.c1ls]
        # c2s = [x.cpu().detach().numpy() for x in self.c2ls]

        c0s = self.c0ls
        c1s = self.c1ls
        c2s = self.c2ls

        plt.plot(xs, c0s)
        plt.title("C0")
        plt.show()
        plt.plot(xs, c1s)
        plt.title("C1")
        plt.show()
        plt.plot(xs, c2s)
        plt.title("C2")
        plt.show()


class MyF1Loss:
    def __init__(self, classes, beta=1, weights=None, smth=1, eps=1e-8):
        if weights is not None:
            assert classes == len(weights)
        else:
            weights = []
            for i in range(classes):
                weights.append(1 / classes)

        self.classes = classes
        self.beta2 = beta**2
        self.weights = weights

    def __call__(self, input: Tensor, target: Tensor):
        sup = 0
        low = 0
        loss = 0
        for i in range(self.classes):
            ytrue = target[:, i, ...].flatten()
            ypred = input[:, i, ...].flatten()

            tp = torch.sum(torch.dot(ypred, ytrue))
            fn = torch.sum(torch.dot(torch.ones_like(ypred) - ypred, ytrue))
            fp = torch.sum(torch.dot(ypred, torch.ones_like(ytrue) - ytrue))
            rec = tp / (tp + fn)
            pre = tp / (tp + fp)

            f = (1 + self.beta2) * pre * rec / (self.beta2 * pre + rec)
            loss += self.weights[i] * f


        return 1 - loss