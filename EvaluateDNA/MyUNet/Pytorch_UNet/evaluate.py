import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from CustomLoss import MyWeightedLoss, MyF1Loss
from MyUNet.Pytorch_UNet.utils.dice_score import multiclass_dice_coeff, dice_coeff

cuda0 = torch.device('cuda:0')

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_scores = []
    size = []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.cuda(device=cuda0)
        mask_true = mask_true.cuda(device=cuda0)
        size.append(mask_true.shape[0])

        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_scores.append(dice_coeff(mask_pred, mask_true, reduce_batch_first=False))
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_scores.append(multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False))

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return 0
    ret = torch.tensor(sum([dice_scores[k].item() / size[k] for k in range(len(dice_scores))]))
    return ret


def evaluateMWL(net, dataloader, device, crit):
    net.eval()
    wdl = MyWeightedLoss(classes=3, weights=[0.017974, 0.042809, 0.939218])
    criterion = lambda x, y: wdl(F.softmax(x, dim=1).float(),
                                 F.one_hot(y, 3).permute(0, 3, 1, 2).float())

    myfb = MyF1Loss(beta=4, classes=3, weights=[0.017974, 0.042809, 0.939218])

    criterion2 = lambda x, y: myfb(F.softmax(x, dim=1).float(),
                                  F.one_hot(y, 3).permute(0, 3, 1, 2).float())


    num_val_batches = len(dataloader)
    dice_scores = []
    wdl_losses = []
    f1_losses = []
    orig_losses = []
    size = []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.cuda(device=cuda0)
        mask_true = mask_true.cuda(device=cuda0)
        size.append(mask_true.shape[0])


        with torch.no_grad():
            # predict the mask


            mask_pred = net(image)

            wdl_losses.append(criterion(mask_pred, mask_true))
            f1_losses.append(criterion2(mask_pred, mask_true))
            orig_losses.append(crit(mask_pred, mask_true) if crit is not None else torch.tensor(0))

            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_scores.append(dice_coeff(mask_pred, mask_true, reduce_batch_first=False))
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_scores.append(
                    multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False))

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return 0
    ret_Dice = torch.tensor(sum([dice_scores[k].item() / size[k] for k in range(len(dice_scores))]))
    ret_WDL = torch.tensor(sum([wdl_losses[k].item() / size[k] for k in range(len(wdl_losses))]))
    ret_F1 = torch.tensor(np.average([x.item() for x in f1_losses]))
    ret_loss = torch.tensor(np.average([x.item() for x in orig_losses]))

    return ret_WDL, ret_Dice, ret_F1,ret_loss
