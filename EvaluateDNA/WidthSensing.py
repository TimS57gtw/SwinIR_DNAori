import random

import torch
import os
import numpy as np
from torch.utils.data import Dataset
import Preprocessing
from SemanticSegmentation import BATCH_SIZE, IMAGE_SIZE
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from torchinfo import summary
import wandb
import torchvision.transforms as T
from skimage.transform import resize
from torchmetrics import MeanAbsolutePercentageError
val_pairs = []
np.random.seed(1337)

# Plotting und co
losses = []
validations = []
def plot_losses_vals():
    print("Plot Losses")
    if len(losses) == 0:
        print("Len 0")
        return
    
    xs = []
    ys = []
    for pair in losses:
        xs.append(pair[0])
        ys.append(pair[1])
    
    plt.plot(xs, ys)
    plt.title("Loss")
    plt.xlabel('steps')
    plt.ylabel("train loss")
    plt.savefig(LOSS_PLOT_FP)
    print("Saved loss to ", LOSS_PLOT_FP)
    plt.clf()
    if len(validations) <= 1:
        return
    xsv = []
    ysv = []
    for pair in validations:
        xsv.append(pair[0])
        ysv.append(pair[1])
    
    plt.plot(xsv, ysv)
    plt.title("Validation loss")
    plt.xlabel('steps')
    plt.ylabel("val loss")
    plt.savefig(LOSSVAL_PLOT_FP)
    plt.clf()
    plt.plot(xs, ys, label='train')
    plt.plot(xsv, ysv, label='val')
    plt.legend()
    plt.title("loss")
    plt.xlabel('steps')
    plt.ylabel("loss")
    plt.savefig(LOSSBOTH_PLOT_FP)
    plt.clf()



# Pretransforming
def pretransform_fkt(dir_img, dir_img_pret, keep_name=False):
    return Preprocessing.pretransform_all(dir_img, None, None, None, dir_img_pret, None,
                                   None, None, show=False, enhance_contrast=True, zfill=6,
                                   threads=14, img_size=IMAGE_SIZE, overwrite=False, lines_corr=False, do_flatten=False,
                                   do_flatten_border=True, flip=False, use_masks=False, use_Test=False, keep_name=keep_name)

# Data Loading
class MyDataset(Dataset):
    
    def __init__(self, img_folder, pret_img_folder, label_file, is_test=False, pretransform_fkt=None, prefix="Image", zfill=6) -> None:
        super().__init__()
        
        self.img_folder = img_folder
        self.pret_img_folder = pret_img_folder
        self.zfill = zfill
        self.prefix = prefix

        self.transform = T.Compose([T.ToTensor(),
                                    T.RandomHorizontalFlip(),
                                    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        self.is_test = is_test

        self.fn_lbl_pairs = []
        if is_test:
            if label_file is None:
                for img in [os.path.join(img_folder, x) for x in os.listdir(img_folder)]:
                    self.fn_lbl_pairs.append( (img, -1) )
            else:
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.split(",")
                        fn = parts[0].strip().split("\\")[-1]
                        try:
                            self.fn_lbl_pairs.append((fn, float(parts[1].strip())))
                        except IndexError:
                            continue

            if not os.path.isfile(self.get_pret_name(self.fn_lbl_pairs[0][0])):
                print("PIF: ", self.pret_img_folder)
                print(is_test)
                print('Pretransforming Dataset: ', self.get_pret_name(self.fn_lbl_pairs[0][0] ), " not found")
                pretransform_fkt(self.img_folder, pret_img_folder, keep_name=is_test)
            return

        with open(label_file, "r") as f:
            for line in f:
                parts = line.split(",")
                fn = parts[0].strip().split("\\")[-1]
                try:
                    self.fn_lbl_pairs.append( (fn, float(parts[1].strip())))
                except IndexError:
                    continue
        
            
        if not os.path.isfile(self.get_pret_name(self.fn_lbl_pairs[0][0])):
            print('Pretransforming Dataset: ', self.get_pret_name(self.fn_lbl_pairs[0][0]), " not found")
            if label_file is None:
                subdir = ""
            else:
                with open(label_file, "r") as f:
                    for line in f:
                        subdir = os.path.dirname(line.split(",")[0].strip())
                        break

            print("PIF", os.path.join(img_folder, subdir))
            Preprocessing.zfill_img_folder(os.path.join(img_folder, subdir))
            Preprocessing.zfill_labelfile(label_file)
            print(pret_img_folder)
            os.makedirs(pret_img_folder, exist_ok=True)
            pretransform_fkt(os.path.join(img_folder, subdir), pret_img_folder)
            Preprocessing.zfill_img_folder(pret_img_folder)

    def get_names(self):
        names = [os.path.basename(x[0]) for x in self.fn_lbl_pairs]
        return names

    def zfill_fn(self, fn):
        dir = os.path.dirname(fn)
        base = os.path.basename(fn)
        assert base.startswith(self.prefix)
        number = int(base[len(self.prefix):-4])
        return os.path.join(dir, base[:len(self.prefix)] + str(number).zfill(self.zfill) + base[-4:])

    def get_pret_name(self, fn):
        if self.is_test:
            return os.path.join(self.pret_img_folder, os.path.basename(fn).split(".")[0] + ".npy")
        return os.path.join(self.pret_img_folder, self.zfill_fn(fn).split(".")[0] + ".npy")

    def __len__(self):
        return len(self.fn_lbl_pairs)
    
    def __getitem__(self, index):
        
        img, lbl = self.fn_lbl_pairs[index]
        pretimg = self.get_pret_name(img)
        pretimg = np.load(pretimg, allow_pickle=True)
        #print("Before: ", pretimg.shape)
        pretimg = resize(pretimg, (IMAGE_SIZE, IMAGE_SIZE))
        #print("Med: ", pretimg.shape)
        pretimg = self.transform(pretimg)
        #print("After: ", pretimg.shape)
        #pretimg = pretimg[np.newaxis, ...]
        #print("End: ", pretimg.shape)
        sample = {'image': pretimg, 'label': lbl}
        return sample
        

# Model
class MyModel(torch.nn.Module):
    
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 8, 5)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(16, 8, 5)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.linear1 = torch.nn.LazyLinear(1024)
        self.bn6 = torch.nn.BatchNorm1d(1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 128)
        self.linear4 = torch.nn.Linear(128, 1)

        self.relu = torch.nn.ReLU(inplace=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        #print("1: ", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        #print("Before Flatten: ", x.shape)
        x = torch.flatten(x, start_dim=1)
        #print("after Flatten: ", x.shape)
        x = self.linear1(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sig(x)
        x = self.linear4(x)
        x = self.sig(x)
        return x
        

# Training
def perform_training(model, train_loader, val_loader, optimizer, scheduler, grad_scaler, criterion, device, modelfolder=None, loaded_epochs=0):
    model = model.to(device=device)
    print("Device: ",  next(model.parameters()).device )
    global_step = 0
    val_step =  int(len(train_loader) / VALIDATION_PER_EPOCH)
    log_step = int(len(train_loader) / LOGS_PER_EPOCH)
    for epoch in range(1 + loaded_epochs, EPOCHS + 1 + loaded_epochs):
        model.train()

        epoch_loss = 0
        with tqdm(total=len(train_loader)*BATCH_SIZE, desc=f"Epoch {epoch}/{EPOCHS + loaded_epochs}", unit="img") as pbar:
            for sample in train_loader:

                imgs = sample['image'].to(device=device, dtype=torch.float)
                labels = sample['label'].to(device=device, dtype=torch.float)
                
                with torch.cuda.amp.autocast(enabled=AMP):
                    lbl_pred = model(imgs)
                    lbl_pred = lbl_pred[:, 0]
                    loss = criterion(lbl_pred, labels)
                    
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                pbar.update(imgs.shape[0])
                epoch_loss += loss.item()
                losses.append((global_step, loss.item()))
                if global_step % log_step == 0:
                    plot_losses_vals()

                # ToDo: Wandb
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                if USE_WANDB:
                    wandb.log({'step': global_step, 'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

                if global_step % val_step == 0 and (global_step != 0 or VALIDATE_START):
                    print("Validating")
                    model.eval()
                    loss_val = 0
                    for vsample in tqdm(val_loader):
                        vimgs = vsample['image'].to(device=device, dtype=torch.float)
                        vlbls = vsample['label'].to(device=device, dtype=torch.float)
                        target = model(vimgs)
                        target = target[:, 0]
                        vloss = criterion(target, vlbls)
                        loss_val += vloss.item()
                    validations.append((global_step, loss_val/len(val_loader)))
                    if USE_WANDB:
                        wandb.log({'v-loss': loss_val})

                    model.train()

                    plot_losses_vals()

                global_step += 1
                scheduler.step()
        if USE_WANDB:
            wandb.log({'epoch': epoch, 'epoch_loss': epoch_loss})

        if modelfolder is not None:
            torch.save(model.state_dict(), os.path.join(modelfolder, "Model_EP{}.pth".format(epoch)))

        print(lbl_pred)



    return model
                    
                    

        
# Evaluation

def visu_batch(imgs, labels, preds, names, resfolder, save_pct=1):
    os.makedirs(resfolder, exist_ok=True)
    for i in range(len(imgs)):
        if random.random() > save_pct:
            continue
        img = imgs[i].cpu().detach().numpy()[0]
        lbl = labels[i].cpu().detach().numpy()
        pred = preds[i].cpu().detach().numpy()[0]
        name = names[i]
        resname = os.path.join(resfolder, name.split(".")[0] + ".png")

        plt.imshow(img, cmap="gray")
        if lbl > 0:
            plt.title("P: {:.3f}%, L: {:.3f}%".format(100*pred, 100*lbl))
        else:
            plt.title("P: {:.3f}%".format(100*pred))
        plt.savefig(resname)
        plt.clf()




def eval_folder(model, folder, folder_pt, results, test_csv=None, save_pct=0.01, name=None):
    eval = test_csv is not None
    dataset = MyDataset(folder, folder_pt, label_file=test_csv, is_test=True, pretransform_fkt=pretransform_fkt)
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=False, drop_last=False)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)
    model = model.to(device=device)
    model.eval()
    names = dataset.get_names()
    if eval:
        tfp = []
        relative_errors = []
    with tqdm(total=len(test_loader) * BATCH_SIZE, desc=f"Evaluation", unit="img") as pbar:
        for i, sample in enumerate(test_loader):
            imgs = sample['image'].to(device=device, dtype=torch.float)
            labels = sample['label'].to(device=device, dtype=torch.float)
            preds = model(imgs)
            pbar.update(imgs.shape[0])

            if eval:
                for k in range(len(labels)):
                    try:
                        lbl = labels[k].detach().cpu().numpy()
                        prd = preds[k].detach().cpu().numpy() # ToDo Label statt i
                        relative_errors.append(np.abs(lbl - prd) / lbl)
                        tfp.append((lbl, prd))
                    except Exception as e:
                        print(e)
                        continue
            #print("Img: ", imgs.shape)
            #print("Names: ", len(names))
            #print("Names: [", i * loader_args['batch_size'], " : ", min((i+1) * loader_args['batch_size'], len(names)), "] -> Len ", len(names[i * loader_args['batch_size']:min((i+1) * loader_args['batch_size'] - 1, len(names))]))
            visu_batch(imgs, labels, preds, names[i * loader_args['batch_size']:min((i+1) * loader_args['batch_size'], len(names))], results, save_pct=save_pct)

    if eval:
        xs = []
        ys = []
        distances = 0
        tfp = sorted(tfp, key=lambda x : x[0])
        for p in tfp:
            xs.append(p[0])
            ys.append(p[1])
            distances += abs(p[0]-p[1])
        plt.plot(xs, ys)
        plt.xlabel("Label")
        plt.ylabel("Prediction")
        plt.title("Avg Distance: {:.2f}%".format(100*distances[0]/len(tfp)))
        plt.savefig(os.path.join(results, "0_Resultsplt.png"))
        plt.clf()

        elems = []
        for e in relative_errors:
            elems.append(e[0])
        relative_errors = elems
        relative_errors = sorted(relative_errors)
        err_50 = relative_errors[int(len(relative_errors)/2)]
        err_90 = relative_errors[int(0.9*len(relative_errors))]
        err_95 = relative_errors[int(0.95 * len(relative_errors))]

        plt.hist(relative_errors, bins=int(np.sqrt(len(relative_errors))))
        plt.title("E50: {:.1f}%, E90: {:.1f}%, E95: {:.1f}%".format(100*err_50, 100*err_90, 100*err_95))
        plt.ylabel("# of cases")
        plt.xlabel("Relative Error")
        plt.savefig(os.path.join(results, "1_RelativeError.png"))
        plt.savefig(os.path.join(modelfolder, "1_RelativeError{}.png".format(folder.split("\\")[-1] if name is None else name)))
        plt.clf()





if __name__ == "__main__":
        IMAGE_SIZE = 128
        TRAIN_SPLIT = 0.9
        BATCH_SIZE = 32
        WORKERS = 2
        LR = 1e-4
        EPOCHS = 0
        AMP=True
        LOGS_PER_EPOCH = 3
        VALIDATION_PER_EPOCH = 2
        USE_WANDB = True
        VALIDATE_START = True
        if EPOCHS == 0:
            USE_WANDB = False

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')



        currentdatetime = datetime.datetime.now()
        modelfolder = "Model{}{}_{}{}".format(str(currentdatetime.day).zfill(2), str(currentdatetime.month).zfill(2),
                                                str(currentdatetime.hour).zfill(2), str(currentdatetime.minute).zfill(2))


        computer = 'home'
        if computer == 'cluster':
            train_datafolder = "/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train/NoisySet3"
            train_datafolder_pt = "/beegfs/work/seifert/DNA_TrainData/SS_DNA_Train/NoisySet3_pt"
            test_imgs = []
            test_csv = []
            test_pt = []
            res = []
            name = []
            save_pct = []
        elif computer == 'home':
            #train_datafolder = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement\\SS_DNA_Train\\NoisySet2\\Train"
            #train_datafolder_pt = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement\\SS_DNA_Train\\NoisySet2\\Train_pt\\sxm\\numpy"
            #train_datafolder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NoiseMedium\\Train"
            #train_datafolder_pt = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NoiseMedium\\Train_pt"
            train_datafolder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\Rects_Width\\Train"
            train_datafolder_pt = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\Rects_Width\\Train_pt"
            model_base = "D:\\Dateien\\KI_Speicher\\DNA\\Models"

            test_imgs = ["D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\Rects_Width",
                         "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\EvaluateReal\\Ziba2308\\Crops\\Images",
                         "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\EvaluateReal\\ImagesOld\\images"]
            test_csv = ["D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\Rects_Width\\Test.csv",
                        None,
                        None]
            test_pt = ["D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\Rects_Width\\Test_pt",
                       "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\EvaluateReal\\Ziba2308\\PretW",
                       "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\EvaluateReal\\ImagesOld\\PretW"]
            res = ["D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\Rects_Width\\Test_Res",
                   "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\EvaluateReal\\Ziba2308\\ResultsW",
                   "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\EvaluateReal\\ImagesOld\\ResultsW"]
            name = ["Rects", "Ziba2308", "ImagesOld"]
            save_pct = [0.01, 1, 1]


        else:
            train_datafolder = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement\\SS_DNA_Train\\NoisySet2\\Train"
            train_datafolder_pt = "C:\\Users\\seife\\PycharmProjects\\DNA_Measurement\\SS_DNA_Train\\NoisySet2\\Train_pt\\sxm\\numpy"
            test_imgs = []
            test_csv = []
            test_pt = []
            res = []
            name = []
            save_pct = []


        train_csv = os.path.join(train_datafolder, "label_od.csv")
        modelfolder = os.path.join(model_base, modelfolder)
        temp = 2
        while os.path.isdir(modelfolder):
            modelfolder = modelfolder + "_" + str(temp)
            temp += 1
        os.makedirs(modelfolder)
        print("Starting model ", modelfolder)
        LOSS_PLOT_FP = os.path.join(modelfolder, "loss.png")
        LOSSVAL_PLOT_FP = os.path.join(modelfolder, "lossVAL.png")
        LOSSBOTH_PLOT_FP = os.path.join(modelfolder, "lossBOTH.png")

        dataset = MyDataset(train_datafolder, train_datafolder_pt, train_csv, is_test=False, pretransform_fkt=pretransform_fkt)
        print("Dataset: ", len(dataset))
        train_imgs = int(TRAIN_SPLIT * len(dataset))
        val_imgs = len(dataset) - train_imgs
        train_set, val_set = random_split(dataset, [train_imgs, val_imgs], generator=torch.Generator().manual_seed(0))
        print("Train Set: ", len(train_set))
        print("Val Set: ", len(val_set))
        loader_args = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=False, drop_last=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=True, **loader_args)

        model = MyModel()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        expected_steps = EPOCHS * len(train_loader)
        gamma = 0.001**(1/(expected_steps + 1e-4))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        #criterion = torch.nn.MSELoss()
        criterion = MeanAbsolutePercentageError().to(device=device)

        model = model.to(device=device)
        summary(model, input_size=(BATCH_SIZE,1, IMAGE_SIZE, IMAGE_SIZE))


        if USE_WANDB:
            wandb.init(project="WidthSensing2", entity="TimS57gtw", config={'epochs': EPOCHS, 'Model': modelfolder, 'PC': computer, 'device': device,
                                                       'trainImgs': train_csv, 'image size': IMAGE_SIZE, 'batchsize': BATCH_SIZE,
                                                       'learningrate': LR, 'train imgs': len(train_set), 'optim': optimizer,
                                                       'scheduler': scheduler, 'criterion': criterion})

        Preprocessing.zfill_img_folder(train_datafolder_pt)

        if EPOCHS != 0:
            model = perform_training(model=model, train_loader=train_loader, val_loader=val_loader,
                                    optimizer=optimizer, scheduler=scheduler, grad_scaler=grad_scaler,
                                    criterion=criterion, device=device, modelfolder=modelfolder)
            model_path = None
        else:
            model_path = "D:\\Dateien\\KI_Speicher\\DNA\\Models\\Model0109_0836\\Model_EP1.pth"

        # test_imgs = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NoiseMedium\\Test\\"
        # test_csv = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NoiseMedium\\Test.csv"
        # test_pt = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NoiseMedium\\Test_pt"
        # res = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\NoiseMedium\\Test_Res"

        if model_path is not None:
            model = MyModel()
            model.load_state_dict(torch.load(model_path))

        for i in range(len(test_imgs)):
            eval_folder(model, test_imgs[i], test_pt[i], res[i], test_csv=test_csv[i], save_pct=save_pct[i], name=name[i])
