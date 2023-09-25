import os
from tqdm import tqdm
import shutil
import numpy as np

def transform_label(inp, res, image_size, klasse=0):
    lines = []
    with open(inp, "r") as dat:
        for line in dat:
            parts = line.strip().split(",")
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            xmin = float(parts[2].strip())
            xmax = float(parts[3].strip())
            ymin = float(parts[4].strip())
            ymax = float(parts[5].strip())

            xnew = x / image_size
            ynew = y / image_size
            w = (xmax - xmin) / image_size
            h = (ymax - ymin) / image_size

            newl = "{} {} {} {} {}\n".format(klasse, xnew, ynew, w, h)
            lines.append(newl)

    with open(res, "w") as dat:
        for line in lines:
            dat.write(line)


def join_sets(sets, result_train, result_test=None, result_val=None,
              train_split=1.0, val_split=0.0, test_split=0.0, transform_labels=True, img_size=256):

    imgs = []
    lbls = []
    zfil = 6
    prefix = "Image"
    for set in sets:
        lims = [os.path.join(set, "bild", x) for x in os.listdir(os.path.join(set, "bild"))]
        llbs = [os.path.join(set, "data", "BB", x) for x in os.listdir(os.path.join(set, "data", "BB"))]
        print("Set ", set, " len ", len(lims))

        for x in lims:
            imgs.append(x)
        for x in llbs:
            lbls.append(x)

    pairs = []
    for i in range(len(imgs)):
        pairs.append((imgs[i], lbls[i]))

    tims = len(pairs)
    np.random.shuffle(pairs)

    train_imgs = int(tims * train_split)
    val_imgs = int(tims * val_split)
    test_imgs = int(tims * test_split)
    print("TVT", train_imgs, val_imgs, test_imgs)

    if train_imgs !=0:
        os.makedirs(os.path.join(result_train, "images"), exist_ok=True)
        os.makedirs(os.path.join(result_train, "labels"), exist_ok=True)
    if val_imgs != 0:
        os.makedirs(os.path.join(result_val, "images"), exist_ok=True)
        os.makedirs(os.path.join(result_val, "labels"), exist_ok=True)
    if test_imgs != 0:
        os.makedirs(os.path.join(result_test, "images"), exist_ok=True)
        os.makedirs(os.path.join(result_test, "labels"), exist_ok=True)

    for i in tqdm(range(tims)):
        if train_imgs > 0:
            train_imgs -= 1

            shutil.copy(pairs[i][0], os.path.join(result_train, "images", f"{prefix}{str(i).zfill(zfil)}.png"))
            if transform_labels:
                transform_label(pairs[i][1], os.path.join(result_train, "labels", f"{prefix}{str(i).zfill(zfil)}.txt"), img_size)
            else:
                shutil.copy(pairs[i][1], os.path.join(result_train, "labels", f"{prefix}{str(i).zfill(zfil)}.txt"))
        elif val_imgs > 0:
            val_imgs -= 1


            shutil.copy(pairs[i][0], os.path.join(result_val, "images", f"{prefix}{str(i).zfill(zfil)}.png"))
            if transform_labels:
                transform_label(pairs[i][1], os.path.join(result_val, "labels", f"{prefix}{str(i).zfill(zfil)}.txt"),
                                img_size)
            else:
                shutil.copy(pairs[i][1], os.path.join(result_val, "labels", f"{prefix}{str(i).zfill(zfil)}.txt"))

        elif test_imgs > 0:
            test_imgs -= 1


            shutil.copy(pairs[i][0], os.path.join(result_test, "images", f"{prefix}{str(i).zfill(zfil)}.png"))
            if transform_labels:
                transform_label(pairs[i][1], os.path.join(result_test, "labels", f"{prefix}{str(i).zfill(zfil)}.txt"),
                                img_size)
            else:
                shutil.copy(pairs[i][1], os.path.join(result_test, "labels", f"{prefix}{str(i).zfill(zfil)}.txt"))

        else:
            pass


def create_labels(start_folder, image_folder, res_folder, image_size=None, image_folder_val=None, res_val_folder=None):
    assert image_size is not None
    os.makedirs(res_folder, exist_ok=True)
    if res_val_folder is not None:
        os.makedirs(res_val_folder, exist_ok=True)

    labels = [os.path.join(start_folder, x) for x in os.listdir(start_folder)]
    for lbl in tqdm(labels, desc="Transforming labels"):
        if image_folder_val is not None:
            assert res_val_folder is not None
            img = os.path.join(image_folder, os.path.basename(lbl).split(".")[0]+".png")
            if os.path.isfile(img):
                rf = res_folder
            elif os.path.isfile(os.path.join(image_folder_val, os.path.basename(lbl).split(".")[0]+".png")):
                rf = res_val_folder
            else:
                print("Not a file ", img)
                continue
        else:
            rf = res_folder

        result = os.path.join(rf, os.path.basename(lbl))
        transform_label(lbl, result, image_size)



if __name__ == "__main__":
    sets = ["C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA\\Set102"]
    res = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNA2_2k_256p"
    join_sets(sets=sets,
              result_train=os.path.join(res, "train"),
              result_test=os.path.join(res, "test"),
              result_val=os.path.join(res, "val"),
              train_split=0.8, val_split=0.1, test_split=0.1, transform_labels=True, img_size=256)


    sets = ["C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA\\Set79",
            "C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA\\Set71",
            "C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA\\Set69",
            "C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA\\Set102"]
    res = "D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNAMix_4k_256p"
    join_sets(sets=sets,
              result_train=os.path.join(res, "train"),
              result_test=os.path.join(res, "test"),
              result_val=os.path.join(res, "val"),
              train_split=0.8, val_split=0.1, test_split=0.1, transform_labels=True, img_size=256)



    # create_labels(start_folder="C:\\Users\\seife\\PycharmProjects\\STMSim2\\STM_Simulation\\bildordner\\SS_DNA\\Set71\\data\\BB",
    #               image_folder="D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNA_1k_256p\\train\\images",
    #               res_folder="D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNA_1k_256p\\train\\labels",
    #               image_size=256,
    #               #image_folder_val="D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNA_1k_128p\\validation\\images",
    #               #res_val_folder="D:\\Dateien\\KI_Speicher\\DNA_YOLO\\datasets\\ssDNA_1k_128p\\validation\\labels")
    #               )