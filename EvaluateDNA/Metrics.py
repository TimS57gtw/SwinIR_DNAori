import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pprint
from PIL import Image
import copy
from EvaluationChain import evaluate_dataset_xy_allargs
PRINT = False
MAX_DETECTIONS = [1000000000]
from ClassOD import class_analyze_folder
import wandb

def evl_lbl(chr):
    if chr.lower().startswith("l"):
        return -1
    elif chr.lower().startswith("r"):
        return 1
    elif int(chr) == 0:
        return -1
    else:
        return int(chr)



def genreate_torch_format(inptdir, fill_conf=1, target=False, as_tensor=True):
    has_conf = True
    file_dict_list = []
    ids = []

    for file in tqdm(os.listdir(inptdir), desc="Generating Torch Format", disable=not PRINT):
        id_extract = lambda x : int(file.split(".")[0][5:]) if file.startswith("Image") else int(file.split(".")[0])
        ids.append(id_extract(file))
        params = []
        for i in range(6):
            params.append([])  # cat, x, y, w, h, conf

        with open(os.path.join(inptdir, file), "r") as f:
            for line in f:
                if "cat" in line:
                    has_conf = "conf" in line
                    continue
                parts = line.strip().split(",")
                for i in range(len(parts)):
                    params[i].append(int(parts[i]) if i == 0 else float(parts[i]))
                if not has_conf:
                    params[-1].append(fill_conf)

        # print(f"params: {params}")
        # print(f"cats: {params[0]}")
        # print(f"xs: {params[1]}")
        # print(f"confs: {params[5]}")
        # exit()

        # generate dict
        num_boxes = len(params[0])

        boxes = np.zeros((num_boxes, 4))
        if not target:
            scores = np.zeros((num_boxes))
        labels = np.zeros((num_boxes), dtype=int)

        for i in range(num_boxes):
            labels[i] = params[0][i]
            boxes[i, 0] = params[1][i]  # x
            boxes[i, 1] = params[2][i]  # y
            boxes[i, 2] = params[3][i] + params[1][i]  # w + x
            boxes[i, 3] = params[4][i] + params[2][i]  # h + y
            if not target:
                scores[i] = params[5][i]

        if not target:
            pairs = []
            for i in range(num_boxes):
                pairs.append((labels[i], boxes[i], scores[i]))
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
            labels = [p[0] for p in pairs]
            boxes = [p[1] for p in pairs]
            scores = [p[2] for p in pairs]

        # if as_tensor:
       # boxes = torch.FloatTensor(boxes)
       # if not target:
       #     scores = torch.FloatTensor(scores)
       # labels = torch.IntTensor(labels)

        img_dict = {"boxes": boxes,
                    "labels": labels}
        if not target:
            img_dict["scores"] = scores

        file_dict_list.append(img_dict)

    return ids, file_dict_list


def filter_existing_pairs(pred_id, pred_list, tar_id, tar_list):
    in_both = [value for value in pred_id if value in tar_id]
    pred_list_fltr = []
    tar_list_fltr = []
    for elem in in_both:
        pred_list_fltr.append(pred_list[pred_id.index(elem)])
        tar_list_fltr.append(tar_list[tar_id.index(elem)])

    if PRINT:
        pprint(f"Used: {in_both}")
        pprint(f"Dropped Pred: {[x for x in pred_id if x not in in_both]}")
        pprint(f"Dropped Target: {[x for x in tar_id if x not in in_both]}")

    return in_both, pred_list_fltr, tar_list_fltr



def visualize_predictions(ids, pred_list, tar_list, imgfld, outf):
    def draw_bbs(image_arr, bbs, scores, resfile):
        image_arr_T = np.array(image_arr).T
        new_image = Image.new('F', (image_arr_T.shape[0], image_arr_T.shape[1]))
        new_image_pixels = new_image.load()
        # new_image_pixels = image_arr_T

        for i in range(image_arr_T.shape[0]):
            for j in range(image_arr_T.shape[1]):
                new_image_pixels[i, j] = 255 * image_arr_T[i, j]

        new_image = new_image.convert('RGB')
        new_image_pixels = new_image.load()

        for i, bb in enumerate(bbs):
            xmin = bb[0]
            ymin = bb[1]
            xmax = bb[2]
            ymax = bb[3]
            fill = int(255 * scores[i])
            try:
                for x in range(int(xmin), int(xmax) + 1):
                    new_image_pixels[x, ymin] = (fill, 0, 0)
                    new_image_pixels[x, ymax] = (fill, 0, 0)
                    new_image_pixels[x, ymin+1] = (fill, 0, 0)
                    new_image_pixels[x, ymax+1] = (fill, 0, 0)
                    new_image_pixels[x, ymin-1] = (fill, 0, 0)
                    new_image_pixels[x, ymax-1] = (fill, 0, 0)
                for y in range(int(ymin), int(ymax) + 1):
                    new_image_pixels[xmin, y] = (fill, 0, 0)
                    new_image_pixels[xmax, y] = (fill, 0, 0)
                    new_image_pixels[xmin+1, y] = (fill, 0, 0)
                    new_image_pixels[xmax+1, y] = (fill, 0, 0)
                    new_image_pixels[xmin-1, y] = (fill, 0, 0)
                    new_image_pixels[xmax-1, y] = (fill, 0, 0)
            except IndexError as e:
                pass

        new_image.save(resfile)


    out_tar = os.path.join(outf, "Vis_Tar")
    os.makedirs(out_tar, exist_ok=True)


    out_det = os.path.join(outf, "Vis_Det")
    os.makedirs(out_det, exist_ok=True)
    for id_id, id in tqdm(enumerate(ids), total=len(ids), desc='Visualization', disable=True):
        pred_bbs = pred_list[id_id]['boxes']
        pred_scores = pred_list[id_id]['scores']
        tar_bbs = tar_list[id_id]['boxes']
        tar_scores = [1 for elem in tar_bbs]

        imgf = os.path.join(imgfld, f'Image{str(id).zfill(4)}.png')
        imarr = Image.open(imgf)
        imarr = np.array(imarr, dtype=float)
        imarr = imarr[:, :, 0]
        imarr -= np.amin(imarr)
        imarr /= np.amax(imarr)

        draw_bbs(copy.deepcopy(imarr), pred_bbs, pred_scores, os.path.join(out_det, f'Image{str(id).zfill(6)}.png'))
        draw_bbs(copy.deepcopy(imarr), tar_bbs, tar_scores, os.path.join(out_tar, f'Image{str(id).zfill(6)}.png'))



def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def _getUnionAreas(boxA, boxB, interArea=None):
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)

def _getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def fn_from_id_lbl_xml(id):
    return str(id) + ".csv"


def id_from_filename(fn):
    id = fn[7:-4]
    return int(id)


def find_bbox_key(file):
    with open(file, "r") as f:
        for line in f:
            if "bbox" in line:
                return "bbox"
            if "bndbox" in line:
                return "bndbox"
    raise Exception("No key found")


def xml2json_cat(id):
    return 0 if id < 0 else 1


def read_xml_preds_vis_pct(folder, resf=None):
    bn = os.path.basename(folder)
    warned = False
    outdir = os.path.join(resf, f"XML_pred_{bn}")
    os.makedirs(outdir, exist_ok=True)

    no_of_files = 0

    idx = 0
    temp_tot = len(os.listdir(folder))
    for fileno, file in tqdm(enumerate(os.listdir(folder)), total=temp_tot, desc="Reading XML-Detections"):
        bbox_key = find_bbox_key(os.path.join(folder, file))

        id = id_from_filename(file)
        outfn = os.path.join(outdir, fn_from_id_lbl_xml(id))
        with open(outfn, "w") as outf:
            outf.write("cat,x,y,w,h,conf\n")

        inpt = ET.parse(os.path.join(folder, file))

        root = inpt.getroot()
        for child in root:
            if child.tag == "object":
                bb = child.find(bbox_key)
                xmin = float(bb.find("xmin").text)
                xmax = float(bb.find("xmax").text)
                ymin = float(bb.find("ymin").text)
                ymax = float(bb.find("ymax").text)

                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin

                label = evl_lbl(child.find("name").text)
                try:
                    conf = float(child.find("score").text)
                except AttributeError as e:
                    if not warned:
                        warned = True
                        print(f"!!! No Score found in {folder}, treat as 1")
                    conf = 1


                with open(outfn, "a") as outf:
                    outf.write(f"{xml2json_cat(label)},{x},{y},{w},{h},{conf}\n")

    return outdir


def read_custom_xyxyxy(folder, resf=None):
    bn = os.path.basename(folder)
    outdir = os.path.join(resf, f"Custom_target_{bn}")
    os.makedirs(outdir, exist_ok=True)
    temp_tot = len(os.listdir(folder))
    for fileno, file in tqdm(enumerate(os.listdir(folder)), total=temp_tot, desc="Reading Custom-targets", disable=True):

        id = id_from_filename(file)- 1
        outfn = os.path.join(outdir, fn_from_id_lbl_xml(id))
        with open(outfn, "w") as outf:
            outf.write("cat,x,y,w,h,conf\n")
        with open(os.path.join(folder, file), 'r') as f:
            for line in f:
                parts = line.split(",")
                xmin = float(parts[2])
                xmax = float(parts[3])
                ymin = float(parts[4])
                ymax = float(parts[5])
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
                label = 0
                conf=1
                with open(outfn, "a") as outf:
                    outf.write(f"{label},{x},{y},{w},{h},{conf}\n")
    return outdir


def read_xml_labels_vis_pct(folder, resf, vis_pct=0):
    bn = os.path.basename(folder)
    outdir = os.path.join(resf, f"XML_lbl_{bn}")
    os.makedirs(outdir, exist_ok=True)
    warned = False

    for file in tqdm(os.listdir(folder), desc="Reading XML-Labels"):
        bbox_key = find_bbox_key(os.path.join(folder, file))
        id = id_from_filename(file)
        outfn = os.path.join(outdir, fn_from_id_lbl_xml(id))
        with open(outfn, "w") as outf:
            outf.write("cat,x,y,w,h,vis_pct\n")

        inpt = ET.parse(os.path.join(folder, file))

        root = inpt.getroot()
        for child in root:
            if child.tag == "object":
                bb = child.find(bbox_key)
                xmin = float(bb.find("xmin").text)
                xmax = float(bb.find("xmax").text)
                ymin = float(bb.find("ymin").text)
                ymax = float(bb.find("ymax").text)

                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin

                label = evl_lbl(child.find("name").text)
                try:
                    loc_vp = float(child.find('visible_percentage').text)
                except AttributeError:
                    if not warned:
                        warned = True
                        print("No Visible percentage found. Assume 1")
                    loc_vp = 1
                if loc_vp < vis_pct:
                    continue

                with open(outfn, "a") as outf:
                    outf.write(f"{xml2json_cat(label)},{x},{y},{w},{h},{loc_vp}\n")

    return outdir


def calculate_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    # Extract coordinates of the boxes

    xA = max(boxA[0], boxB[0])
    xB = min(boxA[2], boxB[2])
    if xA > xB+1:
        return 0
    yA = max(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    if yA > yB+1:
        return 0

    # Calculate intersection area
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate union area
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union_area = boxA_area + boxB_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def eval_metrics_PR_V2(image_names, ids, pred_list, target_list, out_fld, confs=None, ious=None, plot_ious=None,
                       timing_file=None, switch_p_t=False, shuffle=False, disableTQDM=False):
    """

    Args:
        image_names: -
        ids: list of IDs for images
        pred_list: Torch format list of predicted bbs, bbs as xyxy
        target_list: torch format list of target bbs
        out_fld: output folder
        confs: -
        ious: IoUs to Probe. Defaults to np.linspace(0.5, 0.95, 10)
        plot_ious: -

    Returns:
        resuts_npy[i_id, 0] = iou
        resuts_npy[i_id, 1] = ap
        resuts_npy[i_id, 2] = acc_TP[-1]
        resuts_npy[i_id, 3] = acc_FP[-1]
        resuts_npy[i_id, 4] = acc_FN[-1]
        resuts_npy[i_id, 5] = f1[maxi]
        resuts_npy[i_id, 6] = confs[maxi]
    """
    USE_INTEGRATED = True

    # Specify IoU thresholds to evaluate
    if ious is None:
        ious = np.linspace(0.5, 0.95, 10)

    # ious = np.array([0.25, 0.4, 0.5, 0.75])

    lenids = len(ids)
    lenious = len(ious)

    aps = []

    resuts_npy = np.zeros((len(ious), 7))
    resuts_npy_detail = np.zeros((len(ious), 4, 1000))
    if switch_p_t:
        temp = copy.deepcopy(pred_list)
        pred_list = copy.deepcopy(target_list)
        target_list = temp


    # Prepare predictions:
    all_ps = []
    total_gts = 0
    for id_id, id in tqdm(enumerate(ids), disable=True,total=len(ids), desc="Combining Ps", leave=True, position=0):
        total_gts += len(target_list[id_id]['labels'])
        for i in range(len(pred_list[id_id]['labels'])):
            all_ps.append(
                (pred_list[id_id]['labels'][i], pred_list[id_id]['scores'][i] if not switch_p_t else 1, pred_list[id_id]['boxes'][i], id_id))


    all_ps = sorted(all_ps, key=lambda x: x[1], reverse=True)

    if shuffle:
        np.random.shuffle(all_ps)



    if MAX_DETECTIONS[-1] < len(all_ps):
        print(f"Cropping All Detections due to maxDet -> {len(all_ps)} > {MAX_DETECTIONS[-1]}")
        all_ps = all_ps[:MAX_DETECTIONS[-1]]

    os.makedirs(out_fld, exist_ok=True)
    with open(os.path.join(out_fld, "results.csv"), 'w') as f:
        f.write("Iou;mAP;TP;FP;FN;F1;F1conf\n")

    for i_id, iouTH in tqdm(enumerate(ious), disable=True, total=lenious, desc="IoUs..."):

        matched = []
        confs = []

        for id_id in range(len(ids)):
            matched.append([])
        for id_id, id in enumerate(ids):
            for lbl in target_list[id_id]['labels']:
                matched[id_id].append(False)

        TP = np.zeros(len(all_ps), dtype=int)
        FP = np.zeros(len(all_ps), dtype=int)

        for det in tqdm(range(len(all_ps)), desc=f'Finding detections@IoU={iouTH:.2f}', disable=True, leave=True, position=0):
            detect = all_ps[det]
            id_id = detect[3]
            confs.append(detect[1])
            gts = target_list[id_id]
            iouMax = -np.infty
            for j in range(len(gts['labels'])):
                if gts['labels'][j] == detect[0]:
                    iou = calculate_iou(detect[2], gts['boxes'][j])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j

            if iouMax >= iouTH:
                if not matched[id_id][jmax]:
                    TP[det] = 1
                    matched[id_id][jmax] = True
                else:
                    FP[det] = 1
            else:
                FP[det] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        acc_FN = total_gts * np.ones(acc_TP.shape) - acc_TP
        rec = acc_TP / total_gts
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        # Calculate AP

        p_rec = []
        p_rec.append(0)
        [p_rec.append(e) for e in rec]
        p_rec.append(1)
        p_pre = []
        p_pre.append(0)
        [p_pre.append(e) for e in prec]
        p_pre.append(0)

        plt.step(p_rec, p_pre)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        # plt.ylim(-0.02, 1.02)
        plt.title(f"PR_{int(round(100 * iouTH))}")
        plt.savefig(os.path.join(out_fld, f"PR_NINT_{int(round((100 * iouTH)))}.png"))
        plt.clf()


        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1 + i] != mrec[i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        aps.append(ap)

        with open(os.path.join(out_fld, f"IoU_{int(round(100 * iouTH))}.csv"), 'w') as f:
            f.write("Image;Conf;TP;FP;AccTP;AccFP;AccFN;Precision;Recall;BboxX1;BboxY1;BboxX2;BboxY2\n")
            for i in tqdm(range(len(acc_FN)), desc="Writing Results", disable=True):
                f.write(
                    f"{all_ps[i][3]};{all_ps[i][1]};{TP[i]};{FP[i]};{acc_TP[i]};{acc_FP[i]};{acc_FN[i]};{prec[i]};{rec[i]};{all_ps[i][2][0]:.1f};{all_ps[i][2][1]:.1f};{all_ps[i][2][2]:.1f};{all_ps[i][2][3]:.1f}\n")

        # PR Curve
        plt.step(mrec, mpre)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(-0.02, 1.02)
        plt.title(f"mAP_{int(round(100 * iouTH))} = {ap:.4f}")
        plt.savefig(os.path.join(out_fld, f"PR_{int(round((100 * iouTH)))}.png"))
        plt.clf()

        # F1 Score
        f1 = []
        x = []
        for i in range(len(confs) - 1, -1, -1):
            x.append(confs[i])
            if prec[i] + rec[i] == 0:
                f1.append(0)
            else:
                f1.append(2 * prec[i] * rec[i] / (prec[i] + rec[i]))

        plt.plot(x, f1)
        plt.xlabel("Confidence")
        plt.ylabel("F1-Score")
        plt.ylim(-0.02, 1.02)
        maxi = np.argmax(f1)
        plt.title(f"IoU_{int(round(100 * iouTH))} -> F1={f1[maxi]:.4f} @ conf={x[maxi]}")
        plt.savefig(os.path.join(out_fld, f"F1_{int(round((100 * iouTH)))}.png"))
        plt.clf()

        with open(os.path.join(out_fld, "results.csv"), 'a') as f:
            f.write(f"{iouTH};{ap};{acc_TP[-1]};{acc_FP[-1]};{acc_FN[-1]};{f1[maxi]};{confs[maxi]}\n")

        resuts_npy[i_id, 0] = iouTH
        resuts_npy[i_id, 1] = ap
        resuts_npy[i_id, 2] = acc_TP[-1]
        resuts_npy[i_id, 3] = acc_FP[-1]
        resuts_npy[i_id, 4] = acc_FN[-1]
        resuts_npy[i_id, 5] = f1[maxi]
        resuts_npy[i_id, 6] = confs[maxi]


        mf1 = copy.deepcopy(f1)
        mf1.insert(0, 0)
        mf1.append(0)
        mx = copy.deepcopy(x)
        mx.insert(0, 0)
        mx.append(1)

        mxf1 = np.linspace(0, 1, 1000)
        mf1_inter = scipy.interpolate.interp1d(mx, mf1)
        myf1 = mf1_inter(mxf1)

        # plt.plot(mx, mf1, label="Normal")
        # plt.plot(mxf1, myf1, label="Inter")
        # plt.legend()
        # plt.show()



        mxpr = np.linspace(0,1,1000)
        # print("Interpolation Range: ", min(mrec), max(mrec))
        mfpr = scipy.interpolate.interp1d(mrec, mpre, kind='next')
        mypr = mfpr(mxpr)
        mypr[-1] = 0
        # plt.scatter(mrec, mpre, label="Norm")
        # plt.plot(mxpr, mypr, label='inter')
        # plt.legend()
        # plt.show()

        resuts_npy_detail[i_id, 0, :] = mxf1
        resuts_npy_detail[i_id, 1, :] = myf1
        resuts_npy_detail[i_id, 2, :] = mxpr
        resuts_npy_detail[i_id, 3, :] = mypr

    aps_s = []
    ious_s = []
    for i_id, iouTH in tqdm(enumerate(ious), disable=True, total=lenious, desc="IoUs..."):
        plt.plot(resuts_npy_detail[i_id, 2], resuts_npy_detail[i_id, 3], label=f"IoU{int(round(100*resuts_npy[i_id, 0]))}")
        aps_s.append(resuts_npy[i_id, 1])
        ious_s.append(resuts_npy[i_id, 0])

    plt.title(f"mAP50:95 = {np.average(aps_s)}")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(out_fld, f"mAP_50_95.png"))
    plt.clf()

    plt.plot(ious_s, aps_s)
    plt.xlabel("IoU Threshold")
    plt.ylabel("mAP50")
    plt.savefig(os.path.join(out_fld, f"mAP_over_IoU.png"))
    plt.clf()

    with open(os.path.join(out_fld, "mAPs.csv"), 'w') as f:
        f.write(f"mAP50;{aps_s[0]}\n")
        indof75 = np.argwhere(ious == 0.75)[0][0]
        f.write(f"mAP75;{aps_s[indof75]}\n")
        f.write(f"mAP_50_95;{np.average(aps_s)}")

    np.save(os.path.join(out_fld, 'results.npy'), resuts_npy, allow_pickle=True)
    np.save(os.path.join(out_fld, 'results_detail.npy'), resuts_npy_detail, allow_pickle=True)

    return resuts_npy

def combine_runs(folders, out_fld):
    num_res = len(folders)
    results = None
    ious = None
    aps = None

    allres = np.zeros((len(folders), 6))
    for k, folder in enumerate(folders):
        fn = os.path.join(folder, "results_detail.npy")
        res = np.load(fn, allow_pickle=True)
        if results is None:
            results = res
        else:
            results += res

        res_nd = np.load(os.path.join(folder, "results.npy"), allow_pickle=True)
        if ious is None:
            ious = res_nd[:, 0]
        if aps is None:
            aps = res_nd[:, 1]
        else:
            aps += res_nd[:, 1]

        indof75 = np.argwhere(res_nd[:, 0] == 0.75)
        allres[k, 0] = res_nd[0, 1]
        allres[k, 1] = res_nd[indof75, 1]
        allres[k, 2] = np.average(res_nd[:, 1])
        allres[k, 3] = res_nd[0, 5]
        allres[k, 4] = 1 if os.path.basename(folder).startswith("True") else 0
        allres[k, 5] = int(folder.split("_")[-1])


    with open(os.path.join(out_fld, "run_stats.csv"), 'w') as f:
        f.write("flip;shuffle;mAP50;mAP75;mAP50_95;f1_50\n")
        for i in range(len(folders)):
            f.write(f"{allres[i, 4] == 1};{allres[i, 5]};{allres[i, 0]};{allres[i, 1]};{allres[i, 2]};{allres[i, 3]}\n")

    results /= num_res
    aps /= num_res
    with open(os.path.join(out_fld, "results_combine.csv"), 'w') as f:
        f.write("iouTH;AP;f1\n")
        for i_id, iouTH in enumerate(ious):
            plt.plot(results[i_id, 0], results[i_id, 1])
            plt.xlabel("Confidence")
            plt.ylabel("F1-Score")
            plt.ylim(-0.02, 1.02)
            maxi = np.argmax(results[i_id, 1,:])
            plt.title(f"IoU_{int(round(100 * iouTH))} -> F1={results[i_id, 1,maxi]:.4f} @ conf={results[i_id, 0,maxi]}")
            plt.savefig(os.path.join(out_fld, f"F1_{int(round((100 * iouTH)))}.png"))
            plt.clf()
            plt.step(results[i_id, 2], results[i_id, 3])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.ylim(-0.02, 1.02)
            plt.title(f"mAP_{int(round(100 * iouTH))} = {aps[i_id]:.4f}")
            plt.savefig(os.path.join(out_fld, f"PR_{int(round((100 * iouTH)))}.png"))
            plt.clf()
            f.write(f"{ious[i_id]};{aps[i_id]};{results[i_id, 1,maxi]}\n")

    aps_s = []
    ious_s = []
    for i_id, iouTH in enumerate(ious):
        plt.plot(results[i_id, 2], results[i_id, 3], label=f"IoU{int(round(100*res_nd[i_id, 0]))}")
        aps_s.append(res_nd[i_id, 1])
        ious_s.append(res_nd[i_id, 0])

    plt.title(f"mAP50:95 = {np.average(aps_s)}")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(out_fld, f"mAP_50_95.png"))
    plt.clf()


    with open(os.path.join(out_fld, "mAPs.csv"), 'w') as f:
        f.write(f"mAP50;{aps[0]}\n")
        indof75 = np.argwhere(ious == 0.75)[0][0]
        f.write(f"mAP75;{aps[indof75]}\n")
        f.write(f"mAP_50_95;{np.average(aps)}")



def eval_XML_visPct(pred_fld, tar_xml_fld, output_fld, vis_pct, confs=None, ious=None, images=None):
    """

    Args:
        pred_fld: Folder containing predictions as JSON
        tar_xml_fld: Folder containing GT as XML
        output_fld: Folder for results
        vis_pct: min visible_percentage
        confs: Confidences to probe. Defaults to [0.5, 0.6, 0.7, 0.8, 0.9]
        ious: IoUs to Probe. Defaults to np.linspace(0.5, 0.95, 10)
        plot_ious: List of IoU vaules for which PR curves should be plotted

    Returns:
    dict:
        'ImageXXX':     dict:
                            'PR':       np.Array [Iou_index, Confidence_index, 5]
                                                -> Precision, Recall, TP, FP, FN
                            'map':      list of mAPs for each IoU
                            'map_50_95':mAP50_95 value
        'Total:     Dict as for each image, just averaged
    """

    output = os.path.join(output_fld,
                          f"{os.path.basename(pred_fld).split('.')[0]}_{os.path.basename(tar_xml_fld).split('.')[0]}")
    os.makedirs(output, exist_ok=True)

    tar_csv = os.path.join(output, "csv_format")
    os.makedirs(tar_csv, exist_ok=True)

    tdir = read_xml_labels_vis_pct(tar_xml_fld, tar_csv, vis_pct=vis_pct)
    pdir = read_xml_preds_vis_pct(pred_fld, tar_csv)  # Inc ID, because starts with 0

    pred_ids, json_pred_list = genreate_torch_format(pdir, target=False, as_tensor=False)
    tar_ids, json_target_list = genreate_torch_format(tdir, target=True, as_tensor=False)

    ids, pred_list, tar_list = filter_existing_pairs(pred_ids, json_pred_list, tar_ids, json_target_list)
    if images is not None:
        visualize_predictions(ids, pred_list, tar_list, images, output_fld)

    image_names = []
    for id in ids:
        for elem in os.listdir(tar_xml_fld):
            if int(elem[7:-4]) == id:
                image_names.append(elem.split('.')[0])
                continue
    with open("timing.csv", 'w') as timingf:
        eval_metrics_PR_V2(image_names, ids, pred_list, tar_list, output_fld, confs=confs, ious=ious, timing_file=timingf)
        # eval_metrics_torch(ids, pred_list, tar_list, out_fld=output_fld)


def read_yolo_preds(images, pred_fld, outfld, conf_th=None):
    assert len(os.listdir(images)) == len(os.listdir(pred_fld)), f"Lengths of Images and Preds do not match, {len(os.listdir(images))} != {len(os.listdir(pred_fld))}, {images}, {pred_fld}"
    out_dir = os.path.join(outfld, "YOLO_Preds")
    os.makedirs(out_dir, exist_ok=True)
    imfs = [os.path.join(images, x) for x in os.listdir(images)]
    prefs = [os.path.join(pred_fld, x) for x in os.listdir(pred_fld)]

    for i in range(len(imfs)):
        img = Image.open(imfs[i])
        outfn = os.path.join(out_dir, f"{os.path.basename(imfs[i]).split('.')[0]}.csv")
        iw = img.size[0]
        ih = img.size[1]

        with open(outfn, "w") as outf:
            outf.write("cat,x,y,w,h,conf\n")

            with open(prefs[i], 'r') as f:
                for line in f:
                    parts = line.split(" ")
                    cls = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    wr = float(parts[3])
                    hr = float(parts[4])
                    cnf = float(parts[5])

                    if conf_th is not None and cnf < conf_th:
                        continue

                    xci = xc * iw
                    yci = yc * ih
                    w = wr * iw
                    h = hr * ih
                    xmin = xci - (w/2)
                    xmax = xci + (w/2)
                    ymin = yci - (h/2)
                    ymax = yci + (h/2)
                    outf.write(f"{cls},{xmin},{ymin},{w},{h},{cnf}\n")

    return out_dir

def shorten_synth_set(infld, outfld, ims=5):
    for dust in [0, 2, 5]:
        for noise in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            spms = os.path.join(infld, f"Dust{dust}", f"Noise_{noise}", "sxm")
            if os.path.isdir(os.path.join(spms, "files")):
                continue
            os.makedirs(os.path.join(spms, "files"))
            for elem in os.listdir(spms):
                if "." in elem:
                    shutil.move(os.path.join(spms, elem), os.path.join(spms, "files", elem))

    for dust in [0, 2, 5]:
        for noise in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            spms = os.path.join(infld, f"Dust{dust}", f"Noise_{noise}", "sxm", "files")
            out = os.path.join(outfld, f"Dust{dust}", f"Noise_{noise}", "sxm", "files")
            data_fld = os.path.join(infld, f"Dust{dust}", f"Noise_{noise}", "data", "BB")
            data_out = os.path.join(outfld, f"Dust{dust}", f"Noise_{noise}", "data", "BB")
            os.makedirs(data_out, exist_ok=True)

            os.makedirs(out, exist_ok=True)
            for elem in os.listdir(spms):
                if len(os.listdir(out)) >= ims:
                    break
                if "." in elem:
                    shutil.copy(os.path.join(spms, elem), os.path.join(out, elem))
                    shutil.copy(os.path.join(data_fld, elem.split(".")[0] + ".txt"), os.path.join(data_out, elem.split(".")[0] + ".txt"))



def eval_synth(infld, outfld):
    wandb.init(mode="disabled")

    spms = os.path.join(infld, "sxm")
    if not os.path.isdir(os.path.join(spms, "files")):
        os.makedirs(os.path.join(spms, "files"))
        for elem in os.listdir(spms):
            if "." in elem:
                shutil.move(os.path.join(spms, elem), os.path.join(spms, "files", elem))


    eval_fld = os.path.join(outfld, "Evaluations")
    eval_fld_ai = os.path.join(eval_fld, "AI")
    eval_fld_ai70 = os.path.join(eval_fld, "AI_CNF70")
    eval_fld_cls = os.path.join(eval_fld, "cls")

    eval_scenario_ai = os.path.join(eval_fld_ai )
    eval_scenario_ai70 = os.path.join(eval_fld_ai70)
    eval_scenario_cls = os.path.join(eval_fld_cls)
    spms = os.path.join(infld, "sxm")

    if not os.path.isdir(eval_scenario_cls):
        class_analyze_folder(spms, resultf=eval_scenario_cls)

    if not os.path.isdir(eval_scenario_ai):
        os.makedirs(eval_scenario_ai)

        params = {"folder": spms,
                  "resfolder": eval_scenario_ai,
                  "quality_mode": "FIT",
                  "threads": 20,
                  "labels": None,
                  "use_UNet": True,
                  "yolo_conf_thrsh": 0.05
                  }
        evaluate_dataset_xy_allargs(**params)
    if not os.path.isdir(eval_scenario_ai70):
        params = {"folder": spms,
                  "resfolder": eval_scenario_ai70,
                  "quality_mode": "FIT",
                  "threads": 20,
                  "labels": None,
                  "use_UNet": True,
                  "yolo_conf_thrsh": 0.70
                  }
        evaluate_dataset_xy_allargs(**params)

    pred_class = os.path.join(eval_scenario_cls, "boxes", "csv")
    pred_ai = os.path.join(eval_scenario_ai, "yolo_prediction", "labels")
    images = os.path.join(eval_scenario_ai, "pp_sample", "images")
    target_real = os.path.join(infld, "data", "BB")

    out_scenario = os.path.join(outfld)
    out_ai = os.path.join(out_scenario, "AI")
    out_target = os.path.join(out_scenario, "target")
    os.makedirs(out_target, exist_ok=True)
    out_ai_viz = os.path.join(out_ai, "viz")
    out_class = os.path.join(out_scenario, "class")
    out_class_viz = os.path.join(out_class, "viz")
    os.makedirs(out_ai_viz, exist_ok=True)
    os.makedirs(out_class_viz, exist_ok=True)

    pdir_class = read_yolo_preds(images, pred_class, outfld=out_class)
    pdir_ai = read_yolo_preds(images, pred_ai, outfld=out_ai)
    tdir = read_custom_xyxyxy(target_real, resf=out_target)

    pred_ids_class, json_pred_list_class = genreate_torch_format(pdir_class, target=False, as_tensor=False)
    pred_ids_ai, json_pred_list_ai = genreate_torch_format(pdir_ai, target=False, as_tensor=False)
    tar_ids, json_target_list = genreate_torch_format(tdir, target=True, as_tensor=False)

    ids_ai, pred_list_ai, tar_list_ai = filter_existing_pairs(pred_ids_ai, json_pred_list_ai, tar_ids,
                                                                          json_target_list)
    ids_cls, pred_list_cls, tar_list_cls = filter_existing_pairs(pred_ids_class, json_pred_list_class,
                                                                             tar_ids, json_target_list)
    if images is not None:
        visualize_predictions(ids_ai, pred_list_ai, tar_list_ai, images, out_ai_viz)
        visualize_predictions(ids_cls, pred_list_cls, tar_list_cls, images, out_class_viz)

    eval_metrics_PR_V2(None, ids_ai, pred_list_ai, tar_list_ai, out_ai, timing_file=None)
                # eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class, timing_file=None, switch_p_t=False,
                #                    shuffle=False)

    folders = []
    for switch in [True, False]:
        for i in range(10):
            out_class_temp = os.path.join(out_class, "subruns", f"{switch}_{i}")
            folders.append(out_class_temp)
            eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class_temp,
                               disableTQDM=True, timing_file=None, switch_p_t=switch, shuffle=True)

    combine_runs(folders, out_class)

def eval_synth_noise_parts(infld, outfld):
    wandb. init(mode="disabled")
    for dust in [4]: #
        for noise in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]: #         for noise in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            spms = os.path.join(infld, f"Dust{dust}", f"Noise_{noise}", "sxm")
            if os.path.isdir(os.path.join(spms, "files")):
                continue
            os.makedirs(os.path.join(spms, "files"))
            for elem in os.listdir(spms):
                if "." in elem:
                    shutil.move(os.path.join(spms, elem), os.path.join(spms, "files", elem))
    dusts = [4]
    noises = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    with tqdm(total=len(noises) * len(dusts)) as pbar:
        for dust in dusts:
            for noise in noises:
                pbar.set_description(f"---> Dust: {dust}, Noise: {noise}")
                # print(f"\n\n\nStarting Dust {dust}, Noise {noise}\n\n\n")
                # Evaluations
                eval_fld = os.path.join(outfld, "Evaluations")
                eval_fld_ai = os.path.join(eval_fld, "AI")
                eval_fld_ai70 = os.path.join(eval_fld, "AI_CNF70")
                eval_fld_cls = os.path.join(eval_fld, "cls")

                eval_scenario_ai = os.path.join(eval_fld_ai, f"Dust_{dust}", f"Noise_{noise}")
                eval_scenario_ai70 = os.path.join(eval_fld_ai70, f"Dust_{dust}", f"Noise_{noise}")
                eval_scenario_cls = os.path.join(eval_fld_cls, f"Dust_{dust}", f"Noise_{noise}")
                spms = os.path.join(infld, f"Dust{dust}", f"Noise_{noise}", "sxm")

                if not os.path.isdir(eval_scenario_cls):
                    class_analyze_folder(spms, resultf=eval_scenario_cls)

                if not os.path.isdir(eval_scenario_ai):
                    os.makedirs(eval_scenario_ai)


                    params = {"folder": spms,
                              "resfolder": eval_scenario_ai,
                              "quality_mode": "FIT",
                              "threads": 20,
                              "labels": None,
                              "use_UNet": True,
                              "yolo_conf_thrsh": 0.05
                              }
                    evaluate_dataset_xy_allargs(**params)
                if not os.path.isdir(eval_scenario_ai70):

                    params = {"folder": spms,
                              "resfolder": eval_scenario_ai70,
                              "quality_mode": "FIT",
                              "threads": 20,
                              "labels": None,
                              "use_UNet": True,
                              "yolo_conf_thrsh": 0.70
                              }
                    evaluate_dataset_xy_allargs(**params)


                pred_class = os.path.join(eval_scenario_cls, "boxes", "csv")
                pred_ai = os.path.join(eval_scenario_ai, "yolo_prediction", "labels")
                images = os.path.join(eval_scenario_ai, "pp_sample", "images")
                target_real = os.path.join(infld, f"Dust{dust}",f"Noise_{noise}", "data", "BB")

                out_scenario = os.path.join(outfld, f"Dust_{dust}", f"Noise_{noise}")
                out_ai = os.path.join(out_scenario, "AI")
                out_target = os.path.join(out_scenario, "target")
                os.makedirs(out_target, exist_ok=True)
                out_ai_viz = os.path.join(out_ai, "viz")
                out_class = os.path.join(out_scenario, "class")
                out_class_viz = os.path.join(out_class, "viz")
                os.makedirs(out_ai_viz, exist_ok=True)
                os.makedirs(out_class_viz, exist_ok=True)

                pdir_class = read_yolo_preds(images, pred_class, outfld=out_class)
                pdir_ai = read_yolo_preds(images, pred_ai, outfld=out_ai)
                tdir = read_custom_xyxyxy(target_real, resf=out_target)

                pred_ids_class, json_pred_list_class = genreate_torch_format(pdir_class, target=False, as_tensor=False)
                pred_ids_ai, json_pred_list_ai = genreate_torch_format(pdir_ai, target=False, as_tensor=False)
                tar_ids, json_target_list = genreate_torch_format(tdir, target=True, as_tensor=False)

                ids_ai, pred_list_ai, tar_list_ai = filter_existing_pairs(pred_ids_ai, json_pred_list_ai, tar_ids,
                                                                              json_target_list)
                ids_cls, pred_list_cls, tar_list_cls = filter_existing_pairs(pred_ids_class, json_pred_list_class,
                                                                                 tar_ids, json_target_list)
                if images is not None:
                    visualize_predictions(ids_ai, pred_list_ai, tar_list_ai, images, out_ai_viz)
                    visualize_predictions(ids_cls, pred_list_cls, tar_list_cls, images, out_class_viz)

                eval_metrics_PR_V2(None, ids_ai, pred_list_ai, tar_list_ai, out_ai, timing_file=None)
                    # eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class, timing_file=None, switch_p_t=False,
                    #                    shuffle=False)

                folders = []
                for switch in [True, False]:
                    for i in range(10):
                        out_class_temp = os.path.join(out_class, "subruns", f"{switch}_{i}")
                        folders.append(out_class_temp)
                        eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class_temp,
                                           disableTQDM=True, timing_file=None, switch_p_t=switch, shuffle=True)

                combine_runs(folders, out_class)
                pbar.update(1)

def eval_real(infld, outfld, yolo_model=None):
    wandb. init(mode="disabled")
    eval_fld = os.path.join(outfld, "Evaluations")
    eval_fld_ai = os.path.join(eval_fld, "AI")
    eval_fld_ai70 = os.path.join(eval_fld, "AI_CNF70")
    eval_fld_cls = os.path.join(eval_fld, "cls")

    eval_scenario_ai = os.path.join(eval_fld_ai, "Res")
    eval_scenario_ai70 = os.path.join(eval_fld_ai70, "Res")
    eval_scenario_cls = os.path.join(eval_fld_cls, "Res")
    spms = os.path.join(infld, 'SPMrn')

    if os.path.isdir(os.path.join(os.path.dirname(outfld), 'cls')):
        shutil.copytree(os.path.join(os.path.dirname(outfld), 'cls'), eval_fld_cls)
    else:
        print("Is not file", os.path.join(os.path.dirname(outfld), 'cls'))

    if not os.path.isdir(eval_scenario_cls):
        class_analyze_folder(spms, resultf=eval_scenario_cls, abort_od=True)
    else:
        print("CLS found")

    if not os.path.isdir(eval_scenario_ai):
        os.makedirs(eval_scenario_ai)


        params = {"folder": spms,
                  "resfolder": eval_scenario_ai,
                  "quality_mode": "FIT",
                  "threads": 20,
                  "labels": None,
                  "use_UNet": True,
                  "yolo_conf_thrsh": 0.05,
                  "abort_od": True
                  }
        if yolo_model is not None:
            params["yolo_model"] = yolo_model
        evaluate_dataset_xy_allargs(**params)
    if not os.path.isdir(eval_scenario_ai70):

        params = {"folder": spms,
                  "resfolder": eval_scenario_ai70,
                  "quality_mode": "FIT",
                  "threads": 20,
                  "labels": None,
                  "use_UNet": True,
                  "yolo_conf_thrsh": 0.70,
                  "abort_od": True

                  }
        if yolo_model is not None:
            params["yolo_model"] = yolo_model
        evaluate_dataset_xy_allargs(**params)


    pred_class = os.path.join(eval_scenario_cls, "boxes", "csv")
    pred_ai = os.path.join(eval_scenario_ai, "yolo_prediction", "labels")
    images = os.path.join(eval_scenario_ai, "pp_sample", "images")
    target_real = os.path.join(infld, "Labels")

    out_scenario = os.path.join(outfld, "Res")
    out_ai = os.path.join(out_scenario, "AI")
    out_target = os.path.join(out_scenario, "target")
    os.makedirs(out_target, exist_ok=True)
    out_ai_viz = os.path.join(out_ai, "viz")
    out_class = os.path.join(out_scenario, "class")
    out_class_viz = os.path.join(out_class, "viz")
    os.makedirs(out_ai_viz, exist_ok=True)
    os.makedirs(out_class_viz, exist_ok=True)

    pdir_class = read_yolo_preds(images, pred_class, outfld=out_class)
    pdir_ai = read_yolo_preds(images, pred_ai, outfld=out_ai)
    tdir = read_xml_labels_vis_pct(target_real, resf=out_target)

    pred_ids_class, json_pred_list_class = genreate_torch_format(pdir_class, target=False, as_tensor=False)
    pred_ids_ai, json_pred_list_ai = genreate_torch_format(pdir_ai, target=False, as_tensor=False)
    tar_ids, json_target_list = genreate_torch_format(tdir, target=True, as_tensor=False)

    ids_ai, pred_list_ai, tar_list_ai = filter_existing_pairs(pred_ids_ai, json_pred_list_ai, tar_ids,
                                                              json_target_list)
    ids_cls, pred_list_cls, tar_list_cls = filter_existing_pairs(pred_ids_class, json_pred_list_class,
                                                                 tar_ids, json_target_list)
    if images is not None:
        visualize_predictions(ids_ai, pred_list_ai, tar_list_ai, images, out_ai_viz)
        visualize_predictions(ids_cls, pred_list_cls, tar_list_cls, images, out_class_viz)

    eval_metrics_PR_V2(None, ids_ai, pred_list_ai, tar_list_ai, out_ai, timing_file=None)
    # eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class, timing_file=None, switch_p_t=False,
    #                    shuffle=False)

    folders = []
    for switch in [True, False]:
        for i in range(10):
            out_class_temp = os.path.join(out_class, "subruns", f"{switch}_{i}")
            folders.append(out_class_temp)
            eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class_temp,
                               disableTQDM=True, timing_file=None, switch_p_t=switch, shuffle=True)

    combine_runs(folders, out_class)

def xml2yolo(inimg, inlbl, outdir, tempdir):
    read_xml_labels_vis_pct(inlbl, tempdir)

    tempdir = os.path.join(tempdir, "XML_lbl_Labels")

    ims = [os.path.join(inimg, x) for x in os.listdir(inimg)]



    for i in range(len(ims)):
        im = Image.open(ims[i])
        shp = im.size
        print(shp)
        imw = shp[0]
        imh = shp[1]
        lbs = os.path.join(tempdir, f"{i+1}.csv")
        print(f"Matching {os.path.basename(ims[i])}, {os.path.basename(lbs)}")
        resname = os.path.join(outdir, os.path.basename(ims[i]).split(".")[0] + ".txt")
        fst = True
        with open(resname, 'w') as f:
            if not os.path.isfile(lbs):
                f.write("\n")
                continue

            with open(lbs, 'r') as l:
                for line in l:
                    if fst:
                        fst = False
                        continue
                    parts = line.split(",")
                    xl = float(parts[1])
                    yt = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    xc = xl + w/2
                    yc = yt + h/2
                    xc /= imw
                    yc /= imh
                    w /= imw
                    h /= imh
                    f.write(f"{parts[0]} {xc} {yc} {w} {h}\n")


if __name__ == "__main__":
    # inimg = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\INDIV\Images'
    # inlbl = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\INDIV\Labels'
    # outdir = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\INDIV\YoloLbls'
    # tempdir = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\INDIV\CustomLbl'
    # xml2yolo(inimg, inlbl, outdir, tempdir)


    infld = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\DS\INDIV'
    num = 0#len(os.listdir(r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Res\Real\INDIV'))-1
    modelf = None# r'C:\Users\seifert\PycharmProjects\DNA_Measurement\YOLO\ssDNA_Mix_4k_256p_pt2\ssDNA_Mix_4k_256p_pt27Autoanch'
    name = "old"# os.path.basename(modelf)
    ym = None # os.path.join(modelf, "weights", "best.pt")
    outfld = os.path.join('D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Res\Real\INDIV', name)
    eval_real(infld, outfld, yolo_model=ym)
    assert 1 == 2


    # infld = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Synth\Set8'
    # outfld = r'D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Synth\Res3'
    # eval_synth(infld, outfld)
    # assert 5 == 6
#
    # infld = r"D:\seifert\PycharmProjects\DNAmeasurement\datasets\NoiseDep_Conn_40"
    # med_fld = r"D:\seifert\PycharmProjects\DNAmeasurement\datasets\NoiseDep_Conn_5"
    # outfld = r"D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Synth\NoiseDep_Conn_40"
#
    # # shorten_synth_set(infld, med_fld)
    # eval_synth_noise_parts(infld, outfld)
    # assert 1 == -1
#
    dataset = 'NoBirkaNamelessCNF70'
    target_real = r"D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Real"

    pred_class = r"D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Real"
    pred_ai = r"D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Real"
    images = "D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Real"

    pred_class = os.path.join(pred_class, dataset, "Class_Pred")
    pred_ai = os.path.join(pred_ai, dataset, "AI_Pred")
    images = os.path.join(images, dataset, "Images")
    target_real = os.path.join(target_real, dataset, "Labels")

    outfld = os.path.join(r"D:\seifert\PycharmProjects\DNAmeasurement\Metrics\Real", dataset, "Results")
    out_ai = os.path.join(outfld, "AI")
    out_target = os.path.join(outfld, "target")
    os.makedirs(out_target,exist_ok=True)
    out_ai_viz = os.path.join(out_ai, "viz")
    out_class = os.path.join(outfld, "class")
    out_class_viz = os.path.join(out_class, "viz")
    os.makedirs(out_ai_viz, exist_ok=True)
    os.makedirs(out_class_viz, exist_ok=True)




    pdir_class = read_yolo_preds(images, pred_class, outfld=out_class)
    pdir_ai = read_yolo_preds(images, pred_ai,outfld=out_ai, conf_th=0.7)
    tdir = read_xml_preds_vis_pct(target_real, resf=out_target)

    pred_ids_class, json_pred_list_class = genreate_torch_format(pdir_class, target=False, as_tensor=False)
    pred_ids_ai, json_pred_list_ai = genreate_torch_format(pdir_ai, target=False, as_tensor=False)
    tar_ids, json_target_list = genreate_torch_format(tdir, target=True, as_tensor=False)

    ids_ai, pred_list_ai, tar_list_ai = filter_existing_pairs(pred_ids_ai, json_pred_list_ai, tar_ids, json_target_list)
    ids_cls, pred_list_cls, tar_list_cls = filter_existing_pairs(pred_ids_class, json_pred_list_class, tar_ids, json_target_list)
    if images is not None:
        visualize_predictions(ids_ai, pred_list_ai, tar_list_ai, images, out_ai_viz)
        visualize_predictions(ids_cls, pred_list_cls, tar_list_cls, images, out_class_viz)

    eval_metrics_PR_V2(None, ids_ai, pred_list_ai, tar_list_ai, out_ai, timing_file=None)
    # eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class, timing_file=None, switch_p_t=False,
    #                    shuffle=False)

    folders = []
    with tqdm(total=20) as pbar:
        pbar.set_description("Switch + Shuffle")
        for switch in [True, False]:
            for i in range(10):
                out_class_temp = os.path.join(out_class, "subruns", f"{switch}_{i}")
                folders.append(out_class_temp)
                eval_metrics_PR_V2(None, ids_cls, pred_list_cls, tar_list_cls, out_class_temp, disableTQDM=True, timing_file=None, switch_p_t=switch, shuffle=True)
                pbar.update(1)

    combine_runs(folders, out_class)
