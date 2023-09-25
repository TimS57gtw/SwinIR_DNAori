from My_SXM import My_SXM
import SXM_info
from PIL import Image
from tqdm import tqdm

import os
import struct
import matplotlib.pyplot as plt
import numpy as np
#import pySPM


class SPM:
    def __init__(self, filename):
        self.filename = filename
        self.f = open(self.filename, 'rb')

        self.header = {}
        self.data_offset = None
        i = 0
        while True:
            try:
                l = self.f.readline().rstrip()
                if l.startswith(bytes("\\", "UTF-8")):
                    if b':' in l:
                        key = l.split(b':')[0].decode('UTF-8')
                        self.header[key] = l.split(b':')[1].decode('UTF-8')
                if l.startswith(bytes("\\Data offset", "UTF-8")) and self.data_offset is None:
                    self.data_offset = int(self.header["\\Data offset"])
                if l.startswith(bytes("\\*File list end", "UTF-8")):
                    break
                if i > 1e6:
                    raise EOFError("Probably wrong filetype, maybe errorbar set to low")
                i += 1
            except Exception as exc:
                print("Exception: ", exc)
                print("Invalid file ", self.filename)
                return
        scan_size = self.header['\\Scan Size']
        scan_size = scan_size.split()
        self.size = dict(pixels={
            'x': int(self.header['\\Valid data len X']),
            'y': int(self.header['\\Valid data len Y'])
        }, real={
            'x': float(scan_size[0]),
            'y': float(scan_size[1])*int(self.header['\\Valid data len Y'])/int(self.header['\\Valid data len X']),
            'unit': scan_size[2]
        })

    def get_data(self, chID=0, corr_lines=False, byteorder='<', formatcharacter='i'):
        # chID = 0
        # byteorder = '<'  # byteorder: '@', '=', '<' work; !, > don't work https://docs.python.org/3/library/struct.html
        # formatcharacter = 'h'  # formatcharacter: i: int, f: float, l: long, d: double, h: short(int16) (uppercase for unsigned version)

        size = self.size['pixels']['x'] * self.size['pixels']['y']
        self.f.seek(self.data_offset + chID * size * 4)
        datastring = ''.join([byteorder, str(size), formatcharacter])
        data = np.array(struct.unpack(datastring, self.f.read(4 * size))).reshape(
            (self.size['pixels']['y'], self.size['pixels']['x']))



        if corr_lines:
            for i in range(len(data[:, 0])):
                data[i, :] = data[i, :] - np.mean(data[i, :])

        return data

    def get_size_real(self):
        return abs(self.size['real']['x']), abs(self.size['real']['y'])

    def get_unit(self):
        return self.size['real']['unit']

    def get_size_pixel(self):
        return abs(self.size['pixels']['x']), abs(self.size['pixels']['y'])

    def get_size_per_pixel(self):
        return abs(self.size['real']['x'])/abs(self.size['pixels']['x']), abs(self.size['real']['y'])/abs(self.size['pixels']['y'])

class My_SXM_modified(My_SXM):
    """
    Class to deal with SZM files
    """

    @staticmethod
    def write_header(filename, range=None):
        """
        Write Header for SXM File
        :param filename: file to write header tp
        :return:
        """

        if range is None:
            with open(filename, "w") as file:
                settings = SXM_info.get_header_arr()
                for elem in settings:
                    arg = elem[1]
                    string = ""
                    try:
                        arg = arg[0]
                    except IndexError:
                        file.write(":{}:\n\n".format(elem[0]))
                        continue
                    if len(elem[1]) == 1:
                        string = "\t".join(arg)
                        file.write(":{}:\n{}\n".format(elem[0], string))
                        continue
                    else:
                        file.write(":{}:\n".format(elem[0]))
                        for arg in elem[1]:
                            file.write("{}\n".format("\t".join(arg)))
        else:

            # print(f"Range Parameter: {range}")
            # print(f"Range[0] Parameter: {range[0]}")
            # print(f"Range[0].ang Parameter: {range[0].ang}")

            with open(filename, "w") as file:
                settings = SXM_info.get_header_arr()
                for elem in settings:
                    # print(elem)
                    if elem[0] == "SCAN_RANGE":
                        elem = ("SCAN_RANGE", [[str(range[0]), str(range[1])]])
                    #    print(f"Adjusted Range to {elem}")
                    arg = elem[1]
                    string = ""
                    try:
                        arg = arg[0]
                    except IndexError:
                        file.write(":{}:\n\n".format(elem[0]))
                        continue
                    if len(elem[1]) == 1:
                        string = "\t".join(arg)
                        file.write(":{}:\n{}\n".format(elem[0], string))
                        continue
                    else:
                        file.write(":{}:\n".format(elem[0]))
                        for arg in elem[1]:
                            file.write("{}\n".format("\t".join(arg)))


    @staticmethod
    def write_sxm(filename, data, range=None):
        """
        write SXM data
        :param filename: file
        :param data: data to write
        :return:
        """

        # plt.imshow(data)
        # plt.show()
        # print(SXM_info.get_time())
        SXM_info.adjust_to_image(data, filename)
        # print(SXM_info.get_time())
        # print(SXM_info.get_header_dict()["REC_TIME"])
        try:
            with open(filename, "w") as file:
                file.write("")
        except FileNotFoundError:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        My_SXM_modified.write_header(filename, range=range)
        My_SXM_modified.write_image(filename, data)


def spm2arr(file):
    """
    Read .spm file and save result as grayscale png
    :param file:
    :return:
    """
    # print("ReadSPM ", file, " -> ", resultf)
    chID = 0  # 0: Image, 1: DI/DV, 2: ?
    spm = SPM(file)
    dat = spm.get_data(chID=chID)
    dat = dat.astype(float)
    maxi = np.amax(dat)
    mini = np.amin(dat)
    dat = (dat - mini) / (maxi - mini)

    if spm.get_size_real()[0] < 10:
        unit0 = 1E-6
    else:
        unit0 = 1E-9
    if spm.get_size_real()[1] < 10:
        unit1 = 1E-6
    else:
        unit1 = 1E-9
    range = (spm.get_size_real()[0]*unit0, spm.get_size_real()[1] * unit1 )

    return dat, range

def png_to_sxm(folder, resultf):

    files = [str(os.path.join(folder, elem)).split(".")[0] for elem in os.listdir(folder)]
    files = np.unique(files)
    os.makedirs(resultf, exist_ok=True)
    for i, file in enumerate(tqdm(files,desc="PNG2SXM")):
        res = os.path.join(resultf, os.path.basename(file) + ".sxm")
        im = file + ".png"
        tx = file + ".txt"
        range = np.zeros((2))
        with open(tx, 'r') as f:
            for i, line in enumerate(f):
                range[i] = line
        img = Image.open(im)
        arr = np.array(img)
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]
        arr = arr.T
        arr = np.flipud(arr)
        # plt.imshow(arr)
        # plt.title(arr.shape)
        plt.show()
        My_SXM_modified.write_sxm(res, arr, range)

def arr_2_sxm(arr, range, resfile):
    res = resfile
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]
    arr = arr.T
    arr = np.flipud(arr)
    # plt.imshow(arr)
    # plt.title(arr.shape)
    plt.show()
    My_SXM_modified.write_sxm(res, arr, range)

def load_labels(fld, tempdict):
    def get_spm_fn(l):
        cf = get_fn(l)
        parts = cf.split("\\")
        newparts = []
        for part in parts:
            if part.startswith("sample_png_pp"):
                newparts.append("spm")
            elif part == "Image.png":
                pass
            else:
                newparts.append(part)
        newfn = "\\".join(newparts)
        spmfld = os.path.dirname(newfn)
        for file in os.listdir(spmfld):
            if file.startswith(os.path.basename(newfn)):
                fn = os.path.join(spmfld, file)
                assert os.path.isfile(fn)
                return fn


    def get_fn(l):
        with open(tempdict, 'r') as f:
            for line in f:
                parts = line.split(";")
                # print(f"Comparing {parts[0].strip()} -- {l}")
                if parts[0].strip().split(".")[0] == l.split(".")[0]:
                    return parts[1].strip()

        return None

    lbls = os.listdir(os.path.join(fld, 'labels'))
    for lbl in lbls:
        spm = get_spm_fn(lbl)

        if spm is not None:
            img, range = spm2arr(spm)
            arr = np.array(img)
            arr -= np.amin(arr)
            arr *= 255 / np.amax(arr)
            if len(arr.shape) == 3:
                arr = arr[:, :, 0]



            with open(os.path.join(fld, 'labels', lbl), 'r') as f:
                boxes = []
                for line in f:
                    parts = line.split(" ")
                    cls = int(parts[0])
                    xcY = float(parts[1])
                    ycY = float(parts[2])
                    wY = float(parts[3])
                    hY = float(parts[4])
                    imw = arr.shape[1]
                    imh = arr.shape[0]

                    xl = int(np.floor((xcY - wY/2) * imw))
                    yt = int(np.floor((ycY - hY/2) * imh))
                    xr = int(np.ceil((xcY + wY/2) * imw))
                    yb = int(np.ceil((ycY + hY/2) * imh))

                    boxes.append((xl, yt, xr, yb))

            yield arr, boxes, range, os.path.basename(lbl).split(".")[0]

    return


def unit_exp(txt):
    txt = txt.strip()[1:]
    if txt == 'm':
        return ""
    elif txt == 'mm':
        return "E-3"
    elif txt == 'um':
        return "E-6"
    elif txt == 'nm':
        return "E-9"
    else:
        raise Exception(f"Unknown Unit: {txt}")

def spm_to_png(folder, resultf):
    def read_spm(file, resultf, resf2):
        """
        Read .spm file and save result as grayscale png
        :param file:
        :return:
        """

        # print("ReadSPM ", file, " -> ", resultf)

        chID = 0  # 0: Image, 1: DI/DV, 2: ?
        spm = SPM(file)
        dat = spm.get_data(chID=chID)

        dat = dat.astype(float)
        maxi = np.amax(dat)
        mini = np.amin(dat)
        dat = (dat - mini) / (maxi - mini)


        dat *= 255
        dat = dat.astype(np.uint8)
        dat = np.flipud(dat)
        # dat = np.fliplr(dat)
        img = Image.fromarray(dat, "L")
        img.save(resultf)



        with open(resf2, "w") as f:
            if spm.get_size_real()[0] <10:
                unit0 = "E-6"
            else:
                unit0 = "E-9"
            if spm.get_size_real()[1] <10:
                unit1 = "E-6"
            else:
                unit1 = "E-9"
            f.write(f"{spm.get_size_real()[0]}{unit0}\n")
            f.write(f"{spm.get_size_real()[1]}{unit1}\n")


    files = [os.path.join(folder, elem) for elem in os.listdir(folder)]

    os.makedirs(resultf, exist_ok=True)
    os.makedirs(resultf, exist_ok=True)

    for i, file in enumerate(tqdm(files, desc="SXM2PNG")):

        read_spm(file, os.path.join(resultf, f"Image{str(i).zfill(3)}.png"), os.path.join(resultf, f"Image{str(i).zfill(3)}.txt"))


def sxm_to_png(folder, resultf):
    def read_sxm(file, resultf, resf2,  dontflip=True):
        """
        Gets data from existing SXM file
        :param filename: file
        :param dontflip: flips the matrix as default to match image file
        :return:
        """
        assert os.path.exists(file)
        f = open(file, 'rb')
        l = ''
        key = ''
        header = {}
        while l != b':SCANIT_END:':
            l = f.readline().rstrip()
            if l[:1] == b':':
                key = l.split(b':')[1].decode('ascii')
                header[key] = []
            else:
                if l:  # remove empty lines
                    try:
                        header[key].append(l.decode('ascii').split())
                    except KeyError as e:
                        print(f"KeyError: {key}, probably wrong filetype")
        while f.read(1) != b'\x1a':
            pass
        assert f.read(1) == b'\x04'
        assert header['SCANIT_TYPE'][0][0] in ['FLOAT', 'INT', 'UINT', 'DOUBLE']
        data_offset = f.tell()
        size = dict(pixels={
            'x': int(header['SCAN_PIXELS'][0][0]),
            'y': int(header['SCAN_PIXELS'][0][1])
        }, real={
            'x': float(header['SCAN_RANGE'][0][0]),
            'y': float(header['SCAN_RANGE'][0][1]),
            'unit': 'm'
        })
        im_size = size['pixels']['x'] * size['pixels']['y']
        data = np.array(struct.unpack('<>'['MSBFIRST' == header['SCANIT_TYPE'][0][1]] + str(im_size) +
                                      {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[
                                          header['SCANIT_TYPE'][0][0]],
                                      f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))


        # print(size)
        # plt.imshow(data)
        # plt.show()

        # Crop Nan
        x = data[0][0]
         #print(x)
         #print(np.isnan(x))
         #print(np.nan in data)

        #rint("Axis0: ", data[:, 0])
        #rint("Axis1: ", data[0, :])

        for i in range(data.shape[0]):
            if np.isnan(data[i, 0]) or np.isnan(data[i, -1]):
                continue
            data = data[i:, :]
            break

         #plt.imshow(data)
         #plt.title("Cropped")
         #plt.show()

        if not dontflip:
            data = np.flipud(data)

        mini = np.amin(data)
        maxi = np.amax(data)

        data = data.astype(float)
        data = (data - mini) / (maxi - mini)

       # plt.imshow(data)
       # plt.title("normalized")
       # plt.show()
        # plt.imshow(data)
        # plt.title("Normalized")
        # plt.show()

        data *= 255
        data = data.astype(np.uint8)
        img = Image.fromarray(data, "L")
        img.save(resultf)

        with open(resf2, "w") as f:
            f.write(f"{size['real']['x']}\n")
            f.write(f"{size['real']['y']}\n")


    files = [os.path.join(folder, elem) for elem in os.listdir(folder)]

    os.makedirs(resultf, exist_ok=True)
    for i, file in enumerate(tqdm(files, desc="SXM2PNG")):
        read_sxm(file, os.path.join(resultf, f"Image{str(i).zfill(3)}.png"), os.path.join(resultf, f"Image{str(i).zfill(3)}.txt"))

def png2sxm(outfld, pngfld, evldir):
    os.makedirs(outfld, exist_ok=True)
    tempdict = os.path.join(evldir, "Eval_Results", "tempdict_yolo.csv")
    pred_fld = os.path.join(evldir, "yolo_prediction")
    for _, _, rng, name in load_labels(pred_fld, tempdict):
        img = os.path.join(pngfld, name + ".png")
        arr = np.array(Image.open(img))
        arr_2_sxm(arr, rng, os.path.join(outfld, name + ".sxm"))




if __name__ == "__main__":

    pngfld = r'C:\Users\seifert\PycharmProjects\SwinIR\results\gray_dn_50'
    evldir = r'D:\seifert\PycharmProjects\DNAmeasurement\Output\Try148_NoBirka_FIT_UseU_True_LaterTime_Conf70_checkpoint_epoch28_TestM0824_1540'
    outfld = r'D:\seifert\PycharmProjects\DNAmeasurement\SR\SwinIRDenoise\N50'
    png2sxm(outfld, pngfld, evldir)
    assert 1 == 2
    sxm_folder = r'D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Data\INDIV'
    pngf = r"D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Results\INDIV\Transformed\PNG"
    sxmf = r"D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Results\INDIV\Transformed\SXM"

    # sxm_folder = r'D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Test\NRSXM'
    # pngf = r"D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Test\PNG"
    # sxmf = r"D:\seifert\PycharmProjects\DNAmeasurement\IngoEraseMolecules\Test\SXM"

    spm_to_png(sxm_folder, pngf)
    png_to_sxm(pngf, sxmf)
    # png_to_sxm(pngf2, sxmf2)
    # sxm_to_png(sxmf2, pngf3)
    # png_to_sxm(pngf3, sxmf3)
