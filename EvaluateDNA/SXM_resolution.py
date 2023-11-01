from ReadWriteSXM import *
from PIL import Image
from Training.KAIR.read_nid import *
import pySPM


class SM2(pySPM.SXM):
    def __init__(self, filename):
        self.filename = filename
        assert os.path.exists(self.filename)
        self.f = open(self.filename, 'rb')
        self.data_offset = 512
        line = self.f.readline()
        line = str(line)
        lines = line.split()
        # print(f"data_offset: {self.data_offset}")
        self.size = dict(pixels={
            'x': int(lines[7]),
            'y': int(lines[8])
        }, real={
            'x': float(lines[12]) * int(lines[7]),
            'y': float(lines[16]) * int(lines[8]),
            'unit': lines[14]
        })



    def get_data(self, chID=0, corr_lines=False):
        # chID = 0
        byteorder = '<'  # byteorder: @, =, < work; !, > don't work https://docs.python.org/3/library/struct.html
        formatcharacter = 'h'  # formatcharacter: i: int, f: float, l: long, d: double, h: short(int16) (uppercase for unsigned version)
        # zscale = ''
        size = self.size['pixels']['x'] * self.size['pixels']['y']
        self.f.seek(self.data_offset + chID * (size * 2 + self.data_offset))
        # datastring = '<' + str(size) + 'h'
        datastring = ''.join([byteorder, str(size), formatcharacter])
        data = np.array(struct.unpack(datastring, self.f.read(2 * size))).reshape(
            (self.size['pixels']['y'], self.size['pixels']['x']))
        # data = np.fliplr(data)
        # print(data)
        if corr_lines:
            for i in range(len(data[:, 0])):
                data[i, :] = data[i, :] - np.mean(data[i, :])
        return data

def load_png(file):
    arr = np.array(Image.open(file))
    print(arr.shape)
    return arr


def read_sxm(file, dontflip=True):
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

    x = data[0][0]


    for i in range(data.shape[0]):
        if np.isnan(data[i, 0]) or np.isnan(data[i, -1]):
            continue
        data = data[i:, :]
        break


    if not dontflip:
        data = np.flipud(data)

    mini = np.amin(data)
    maxi = np.amax(data)

    data = data.astype(float)
    data = (data - mini) / (maxi - mini)


    return data

def read_spm(file):
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
    return dat

if __name__ == "__main__":
    nid_file = r"D:\Dateien\KI_Speicher\SampleSXM\FormatComparison\Image01368.nid"
    sxm_file = r"D:\Dateien\KI_Speicher\SampleSXM\FormatComparison\Au-Au094.sxm"
    sm2_file = r"D:\Dateien\KI_Speicher\SampleSXM\FormatComparison\Image001255.SM2"
    bmp_file = r"D:\Dateien\KI_Speicher\SampleSXM\FormatComparison\Au-Au094.bmp"

    sm2 = SM2(sm2_file)
    sm2_arr = sm2.get_data()
    bmp_arr = np.array(Image.open(bmp_file))[:, :, 0]
    sxm_arr = read_sxm(sxm_file)
    nid_dat = read(nid_file)
    nid_arr = nid_dat.data[2]

    plt.imshow(nid_arr)
    plt.show()


    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(sm2_arr)
    axs[0, 0].set_title('SM2')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(sxm_arr)
    axs[0, 1].set_title('SXM')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(nid_arr)
    axs[1, 0].set_title('NID')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(bmp_arr)
    axs[1, 1].set_title('bmp')
    axs[1, 1].axis('off')

    plt.show()


    vals_bmp = bmp_arr.flatten().astype(float)
    vals_bmp -= np.amin(vals_bmp)
    vals_bmp /= np.amax(vals_bmp)

    vals_sm2 = sm2_arr.flatten().astype(float)
    vals_sm2 -= np.amin(vals_sm2)
    vals_sm2 /= np.amax(vals_sm2)

    vals_sxm = sxm_arr.flatten().astype(float)
    vals_sxm -= np.amin(vals_sxm)
    vals_sxm /= np.amax(vals_sxm)

    vals_nid_zr = nid_arr.flatten().astype(float)
    vals_nid = np.array([x for x in vals_nid_zr if x!=0])
    vals_nid -= np.amin(vals_nid)
    vals_nid /= np.amax(vals_nid)

    nid_uq = len(np.unique(nid_arr))
    sm2_uq = len(np.unique(sm2_arr))
    sxm_uq = len(np.unique(sxm_arr))
    bmp_uq = len(np.unique(bmp_arr))

    bins=1000
    plt.hist(vals_nid, label=f'nid - {nid_uq}', bins=bins)
    plt.hist(vals_bmp, label=f'bmp - {bmp_uq}', bins=bins)
    plt.hist(vals_sm2, label=f'sm2 - {sm2_uq}', bins=bins)
    plt.hist(vals_sxm, label=f'sxm - {sxm_uq}', bins=bins)
    plt.legend()
    plt.show()


