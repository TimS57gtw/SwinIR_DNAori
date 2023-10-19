import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from datetime import datetime
from read_nid import read
import SXM_info
class SXM():

    def __init__(self, file, dontflip=True):
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

        self.data = data
        self.real_size = size['real']['x'], size['real']['y']
        self.time_per_line = float(header['SCAN_TIME'][0][0])
        self.scan_dir = header['SCAN_DIR'][0][0]
        dte = header['REC_DATE'][0][0].split('.')
        day, month, year = dte
        tim = header['REC_TIME'][0][0].split(':')
        hour, min, sec = tim
        day = int(day)
        month = int(month)
        year = int(year)
        hour = int(hour)
        min = int(min)
        sec = int(sec)

        dtt = datetime(year, month, day, hour, min, sec)
        self.date = dtt


class NID():
    def __init__(self, fn, ch_id=0):
        nid = read(fn)
        p = nid.param

        with open(fn, 'r') as f:
            try:
                for line in f:
                    if line.startswith('Date'):
                        date = line.split('=')[1].strip()
                    elif line.startswith('Time'):
                        time = line.split('=')[1].strip()
                    elif line.startswith('Scan direction'):
                        self.scan_dir = line.split('=')[1].strip().lower()

            except Exception as e:
                pass

        dt = date.split('-')
        day, month, year = dt
        hour, min, sec = time.split(':')
        day = int(day)
        month = int(month)
        year = int(year)
        hour = int(hour)
        min = int(min)
        sec = int(sec)
        dtt = datetime(year, month, day, hour, min, sec)
        self.date = dtt

        self.real_size = p['Scan']['range']['Value'][0], p['Scan']['range']['Value'][1]
        self.time_per_line = p['Scan']['time/line']['Value'][0]
        hd = p['HeaderDump']
        self.data = nid.data[ch_id]




class SM2():
    def __init__(self, filename):
        self.filename = filename
        assert os.path.exists(self.filename)
        self.f = open(self.filename, 'rb')
        self.data_offset = 512
        line = self.f.readline()
        line = str(line)
        lines = line.split()
        header = {}
        cats = ['name',
                'version',
                'date',
                'time',
                'type',
                'data_type',
                'line_type',
                'xres',
                'yres',
                'data_size',
                'page_type',
                None,
                'xrange_min',
                'x_nmPpx',
                'xrange_unit',
                None,
                'yrange_min',
                'y_nmPpx',
                'yrange_unit',
                None,
                'zrange_min',
                'zrange_max',
                'zrange_unit',
                None, #XY
                None, #0
                None, #END
                None, #0
                None, #IV
                'scan',
                'period',
                None, # Scan
                None, #1
                'alt_speed',
                None,
                'id',
                'px',
                'Topography',
                'surface',
                'tip',
                None,
                'molecule',
                None,
                None
                ]
        for i in range(len(lines)):
            if cats[i] is not None:
                header[cats[i]] = lines[i]


        # for k in header.keys():
        #     print(f"{k} --> {header[k]}")
        # print(f"data_offset: {self.data_offset}")
        self.size = dict(pixels={
            'x': int(header['xres']),
            'y': int(header['yres'])
        }, real={
            'x': float(header['x_nmPpx']) * int(header['xres']),
            'y': float(header['y_nmPpx']) * int(header['yres']),
            'unit': header['xrange_unit']
        })

        self.data = self.get_data(chID=0)

        self.real_size = self.size['real']['x'], self.size['real']['y']
        self.time_per_line = float(header['period'])
        self.scan_dir = 'down'
        dte = header['date'].split('/')
        month, day, year = dte
        tim = header['time'].split(':')
        hour, min, sec = tim
        day = int(day)
        month = int(month)
        year = int(year)
        hour = int(hour)
        min = int(min)
        sec = int(sec)

        dtt = datetime(year, month, day, hour, min, sec)
        self.date = dtt


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


class Spectrum_dat():
    def __init__(self, fn):


        with open(fn, 'r') as f:
            for line in f:
                if line.startswith("Saved Date"):
                    parts = line.split('\t')
                    dt = parts[1]
                    dt = dt.split(' ')
                    date = dt[0]
                    time = dt[1]
                    day, month, year = date.split('.')
                    hour, min, sec = time.split(':')
                    day = int(day)
                    month = int(month)
                    year = int(year)
                    hour = int(hour)
                    min = int(min)
                    sec = int(sec)
                    self.timestamp = datetime(year, month, day, hour, min, sec)
                elif line.startswith("[DATA]"):
                    break

            header = f.readline()
            header = header.split('\t')
            self.f_head = header[0].strip()
            self.a_head = header[1].strip()
            freqs = []
            ampls = []
            for line in f:
                parts = line.split('\t')
                freq = float(parts[0])
                ampl = float(parts[1])
                freqs.append(freq)
                ampls.append(ampl)

            self.freq = np.array(freqs)
            self.ampl = np.array(ampls)



class My_SXM():
    """
    Class to deal with SZM files
    """

    @staticmethod
    def write_header(filename, range, date, scn_dir, scan_time):
        """
        Write Header for SXM File
        :param filename: file to write header tp
        :return:
        """


        with open(filename, "w") as file:
            settings = SXM_info.get_header_arr()
            for elem in settings:
                # print(elem)
                if elem[0] == "SCAN_RANGE":
                    elem = ("SCAN_RANGE", [[str(range[0]), str(range[1])]])


                if elem[0] == "REC_DATE":
                    elem = ("REC_DATE", [[f'{date.day}.{date.month}.{date.year}']])
                if elem[0] == "REC_TIME":
                    elem = ("REC_TIME", [[f'{date.hour}:{date.minute}:{date.second}']])

                if elem[0] == "SCAN_DIR":
                    elem = ("SCAN_DIR", [[str(scn_dir)]])
                if elem[0] == "SCAN_TIME":
                    elem = ("SCAN_TIME", [[str(scan_time), str(scan_time)]])

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
    def write_sxm(filename, data, range=None, date=None, scn_dir=None, linespeed=1):
        """
        write SXM data
        :param filename: file
        :param data: data to write
        :return:
        """
        if date is None:
            date = datetime.now()

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

        My_SXM.write_header(filename, range=range, date=date, scn_dir=scn_dir, scan_time=linespeed)
        My_SXM.write_image(filename, data)


    @staticmethod
    def _fill_with_zeros(mat):
        """
        Pads a matrix with zeros to make it square
        :param mat:
        :return:
        """
        w, h = np.shape(mat)

        newmat = -10 * np.ones((max(w, h), max(w, h)))
        for i in range(w):
            for j in range(h):
                newmat[i, j] = mat[i, j]

        for i in range(np.shape(newmat)[0]):
            for j in range(np.shape(newmat)[1]):
                if newmat[i, j] == -10:
                    #newmat[i, j] = np.nan
                    newmat[i, j] = 0


        return newmat

    @staticmethod
    def write_image(filename, image):
        """
        Write the image as an SXM File
        :param filename: filename to write to
        :param image: image
        :return:
        """

        ang_per_bright = 1000 * 1e-10 / 255
        newmat = -4 * np.ones(np.shape(image))


        for i in range(np.shape(image)[0]):
            for j in range(np.shape(image)[1]):
                newmat[i, j] = ang_per_bright * image[i, j]



        #plt.imshow(newmat)
        #plt.show()

        if np.shape(newmat)[0] != np.shape(newmat)[1]:
            newmat = My_SXM._fill_with_zeros(newmat)

        im_size = np.shape(newmat)[0] * np.shape(newmat)[1]
        #print("XM: {}".format(np.shape(newmat)[0]))
        #print("YM: {}".format(np.shape(newmat)[1]))

        flippedmat = np.zeros(np.shape(newmat))
        hi = np.shape(flippedmat)[1]
        wi = np.shape(flippedmat)[0]
        for i in range(wi):
            for j in range(hi):
                flippedmat[i, j] = newmat[hi - j - 1, i]

        newmat = flippedmat

        #plt.imshow(newmat)
        #plt.show()

        with open(filename, "ab") as file:
            file.write(b'\n')
            file.write(b'\x1a')
            file.write(b'\x04')

            header = SXM_info.get_header_dict()

            size = dict(pixels={
                'x': int(header['SCAN_PIXELS'][0][0]),
                'y': int(header['SCAN_PIXELS'][0][1])
            }, real={
                'x': float(header['SCAN_RANGE'][0][0]),
                'y': float(header['SCAN_RANGE'][0][1]),
                'unit': 'm'
            })
            #print("X: {}".format(size['pixels']['x']))
            #print("Y: {}".format(size['pixels']['y']))

            if header['SCANIT_TYPE'][0][1] == 'MSBFIRST':
                bitorder = '>'
            else:
                bitorder = '<'

            length = '1'

            if header['SCANIT_TYPE'][0][0] == 'FLOAT':
                d_type = 'f'
            elif header['SCANIT_TYPE'][0][0] == 'INT':
                d_type = 'i'
            elif header['SCANIT_TYPE'][0][0] == 'UINT':
                d_type = 'I'
            elif header['SCANIT_TYPE'][0][0] == 'DOUBLE':
                d_type = 'd'
            else:
                print("Error reading SCANIT_TYPE. Unexpected: {}".format(header['SCANIT_TYPE'][0][0]))
                d_type = 'f'

            data = newmat.reshape(im_size,)

            format = bitorder + length + d_type

            for elem in data:
                file.write(struct.pack(format, elem))


    @staticmethod
    def get_data_test(filename):
        """
        Test method to get data from existing SXM file
        :param filename: existing SXM file
        :return:
        """
        assert os.path.exists(filename)
        f = open(filename, 'rb')
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
                    header[key].append(l.decode('ascii').split())

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
        print(struct.unpack('>' + str(im_size) + 'f',
                                      f.read(4 * im_size)))

        data = np.flipud(data)
        return data


    @staticmethod
    def get_informations(filename):
        """
        Testing method to get Header information from SXM file
        :param filename: path to SXM file
        :return:
        """
        assert os.path.exists(filename)
        f = open(filename, 'rb')
        l = ''
        key = ''
        header = {}
        ret_str = []
        ret_str.append("Header_Information for File {}".format(filename))
        while l != b':SCANIT_END:':
            l = f.readline().rstrip()
            if l[:1] == b':':
                key = l.split(b':')[1].decode('ascii')
                header[key] = []
            else:
                if l:  # remove empty lines
                    header[key].append(l.decode('ascii').split())

        ret_str.append("Key: {}".format(key))
        ret_str.append("header[key]: {}".format(header[key]))
        ret_str.append("header:")
        for x in header.keys():
            ret_str.append("{}: {}".format(x, header[x]))

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
                                      {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[header['SCANIT_TYPE'][0][0]],
                                      f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))



        return "\n".join(ret_str)

    @staticmethod
    def get_data(filename, dontflip=False):
        """
        Gets data from existing SXM file
        :param filename: file
        :param dontflip: flips the matrix as default to match image file
        :return:
        """
        assert os.path.exists(filename)
        f = open(filename, 'rb')
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
                    header[key].append(l.decode('ascii').split())
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
                                      {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[header['SCANIT_TYPE'][0][0]],
                                      f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))

        if not dontflip:
            data = np.flipud(data)


        return data


    @staticmethod
    def show_data(filename):
        """
        Shows data from sxm file using matplotlib.imshow
        :param filename: sxm file
        :return:
        """
        #print(My_SXM.get_informations(filename))
        plt.imshow(My_SXM.get_data(filename))
        plt.show()



