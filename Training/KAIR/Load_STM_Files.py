import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from datetime import datetime
from read_nid import read
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



