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

    def get_size_pixel(self):
        return abs(self.size['pixels']['x']), abs(self.size['pixels']['y'])

    def get_size_per_pixel(self):
        return abs(self.size['real']['x'])/abs(self.size['pixels']['x'])