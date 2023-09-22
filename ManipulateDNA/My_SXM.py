import SXM_info
import os
import struct

import matplotlib.pyplot as plt
import numpy as np



class My_SXM():
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

            #print(f"Range Parameter: {range}")
            #print(f"Range[0] Parameter: {range[0]}")
            #print(f"Range[0].ang Parameter: {range[0].ang}")

            with open(filename, "w") as file:
                settings = SXM_info.get_header_arr()
                for elem in settings:
                   # print(elem)
                    if elem[0] == "SCAN_RANGE":
                        elem = ("SCAN_RANGE", [[str(range[0].ang) + "E-10", str(range[1].ang) + "E-10"]])
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


        #plt.imshow(data)
        #plt.show()
        #print(SXM_info.get_time())
        SXM_info.adjust_to_image(data, filename)
        #print(SXM_info.get_time())
        #print(SXM_info.get_header_dict()["REC_TIME"])
        try:
            with open(filename, "w") as file:
                file.write("")
        except FileNotFoundError:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        My_SXM.write_header(filename, range=range)
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

    #@staticmethod
    #def test():

