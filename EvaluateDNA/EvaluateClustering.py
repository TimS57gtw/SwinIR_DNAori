import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pySPM
import struct


def get_position_list(txt_list, use_unsuccessful=True):
    ret_list = []
    for file in txt_list:

        with open(file, "r") as f:
            x_pos_px = None
            y_pos_px = None
            cov = None
            success = False
            distance = None

            for line in f:
                if line.startswith("Pos_Px_X"):
                    parts = line.split(":")
                    x_pos_px = float(parts[1])
                elif line.startswith("Pos_Px_Y"):
                    parts = line.split(":")
                    y_pos_px = float(parts[1])
                elif line.startswith("Crop_nMpPX_X"):
                    parts = line.split(":")
                    cov = float(parts[1])
                elif line.startswith("Success"):
                    parts = line.split(":")
                    success = parts[1].strip() == "True"
                elif line.startswith("Dist_nm"):
                    parts = line.split(":")
                    distance = float(parts[1])
                else:
                    pass
            if use_unsuccessful:
                if not success:
                    distance = -1
            else:
                if not success:
                    continue
                #assert success

            ret_list.append([x_pos_px * cov, y_pos_px * cov, distance])

    return ret_list






def get_image_statistics(txt_list, no_neighbors, use_unsuccessful=True):
    """
    takes list of crops in one image and returns list (k neighbour_distances -> markerDistance)
    """
    # print(f"With successless: {len(get_position_list(txt_list, True))}, without: {len(get_position_list(txt_list, False))}")
    # input()
    positions = get_position_list(txt_list, use_unsuccessful) # 1 image

    relations = []

    # molecule_list: List of crops in one image
    img_pos = []
    for molecule in positions:
        img_pos.append(np.array([molecule[0], molecule[1]]))
    distance_matrix = 10_000 * np.ones((len(positions), len(positions)))
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            distance = np.linalg.norm(img_pos[i] - img_pos[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    for i in range(len(positions)):
        marker_dist = positions[i][2]
        if marker_dist > 0:
            distances = distance_matrix[i, :]
            srt_ds = sorted(distances)
            relations.append(([srt_ds[k] for k in range(no_neighbors)], marker_dist))

    return relations






class SPM(pySPM.SXM):
    def __init__(self, filename):
        self.filename = filename
        assert os.path.exists(self.filename)
        self.f = open(self.filename, 'rb')

        self.header = {}
        self.data_offset = None

        self.header_raw = b""
        while True:
            l = self.f.readline().rstrip()
            self.header_raw += l + b"\n"
            if l.startswith(bytes("\\", "ansi")):
                if b':' in l:
                    key = l.split(b':')[0].decode('ansi')
                    self.header[key] = l.split(b':')[1].decode('ansi')
            if l.startswith(bytes("\\Data offset", "ansi")) and self.data_offset is None:
                self.data_offset = int(self.header["\\Data offset"])
            if l.startswith(bytes("\\*File list end", "ansi")):
                break
        scan_size = self.header['\\Scan Size']
        scan_size = scan_size.split()
        self.size = dict(pixels={
            'x': int(self.header['\\Valid data len X']),
            'y': int(self.header['\\Valid data len Y'])
        }, real={
            'x': float(scan_size[0]),
            'y': float(scan_size[1]) * int(self.header['\\Valid data len Y']) / int(self.header['\\Valid data len X']),
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

        #data = np.flipud(data)

        if corr_lines:
            for i in range(len(data[:, 0])):
                data[i, :] = data[i, :] - np.mean(data[i, :])

        return data

    def get_size_real(self):
        return abs(self.size['real']['x']), abs(self.size['real']['y']), self.size['real']['unit']

    def get_size_pixel(self):
        return abs(self.size['pixels']['x']), abs(self.size['pixels']['y'])

    def get_size_per_pixel(self):
        return abs(self.size['real']['x']) / abs(self.size['pixels']['x'])

    def get_header_raw(self, as_string=False):
        if as_string:
            return self.header_raw.decode('ansi')
        return self.header_raw





def average_distances(img_stats, function="mean"):
    ret_list = []

    for elem in img_stats:
        k_distances = elem[0]
        if function=="mean":
            ret_list.append((np.average(k_distances), elem[1]))
        elif function=="exp":
            sum = 0
            weight = 0
            for i in range(len(k_distances)):
                sum += k_distances[i] * np.exp(-(i+1))
                weight += np.exp(-(i+1))
            sum /= weight
            ret_list.append((sum, elem[1]))
        else:
            raise NotImplementedError

    return ret_list

def was_successful(f):
    for line in f:
        if line.startswith("Success"):
            if "True" in line:
                return True
    return False


def analyze_clustering(infld, outf, no_neighbors=3, mittl_fkt="exp", use_unsuccessful=True, subcat=True):
    plot_img = False
    os.makedirs(outf, exist_ok=True)
    wghts_nn_ds = []
    marker_ds = []
    image_name = []

    images = []
    infld = os.path.join(infld, "crop_origami")

    if subcat:
        for subf in os.listdir(infld):
            for tmp in os.listdir(os.path.join(infld, subf)):
                for fld in os.listdir(os.path.join(infld, subf, tmp)):
                    # print(os.path.join(infld, subf))
                    assert os.path.isdir(os.path.join(infld, subf, tmp, fld)), f"Invalid Dir: {(os.path.join(infld, subf, tmp, fld))}"
                    images.append(os.path.join(infld, subf, tmp, fld))

    else:
        for subf in os.listdir(infld):
            for fld in os.listdir(os.path.join(infld, subf)):
                # print(os.path.join(infld, subf))
                assert os.path.isdir(os.path.join(infld, subf, fld)), f"Invalid Dir: {(os.path.join(infld, subf, fld))}"
                images.append(os.path.join(infld, subf, fld))

    for img_fld in tqdm(images):
        crops = []
        for elem in os.listdir(img_fld):
            if os.path.isdir(os.path.join(img_fld, elem)):
                if os.path.isfile(os.path.join(img_fld, elem, "Image.txt")):
                    with open(os.path.join(img_fld, elem, "Image.txt"), "r") as f:
                        if use_unsuccessful or was_successful(f):
                            crops.append(os.path.join(img_fld, elem, "Image.txt"))
                        # else:
                        #     print(f"False Crop {os.path.join(img_fld, elem, 'Image.txt')}")
                else:
                    print(f"Invalid crop {os.path.join(img_fld, elem)}")

        if len(crops) <= no_neighbors:
            continue

        image_statistics = get_image_statistics(crops, no_neighbors, use_unsuccessful=use_unsuccessful)
        # image_statistics: [ ([d1, d2, d3, ..., dk], md) ] for one image

        wght_img_stats = average_distances(image_statistics, function=mittl_fkt)
        for elem in wght_img_stats:
            assert elem[1] > 0
            wghts_nn_ds.append(elem[0])
            marker_ds.append(elem[1])
            image_name.append(os.path.basename(img_fld))

        if plot_img:
            xs = [e[0] for e in wght_img_stats]
            ys = [e[1] for e in wght_img_stats]

            plt.scatter(xs, ys)
            plt.xlabel(f"Weighted NN distance")
            plt.ylabel("marker-distance")
            plt.title(f"{mittl_fkt}-averaged {no_neighbors}-NN")
            plt.show()


    if True:
        plt.scatter(wghts_nn_ds, marker_ds)
        plt.xlabel(f"Weighted NN distance")
        plt.ylabel("marker-distance")
        plt.title(f"Total {mittl_fkt}-averaged {no_neighbors}-NN")
        plt.savefig(os.path.join(outf, f"scatter_{no_neighbors}_{mittl_fkt}.png"))
        plt.clf()

    with open(os.path.join(outf, f"cluster_resutls_{no_neighbors}_{mittl_fkt}.csv"), "w") as f:
        f.write(f"ImageName,{no_neighbors}-{mittl_fkt},markdist\n")
        for i in tqdm(range(len(wghts_nn_ds)), desc="Saving Results"):
            f.write(f"{image_name[i]},{wghts_nn_ds[i]},{marker_ds[i]}\n")

    print(f"Saved as {os.path.join(outf, f'cluster_results_{no_neighbors}_{mittl_fkt}.csv')}")



def analyze_custering_perImage(infld, outfld, subcat=True):
    infld_co = os.path.join(infld, "crop_origami")
    infld_sp = os.path.join(infld, "spm")

    os.makedirs(outfld, exist_ok=True)
    use_unsuccessful = True
    images = []
    spm_files = []

    if subcat:
        for subf in os.listdir(infld_co):
            for tmp in os.listdir(os.path.join(infld_co, subf)):
                for fld in os.listdir(os.path.join(infld_co, subf, tmp)):
                    # print(os.path.join(infld, subf))
                    assert os.path.isdir(os.path.join(infld_co, subf, tmp, fld)), f"Invalid Dir: {(os.path.join(infld_co, subf, tmp, fld))}"
                    images.append(os.path.join(infld_co, subf, tmp, fld))
                    spm_files.append(os.path.join(infld_sp, subf, tmp, fld + ".spm"))

    else:
        for subf in os.listdir(infld_co):
            for fld in os.listdir(os.path.join(infld_co, subf)):
                # print(os.path.join(infld, subf))
                assert os.path.isdir(os.path.join(infld_co, subf, fld)), f"Invalid Dir: {(os.path.join(infld_co, subf, fld))}"
                images.append(os.path.join(infld_co, subf, fld))
                spm_files.append(os.path.join(infld_sp, subf, fld + ".spm"))


    num_molsPerImg = []
    avg_MDTs = []
    med_MDTs = []
    areas = []

    for img_fld in tqdm(images, desc="Iterating Images..."):
        crops = []
        for elem in os.listdir(img_fld):
            if os.path.isdir(os.path.join(img_fld, elem)):
                if os.path.isfile(os.path.join(img_fld, elem, "Image.txt")):
                    with open(os.path.join(img_fld, elem, "Image.txt"), "r") as f:
                        if use_unsuccessful or was_successful(f):
                            crops.append(os.path.join(img_fld, elem, "Image.txt"))

                else:
                    print(f"Invalid crop {os.path.join(img_fld, elem)}")

        pos_list = get_position_list(crops, use_unsuccessful)
        num_crops = len(pos_list)
        markdist = [elem[2] for elem in pos_list]
        validmd = []
        for i in range(len(markdist)):
            if markdist[i] > 0:
                validmd.append(markdist)
        num_molsPerImg.append(num_crops)
        if len(validmd) > 0:
            avg_MDTs.append(np.average(validmd))
            med_MDTs.append(np.median(validmd))
        else:
            avg_MDTs.append(-1)
            med_MDTs.append(-1)

    for spmf in tqdm(spm_files, desc="Iterating SPMs"):
        spm_file = SPM(spmf)
        scansize = spm_file.get_size_real()
        area = scansize[0] * scansize[1]
        if scansize[2] == "nm":
            area /= 1e6
        areas.append(area)
    assert len(avg_MDTs) == len(areas)
    with open(os.path.join(outfld, "res.csv"), 'w') as f:
        f.write(f"num_molecules;averageMDT;medianMDT;area;density;inverseDensity\n")
        for i in range(len(avg_MDTs)):
            if avg_MDTs[i] > 0:
                f.write(f"{num_molsPerImg[i]};{avg_MDTs[i]};{med_MDTs[i]};{areas[i]};{num_molsPerImg[i]/areas[i]:.3f};{areas[i]/num_molsPerImg[i]}\n")
    avg_pairs = []
    med_pairs = []
    for i in range(len(avg_MDTs)):
        if avg_MDTs[i] > 0:
            avg_pairs.append((num_molsPerImg[i]/areas[i], avg_MDTs[i]))
            med_pairs.append((num_molsPerImg[i]/areas[i], med_MDTs[i]))
    plt.scatter([e[0] for e in avg_pairs], [e[1] for e in avg_pairs])
    plt.title("AverageDistance over Density")
    plt.ylabel("Average Markerdistance")
    plt.xlabel("Density in um^-2")
    plt.savefig(os.path.join(outfld, "average_dens.png"))
    plt.cla()
    plt.scatter([e[0] for e in med_pairs], [e[1] for e in med_pairs])
    plt.title("MedianDistance over Density")
    plt.ylabel("Median Markerdistance")
    plt.xlabel("Density in um^-2")
    plt.savefig(os.path.join(outfld, "median_dens.png"))
    plt.cla()

    plt.scatter([1/e[0] for e in avg_pairs], [e[1] for e in avg_pairs])
    plt.title("AverageDistance over InverseDensity")
    plt.ylabel("Average Markerdistance")
    plt.xlabel("Density in um^2")
    plt.savefig(os.path.join(outfld, "average_invDens.png"))
    plt.cla()
    plt.scatter([1/e[0] for e in med_pairs], [e[1] for e in med_pairs])
    plt.title("MedianDistance over InverseDensity")
    plt.ylabel("Median Markerdistance")
    plt.xlabel("Density in um^2")
    plt.savefig(os.path.join(outfld, "median_InvDens.png"))
    plt.cla()

if __name__ == "__main__":
    # in_fld = r"D:\seifert\PycharmProjects\DNAmeasurement\Output\Try10_INDIV_FIT_UseU_True_LaterTime_Conf30"
    in_fld = r"D:\seifert\PycharmProjects\DNAmeasurement\Output\Try6_dens_FIT_UseU_True_LaterTime_Conf30"

    subcat=False # 1 additional subcategory

    out_dlsPI = os.path.join(in_fld, f"ClusterEval_PerImage")
    analyze_custering_perImage(in_fld, out_dlsPI, subcat=subcat)
    for use_unsuccessful in [True, False]:
        out_dls = os.path.join(in_fld, f"ClusterEval_{str(use_unsuccessful)}")
        for nn in [1, 3, 5, 10]:
            for fkt in ["mean", "exp"]:
                analyze_clustering(in_fld, out_dls, no_neighbors=nn, mittl_fkt=fkt, use_unsuccessful=use_unsuccessful, subcat=subcat)

