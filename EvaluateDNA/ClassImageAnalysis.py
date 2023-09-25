import copy
import random
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
from Preprocessing import pretransform_image
from skimage.morphology import skeletonize
import traceback


def show_img(fp):
    img = Image.open(fp)
    img.show()


def pretransform_img(img, temp_folder):
    idx = random.randint(0, 10000)
    fn_after = os.path.join(temp_folder, f"temp{idx}.npy")
    pretransform_image(img, fn_after, enhance_contrast=True, do_flatten_border=True, flatten_line_90=False)
    img = np.load(fn_after, allow_pickle=True)
    # plt.imshow(img)
    # plt.show()
    return img


def canny(image, f1=None, f2=None, show_all=False):
    if f1 is not None:
        plt.imsave(f1, image, cmap='gray')
    if show_all:
        plt.imshow(image, cmap='gray')
        plt.title("Read Image")
        plt.show()
    s = 5
    kernel_size = int(image.shape[0] / 4) + (int(image.shape[0] / 4) + 1) % 2
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=s, sigmaY=s)

    if show_all:
        plt.imshow(image, cmap='gray')
        plt.title("Blurred image")
        plt.show()
    th1 = 50
    th2 = 60
    edges = cv2.Canny(image, th1, th2)
    if f2 is not None:
        plt.imsave(f2, edges)

    if show_all:
        plt.imshow(edges)
        plt.title(f"Thrsh: {th1}, {th2}")
        plt.show()

    while False:
        th1 = int(input("Th1 -> "))
        th2 = int(input("Th2 -> "))

        edges = cv2.Canny(image, th1, th2)
        plt.imshow(edges)
        plt.title(f"Thrsh: {th1}, {th2}")
        plt.show()

    return image, edges


def psotprocessing_hough_lines(lines, cdst, edges, show_all=False):
    """
    rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    :param lines:
    :return:
    """

    img_loc = copy.deepcopy(cdst)
    rhos = []

    # for line in lines:
    #     img_loc = copy.deepcopy(cdst)
#
    #     rho = line[0][0]
    #     theta = line[0][1]
    #     rhos.append(rho)
    #     thetas.append(theta)
    #     a = math.cos(theta)
    #     b = math.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #     pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #     cv2.line(img_loc, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    #     # plt.title(f"r: {rho}, t: {theta}")
    #     # plt.imshow(img_loc)
    #     # plt.show()

    def to_basis(ang):
        ang = ang * 180 / np.pi
        while ang >= 180:
            ang -= 180
        if ang > 90:
            ang = 180 - ang
        return ang

    def pt_lin_dist(p0, line):
        try:
            theta = line[1]
            rho = line[0]
        except IndexError:
            rho = line[0][0]
            theta = line[0][1]

        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho

        p2 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        p1 = (x0, y0)

        ob = (p2[0] - p1[0]) * (p1[1] - p0[1]) - (p1[0] - p0[0]) * (p2[1] - p1[1])
        ut = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
        return abs(ob) / np.sqrt(ut)

    def pts_dist(line, pts=None,gaussian=True):
        if pts is None:
            pts = np.argwhere(edges == 255)
        imgw = edges.shape[0]
        if gaussian:
            gaussian_weight= lambda x: np.exp(-x ** 2 / (imgw / 60))
        else:
            gaussian_weight = lambda x : x
        sum = 0
        for pt in pts:
            sum += gaussian_weight(pt_lin_dist(pt, line))
        return sum

    def visu_lines(loclines, tit=None):
        if not show_all:
            return
        img_loc = copy.deepcopy(cdst)
        for line in loclines:
            try:
                rho = line[0]
                theta = line[1]
            except IndexError:
                rho = line[0][0]
                theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img_loc, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        plt.imshow(img_loc)
        if tit is not None:
            plt.title(tit)
        plt.show()

    # Find maximum orthogonal pairs

    def find_correct_angle(lines):

        erg_dict = {}
        for line in lines:
            theta = line[0][1]
            while theta > np.pi / 2:
                theta -= np.pi / 2
            five_deg =  (theta * 180 / np.pi) / 9
            five_deg = int(round(five_deg))
            if five_deg not in erg_dict.keys():
                erg_dict[five_deg] = pts_dist(line)
            else:
                erg_dict[five_deg] += pts_dist(line)

        kvps = [(k, erg_dict[k]) for k in erg_dict.keys()]
        kvps = sorted(kvps, key = lambda x : x[1])

        minangles = []
        for i, kvp in enumerate(kvps):
            loclines = []
            for line in lines:
                theta = line[0][1]
                while theta > np.pi / 2:
                    theta -= np.pi / 2
                five_deg = (theta * 180 / np.pi) / 9
                five_deg = int(round(five_deg))
                if five_deg == kvp[0]:
                    loclines.append(line)
            if show_all:
                visu_lines(loclines, f"Dist: {kvp[1]}")

        min_fp = kvps[-1][0]
        for line in lines:
            theta = line[0][1]
            while theta > np.pi / 2:
                theta -= np.pi / 2
            five_deg = (theta * 180 / np.pi) / 9
            five_deg = int(round(five_deg))
            if five_deg == min_fp:
                minangles.append(theta)

        return np.average(minangles)


        # thetas = []
        # for line in lines:
        #     thetas.append(line[0][1])
        # rearr_th = []
        # for th in thetas:
        #     x = th
        #     while x > np.pi / 2:
        #         x -= np.pi / 2
        #     rearr_th.append(x)
#
        # rearr_th = sorted(rearr_th)
        # if show_all:
        #     plt.plot(rearr_th)
        #     plt.title("Thetas")
        #     plt.show()
#
        # ang = rearr_th[int(len(rearr_th) / 2)]
        # return ang


    # Test Dist to pts
    def test_pts_dist(pts=None):
        if pts is None:
            pts = np.argwhere(edges == 255)
        pairs = []
        for line in lines:
            pairs.append((line, pts_dist(line, pts, True)))
        pairs = sorted(pairs, key=lambda x : x[1])

        for i in range(1, len(pairs), 5):
            lineslocci = [p[0] for p in pairs[:i]]
            visu_lines(lineslocci, f"Top {i+1} lines ")

    if show_all:
        test_pts_dist()



    ang = find_correct_angle(lines)
    correct_angle = ang  # and + 90
    correct_angle2 = ang + np.pi / 2
    # print("Correct angles: ", correct_angle, correct_angle2)

    deviation = 5 * np.pi / 180  # Which deviation is allowed to be comsidred correct
    angle_filtered_lines = []
    for line in lines:
        theta = line[0][1]
        # print(f"{theta}: {abs(correct_angle - theta)} - {abs(correct_angle2 - theta)}")
        if abs(correct_angle - theta) > deviation and abs(correct_angle2 - theta) > deviation:
            continue
        else:
            angle_filtered_lines.append(line)

    img_loc = copy.deepcopy(cdst)
    for line in angle_filtered_lines:
        rho = line[0][0]
        theta = line[0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(img_loc, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    if show_all:
        plt.imshow(img_loc)
        plt.title("Angle_Filtered")
        plt.show()

    # Position Filtering
    # Bekannt: "u-ere Linie wahrscheinlicher, innere kann ssDNA sein
    # Bakannt: Ungef'hres Seitenverhaeltnis der DNA

    # berechnung distanz von lines
    def line_dist(l1, l2):
        if abs(l1[1] - l2[1]) > deviation:
            return 0
        else:
            return abs(l1[0] - l2[0])


    pts = np.argwhere(edges == 255)
    assert len(pts) > 0

    # plt.imshow(edges)
    # plt.title("edges")
    # plt.show()



        # Rem

    # for line in angle_filtered_lines:
    ##
    #     testcpy = copy.deepcopy(edges)
    ##
    #     rho = line[0][0]
    #     theta = line[0][1]
    #     a = math.cos(theta)
    #     b = math.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #     pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #     cv2.line(testcpy, pt1, pt2, 255, 1, cv2.LINE_AA)
    ##
    #     plt.imshow(testcpy)
    #     plt.title(pts_dist(line[0]))
    #     plt.show()
    #
    if show_all:
        visu_lines(lines, "All")

    clusters = []  # better: Choose lines iwth maximum intersection of white spots in img
    for line in angle_filtered_lines:
        if len(clusters) == 0:
            clusters.append([line])
        else:
            found = False
            # print("Clusters: ", clusters)

            for cluster in clusters:
                # print("Cluster: ", cluster)

                avg_t = 0
                ct = 0
                for lineloc in cluster:
                    # print("Lineloc: ", lineloc)
                    avg_t += lineloc[0][1]
                    ct += 1
                avg_t /= ct
                if abs(avg_t - line[0][1]) < deviation:
                    cluster.append(line)
                    found = True
                    break
            if not found:
                clusters.append([line])

    assert len(clusters) == 2, f"Found multiple clusters of lines: {len(clusters)}"
    visu_lines(angle_filtered_lines)

    for cluster in clusters:
        visu_lines(cluster)


    dist_filtered_lines = []

    for cluster in clusters:
        rhos = []
        if len(cluster) < 2:
            if len(cluster) == 1:
                dist_filtered_lines.append(cluster[0])
            continue
        for line in cluster:
            rhos.append(line[0][0])

        center = (max(rhos) + min(rhos)) / 2

        subclusters = [[], []]

        for line in cluster:
            if line[0][0] <= center:
                subclusters[0].append(line[0])
            else:
                subclusters[1].append(line[0])

        rhos_left = []
        rhos_right = []
        for line in subclusters[0]:
            rhos_left.append(line[0])
        for line in subclusters[1]:
            rhos_right.append(line[0])

        if len(subclusters[0]) == 1:
            # pass
            dist_filtered_lines.append(subclusters[0][0])
            visu_lines(dist_filtered_lines, "Dist filtered appended")
            # dist_filtered_lines.append(subclusters[0][0])
        else:
            med_l = sorted(rhos_left)[int(len(rhos_left) / 2)]
            for line in subclusters[0]:
                if line[0] == med_l:
                    dist_filtered_lines.append(line)
                    visu_lines(dist_filtered_lines, "Dist filtered appended")

                    break
        if len(subclusters[1]) == 1:
            # pass
            dist_filtered_lines.append(subclusters[1][0])
            visu_lines(dist_filtered_lines, "Dist filtered appended")

        else:
            med_r = sorted(rhos_right)[int(len(rhos_right) / 2)]
            for line in subclusters[1]:
                if line[0] == med_r:
                    dist_filtered_lines.append(line)
                    visu_lines(dist_filtered_lines, "Dist filtered appended")
                    break


    img_loc = copy.deepcopy(cdst)
    for line in dist_filtered_lines:
        rho = line[0]
        theta = line[1]
        rhos.append(rho)
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(img_loc, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    if show_all:
        plt.imshow(img_loc)
        plt.title("Dist_Filtered")
        plt.show()

    assert len(dist_filtered_lines) == 4, "Not enough Lines found"
    return dist_filtered_lines


def hough(edges, no_lines=32, f3=None, f4=None, show_all=False):
    image = edges
    cdst = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    ang = 1.0 / np.pi
    xs = []
    ys = []
    for i in range(1000):
        lines = cv2.HoughLines(image, 1.0, ang, threshold=i)
        xs.append(i)
        if lines is not None:
            ys.append(len(lines))
        else:
            ys.append(0)
        if len(lines) < no_lines:
            i -= 1
            lines = cv2.HoughLines(image, 1.0, ang, threshold=i)
            break

    if show_all:
        plt.plot(xs, ys)
        plt.title("Lines over threshold")
        plt.show()

    cdst_pp = copy.deepcopy(cdst)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    if f3 is not None:
        plt.imsave(f3, cdst)
    if show_all:
        plt.imshow(cdst)
        plt.title("Standard Hough before PP")
        plt.show()

    lines = psotprocessing_hough_lines(lines, cdst_pp, edges=edges)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0]
            theta = lines[i][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst_pp, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    if f4 is not None:
        plt.imsave(f4, cdst_pp)
    if show_all:
        plt.imshow(cdst_pp)
        plt.title("Standard Hough - Postprocessed")
        plt.show()

    return lines


def npy2img(arr):
    arr *= 255
    arr = arr.astype(np.uint8)
    # cv2.imshow("Hi", arr)
    return arr


def blur_thin_edges(edg):
    kernel_size = int(edg.shape[0] / 5) + (int(edg.shape[0] / 5) + 1) % 2
    mat = cv2.GaussianBlur(edg, (kernel_size, kernel_size), 0)
    pct = 0.6
    thrsh = sorted(mat.flatten())[int(pct * len(mat.flatten()))]

    binary = np.zeros(mat.shape, dtype=int)
    args = np.argwhere(mat > thrsh)
    for arg in args:
        binary[arg[0], arg[1]] = 1

    plt.imshow(binary)
    plt.title("Binary")
    plt.show()

    thinned = skeletonize(binary)

    plt.imshow(thinned)
    plt.title("Thinned")
    plt.show()

    thinned = thinned.astype(np.uint8)

    return thinned


def find_direction_range(lines, img, show_all=False, name=None):
    line_vecs = []

    def find_intersection(line_vec1, line_vec2):
        auf1 = line_vec1[0]
        vec1 = line_vec1[1]
        auf2 = line_vec2[0]
        vec2 = line_vec2[1]

        x1 = auf1[0]
        y1 = auf1[1]
        x3 = auf2[0]
        y3 = auf2[1]
        x2 = auf1[0] + vec1[0]
        y2 = auf1[1] + vec1[1]
        x4 = auf2[0] + vec2[0]
        y4 = auf2[1] + vec2[1]

        m1 = np.array([[x1, y1], [x2, y2]])
        m2 = np.array([[x1, 1], [x2, 1]])
        m3 = np.array([[x3, y3], [x4, y4]])
        m4 = np.array([[x3, 1], [x4, 1]])

        m5 = np.array([[y1, 1], [y2, 1]])
        m6 = np.array([[y3, 1], [y4, 1]])

        det = lambda x: np.linalg.det(x)
        px_o = np.array([[det(m1), det(m2)], [det(m3), det(m4)]])
        p_u = np.array([[det(m2), det(m5)], [det(m4), det(m6)]])
        px = det(px_o) / det(p_u)

        py_o = np.array([[det(m1), det(m5)], [det(m3), det(m6)]])
        py = det(py_o) / det(p_u)

        return np.array([px, py])

    for line in lines:
        rho = line[0]
        theta = line[1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho

        auf = np.array([x0, y0])
        vec = np.array([-b, a])
        line_vecs.append((auf, vec))

    intersects = []

    # Find baseline: Minimum distance of points

    marked_img = copy.deepcopy(img)

    for i in range(len(lines)):
        rho1 = lines[i][0]
        theta1 = lines[i][1]
        for j in range(i + 1, len(lines)):
            rho2 = lines[j][0]
            theta2 = lines[j][1]

            if theta1 == theta2:
                continue
            intersects.append(find_intersection(line_vecs[i], line_vecs[j]))
    show_all = False
    if show_all:
        for intersect in intersects:
            x, y = intersect
            x0 = int(x)
            y0 = int(y)

            cv2.line(marked_img, (x0 - 1, y0), (x0 + 1, y0), (0, 0, 255), 1)
            cv2.line(marked_img, (x0, y0 - 1), (x0, y0 + 1), (0, 0, 255), 1)

        plt.imshow(marked_img)
        plt.show()

    mini = np.infty
    minidx = (0, 1)

    for i in range(len(intersects)):
        for j in range(i + 1, len(intersects)):
            dist = np.linalg.norm(intersects[i] - intersects[j])
            if dist < mini:
                mini = dist
                minidx = (i, j)

    base_auf = intersects[minidx[0]]
    base_vec = intersects[minidx[1]] - intersects[minidx[0]]
    scan_vec = None

    skps = []
    tvs = []
    for i in range(len(intersects)):
        if i == minidx[0]:
            continue
        test_vec = intersects[i] - intersects[minidx[0]]
        skp = np.dot(test_vec, base_vec)
        skps.append(skp)
        tvs.append(test_vec)

    ortho = np.argmin(skps)
    scan_vec = tvs[ortho]

    # Check AspectRatio

    aspect = np.linalg.norm(scan_vec) / np.linalg.norm(base_vec)
    # Threshold aus D:\Dateien\KI_Speicher\Dokumentation\MessungAspectRatioFilterClassImageAnalysis.ods
    assert abs(aspect - 1.252) / 1.252 < 0.2, "Incorrect Aspect Ratio: {:.3f}, True: 1.252, Abw: {:.1f}%".format(aspect,
                                                                                                                 100 * abs(
                                                                                                                     aspect - 1.252) / 1.252)

    # if skps[ortho] > 1e-3:
    #     ortho_img = copy.deepcopy(img)
    #     print("Might be not orthogonal: ", skps[ortho])
    #     start = (int(base_auf[0]), int(base_auf[1]))
    #     end_base = (int(base_auf[0] + base_vec[0]), int(base_auf[1] + base_vec[1]))
    #     end_scan = (int(base_auf[0] + scan_vec[0]), int(base_auf[1] + scan_vec[1]))
    #     cv2.arrowedLine(ortho_img, start, end_base, (0, 0, 255), 1)
    #     cv2.arrowedLine(ortho_img, start, end_scan, (255, 255, 0), 1)
    #
    #     plt.imshow(ortho_img)
    #     plt.title("Blue: base, Orange: Scan")
    #     plt.show()
    #
    # print("SV: ", scan_vec)
    # print("IS: ", intersects)
    #
    return base_auf, base_vec, scan_vec


def scan_origami(img_norm, img_blr, base_auf, base_vec, scan_vec, f5=None, f6=None, show_all=False, visualize_folder=None, mode='median'):
    def bilinear_interpol(mat, pos):
        if not (0 <= pos[0] <= np.shape(mat)[0] - 1 and 0 <= pos[1] <= np.shape(mat)[1] - 1):
            raise IndexError
        x = pos[0]
        y = pos[1]
        xu = int(np.floor(x))
        xfrac = x - xu
        xo = int(np.ceil(x))
        yo = int(np.floor(y))
        yu = int(np.ceil(y))
        yfrac = y - yo

        sum = xfrac * (1 - yfrac) * mat[xo, yo] + (1 - xfrac) * (1 - yfrac) * mat[xu, yo] + xfrac * (yfrac) * mat[
            xo, yu] + (1 - xfrac) * (yfrac) * mat[xu, yu]

        return sum

    if show_all:
        plt.imshow(img_norm)
        plt.title("Scan")
        plt.show()

    scanline_acc = 100
    line_samples = 20

    scan_dir = np.linspace(0, 1, scanline_acc)
    cross_dir = np.linspace(0, 1, line_samples)

    # Use bilinear interpol
    averages = []

    # print("Base Auf: ", base_auf)
    # print("Base vec: ", base_vec)
    # print("Scan Vec: ", scan_vec)
    total_pts = []
    frameno = 0
    for lambda_y in tqdm(scan_dir, desc="Scanning", disable=scanline_acc * line_samples < 1e6):
        line_pts = []
        vals = []


        for lambda_x in cross_dir:
            pt = base_auf + lambda_y * scan_vec + lambda_x * base_vec
            total_pts.append(pt)
            line_pts.append(pt)

            try:
                vals.append(bilinear_interpol(img_norm, pt))
            except IndexError as e:
                pass

        if mode == 'average':
            averages.append(np.average(vals))
        elif mode == 'median':
            averages.append(np.median(vals))
        else:
            raise Exception(f"unknown scan mode {mode} != average, median")


        if visualize_folder is not None:
            plt.imshow(img_norm)
            plt.scatter([p[0] for p in line_pts], [p[1] for p in line_pts])
            plt.xlim(0, img_norm.shape[0])
            plt.ylim(0, img_norm.shape[1])
            plt.title(f"Line Scan: {averg}")
            plt.savefig(os.path.join(visualize_folder, f"Image{str(frameno).zfill(5)}"))
            plt.cla()
            plt.clf()
            frameno += 1

    if show_all:
        plt.imshow(img_norm)
        plt.scatter([p[0] for p in total_pts], [p[1] for p in total_pts])
        plt.xlim(0, img_norm.shape[0])
        plt.ylim(0, img_norm.shape[1])
        plt.title("Total Scan")
        plt.show()

    if f5 is not None:
        plt.plot(scan_dir, averages)
        plt.title("Scanned Image Norm")
        # plt.show()
        plt.savefig(f5)
        plt.clf()

    if show_all:
        plt.plot(scan_dir, averages)
        plt.title("Scanned Image Norm")
        plt.show()

    averages_blur = []
    for lambda_y in tqdm(scan_dir, desc="Scanning", disable=scanline_acc * line_samples < 1e6):
        averg = 0

        for lambda_x in cross_dir:
            pt = base_auf + lambda_y * scan_vec + lambda_x * base_vec
            averg += bilinear_interpol(img_blr, pt) / line_samples
        averages_blur.append(averg)

    if f6 is not None:
        plt.plot(scan_dir, averages_blur)
        plt.title("Scanned Image Blur")
        plt.savefig(f6)
        plt.clf()
    if show_all:
        plt.plot(scan_dir, averages_blur)
        plt.title("Scanned Image Blur")
        plt.show()

    return averages, averages_blur


# Not Working
def remap_scan(averages, base_auf, base_vec, scan_vec, image):
    # plt.plot(averages)
    # plt.show()
    lbd = 4
    avg_scale = len(averages)
    resc = lambda x: (1 / np.exp(lbd)) * (np.exp(lbd * x) - 1)

    def params_from_px(x, y):
        v = np.array([x, y]) - base_auf
        s = scan_vec
        b = base_vec
        sl = np.dot(s, s)
        bl = np.dot(b, b)
        lbd_s = (1 / sl) * (1 / (1 - (np.dot(s, b) ** 2) / (sl * bl))) * (
                    np.dot(s, v) - (np.dot(b, v) * np.dot(s, b)) / (bl))
        lbd_b = (np.dot(b, v) - lbd_s * np.dot(s, b)) / sl

        return lbd_s, lbd_b

    resc = np.vectorize(resc)
    averages = (averages - np.amin(averages)) / (np.amax(averages) - np.amin(averages))
    averages = resc(averages)
    # plt.plot(averages)
    # plt.show()
    img = np.zeros([image.shape[0], image.shape[1], 3])
    img[:, :, 0] = image / 510
    img[:, :, 1] = image / 510
    img[:, :, 2] = image / 510

    mat0 = np.array(copy.deepcopy(image)) / 512
    mat1 = np.array(copy.deepcopy(image)) / 512
    height_map = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lbd_b, lbd_s = params_from_px(j, i)
            if 0 <= lbd_b < 1 and 0 <= lbd_s < 1:
                idx = min(len(averages) - 1, int(round(avg_scale * lbd_s)))
                mat0[j, i] = lbd_b
                mat1[j, i] = lbd_s

    img[:, :, 0] = mat0.transpose()
    img[:, :, 1] = mat1.transpose()
    img[:, :, 2] += height_map / 2
    plt.imshow(img)
    plt.show()


def find_maxima(averages, scan_vec, base_auf=None, base_vec=None, file=None, f7=None, f8=None, show_all=False, mode='plain', upscale=None):

    if mode == 'plain':
        array = averages
    elif mode == 'convolve':
        convolve_kernel = [np.exp(-x ** 2) for x in [-2, -1, 0, 1, 2]]
        convolve_kernel /= np.sum(convolve_kernel)
        array = np.convolve(averages, convolve_kernel, mode='same')
    elif mode == 'convolve2':
        sig = 2
        convolve_kernel = [np.exp(-x ** 2 / sig) for x in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]]
        convolve_kernel /= np.sum(convolve_kernel)
        if show_all:
            plt.plot(convolve_kernel)
            plt.title("Convolutiion Kernel")
            plt.show()
        array = np.convolve(averages, convolve_kernel, mode='same')
    else:
        raise Exception("Unknown Mode for Maxima")



    # Left maxima
    left = len(array)
    right = 0
    #left = int(0.9 * len(array))
    #right = int(0.1 * len(array))

    for i in [x for x in range(1, len(array) - 1)]:
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            left = i
            break

    for i in [x for x in range(1, len(array) - 1)][::-1]:
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            right = i
            break
    assert left <= right, "Not enough maxima found"

    distance = (right - left) * np.linalg.norm(scan_vec) / len(array)

    if show_all:
        plt.plot(array)
        plt.title(f"Finding Distance {mode}" )
        xmin, xmax, ymin, ymax = plt.axis()
        plt.plot([left, left], [ymin, ymax], color='r', linestyle='--')
        plt.plot([right, right], [ymin, ymax], color='r', linestyle='--')
        plt.show()

    if f7 is not None:
        plt.plot(array)
        plt.title(f"Finding Distance {mode} -> {distance}")
        xmin, xmax, ymin, ymax = plt.axis()
        plt.plot([left, left], [ymin, ymax], color='r', linestyle='--')
        plt.plot([right, right], [ymin, ymax], color='r', linestyle='--')
        plt.savefig(f7)

    if base_vec is not None and file is not None and f8 is not None:
        image = cv2.imread(file, 0)
        # print("Left: ", left)
        # print("Right: ", right)
        left /= len(averages)
        right /= len(averages)
        marker_left = base_auf + 0.5 * base_vec + left * scan_vec
        marker_right = base_auf + 0.5 * base_vec + right * scan_vec
        p1 = (int(round(marker_left[0])), int(round(marker_left[1])))
        p2 = (int(round(marker_right[0])), int(round(marker_right[1])))
        # print(p1, " - ", p2)
        plt.cla()
        plt.clf()
        cv2.arrowedLine(image, p1, p2, thickness=1, color=(255, 0, 0))

        plt.imshow(image, cmap='gray')
        plt.title("Distance: " + "{:.3f}px".format(distance))
        plt.savefig(f8)
        plt.clf()

    if upscale is not None and base_vec is not None and file is not None and f8 is not None:
            image = cv2.imread(file, 0)
            if upscale is not None and image.shape[0] < 64:
                scalefac = upscale / image.shape[0]
                image = cv2.resize(image, dsize=(upscale, upscale), interpolation=cv2.INTER_CUBIC)
            else:
                scalefac = 1.0
            # print("Left: ", left)
            # print("Right: ", right)
            marker_left = base_auf + 0.5 * base_vec + left * scan_vec
            marker_right = base_auf + 0.5 * base_vec + right * scan_vec
            marker_right *= scalefac
            marker_left *= scalefac
            p1 = (int(round(marker_left[0])), int(round(marker_left[1])))
            p2 = (int(round(marker_right[0])), int(round(marker_right[1])))
            # print(p1, " - ", p2)
            plt.cla()
            plt.clf()
            cv2.arrowedLine(image, p1, p2, thickness=1, color=(255, 0, 0))

            plt.imshow(image, cmap='gray')
            plt.title("Distance: " + "{:.3f}px".format(distance * scalefac))
            plt.savefig(f8[:-4] + "_upsc.png")
            plt.clf()

    return distance


if __name__ == "__main__":
    SHOW_ALL = False
    temp_folder = "temp_files"
    os.makedirs(temp_folder, exist_ok=True)
    ftps = ["real", "perfect", "ideal", "synth"]
    ftps = ["eval"]

    for ftp in ftps:
        if ftp == "synth":
            folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\TestSet_SSMeasure6\\Image"
        elif ftp == "ideal":
            folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\TestSet_SSMeasure2\\Image"
        elif ftp == "perfect":
            folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\PerfectClass"
        elif ftp == "test":
            folder = "D:\\Dateien\\KI_Speicher\\DNA\\Complete_Eval\\Test4"
        elif ftp == "eval":
            folder = "D:\\Dateien\\KI_Speicher\\DNA\\Eval_Folder"
        else:
            folder = "D:\\Dateien\\KI_Speicher\\DNA\\SS_TrainData\\EvaluateReal\\Ziba2308\\Crops\\Images"

        #folder = "D:\\Dateien\\KI_Speicher\\DNA\\Eval_Folder\\Image"
        #ftp = "eval"

        result_folder = os.path.join(folder, "Results_class", ftp)
        os.makedirs(result_folder, exist_ok=True)
        img_folder= os.path.join(folder, "image")

        resultfile = os.path.join(result_folder, "results2.csv")
        shutil.rmtree(resultfile, ignore_errors=True)
        res_f = open(resultfile, "a")
        # file = os.path.join(folder, random.choice(os.listdir(folder)))
        #pt = pretransform_img(file, temp_folder)
        #pt = npy2img(pt)
        #img, edg = canny(pt)

        files = [x for x in os.listdir(img_folder)]
        #if len(files) > 40:
        #    files = files[:40]
        for f in tqdm(files, desc="Analyzing images"):
            if f.endswith("csv"):
                continue
            dir = os.path.join(result_folder, f.split(".")[0])
            os.makedirs(dir, exist_ok=True)

            f1 = os.path.join(result_folder, f.split(".")[0], "1_ReadImg.png")
            f2 = os.path.join(result_folder, f.split(".")[0], "2_Canny.png")
            f3 = os.path.join(result_folder, f.split(".")[0], "3_Hough.png")
            f4 = os.path.join(result_folder, f.split(".")[0], "4_Filtered.png")
            f5 = os.path.join(result_folder, f.split(".")[0], "5_ScanNorm.png")
            f6 = os.path.join(result_folder, f.split(".")[0], "6_ScanBlur.png")
            f7 = os.path.join(result_folder, f.split(".")[0], "7_Distance.png")
            f8 = os.path.join(result_folder, f.split(".")[0], "8_Result.png")
            err = os.path.join(result_folder, f.split(".")[0], "99_Err.txt")
            file = os.path.join(img_folder, f)
            try:
                pt = pretransform_img(file, temp_folder)
                pt = npy2img(pt)
                img_blr, edg = canny(pt, f1, f2, show_all=SHOW_ALL)
                lines = hough(edg, f3=f3, f4=f4, no_lines=32, show_all=SHOW_ALL)
                base_auf, base_vec, scan_vec = find_direction_range(lines, img_blr, name=f.split(".")[0], show_all=SHOW_ALL)
                img_norm = pt
                # os.makedirs(os.path.join(result_folder, f.split(".")[0], "scanning"), exist_ok=True)
                averages, averages_blur = scan_origami(img_norm, img_blr, base_auf, base_vec, scan_vec, f5=f5, f6=f6,
                                                       show_all=SHOW_ALL, visualize_folder=None, mode='median')
                distance = find_maxima(averages, scan_vec, base_auf=base_auf,
                                       base_vec=base_vec, file=file, f7=f7, f8=f8, show_all=SHOW_ALL, mode='convolve')
                res_f.write(f + ";{:.6f}\n".format(distance))
            except Exception as e:
                if str(e).startswith("list"):
                    raise e
                with open(err, "w") as fexc:
                    traceback.print_exc(file=fexc)
                    print(e)
                    # raise e
                res_f.write(f + ";-1\n")


    shutil.rmtree(temp_folder)
    res_f.close()
