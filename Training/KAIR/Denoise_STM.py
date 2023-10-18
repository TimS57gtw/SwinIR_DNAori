import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import Load_STM_Files

from PIL import Image
def load_file(fn, linespeed=1, tolerance=0.01):

    date = datetime.now()
    linespeed = 1

    ext = os.path.basename(fn).split('.')[-1]

    if ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        arr = np.array(Image.open(fn))
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]
    elif ext == 'sxm':
        sxm = Load_STM_Files.SXM(fn)
        arr = sxm.data
        linespeed = sxm.time_per_line
        date = sxm.date
        if sxm.scan_dir == 'up':
            arr = np.flipud(arr)
    elif ext.lower() == 'sm2':
        sm2 = Load_STM_Files.SM2(fn)
        arr = sm2.data
        linespeed = sm2.time_per_line
        date = sm2.date
        if sm2.scan_dir == 'up':
            arr = np.flipud(arr)
    elif ext.lower() == 'nid':
        nid = Load_STM_Files.NID(fn)
        arr = nid.data
        linespeed = nid.time_per_line
        date = nid.date
        if nid.scan_dir == 'up':
            arr = np.flipud(arr)

    else:
        raise NotImplementedError(f"Unknown Filetype: {ext}")
    afl  = arr.flatten()
    afl = sorted(afl)
    mini = afl[int(tolerance * len(afl))]
    maxi = afl[int((1-tolerance) * len(afl))]
    arr = arr.astype(float)
    arr -= mini
    arr /= maxi - mini
    arr = np.clip(arr, 0, 1)

    plt.imshow(arr, cmap='gray')
    plt.show()

    return arr, linespeed, date

def load_closest_spectrum(fld, date):

    def print_time_format(secs):
        if type(secs) is timedelta:
            secs = timedelta.total_seconds(secs)
        sign = np.sign(secs)
        secs /= sign
        ds = secs // (3600 * 24)
        secs = secs % (3600 * 24)
        hs = secs // 3600
        secs = secs % 3600
        mins = secs // 60
        secs = secs % 60

        ds = int(ds)
        hs = int(hs)
        mins = int(mins)
        secs = int(secs)

        ss = "-" if sign < 0  else "+"
        return f"{ss}{hs}:{mins}:{secs}" if ds == 0 else f"{ss} {ds}d {hs}:{mins}:{secs}"


    mode = 'nearest'
    files = []
    times = []
    diffs = []
    spcs = []

    for file in os.listdir(fld):
        spc = Load_STM_Files.Spectrum_dat(os.path.join(fld, file))
        files.append(file)
        times.append(spc.timestamp)

        timediff = (spc.timestamp - date).total_seconds()
        # print(f"{date} -> {spc.timestamp} --> {print_time_format(timediff)}")
        diffs.append(timediff)
        spcs.append(spc)

    if mode == 'closest_before':
        mind = diffs[0]
        cls = spcs[0]
        time = times[0]
        file = None
        for i in range(len(diffs)):
            if diffs[i] < 0 and diffs[i] > mind:
                mind = diffs[i]
                cls = spcs[i]
                time = times[i]
                file = files[i]


    elif mode == 'nearest':
        absdiff = np.infty
        cls = None
        time = None
        file = None
        for i in range(len(diffs)):
            if abs(diffs[i]) < absdiff:
                absdiff = abs(diffs[i])
                cls = spcs[i]
                time = times[i]
                file = files[i]

    else:
        raise NotImplementedError
    print("target_time: ", date)
    print(f"Found closest Spectrum at {time}: {print_time_format(time - date)}\nFile: {file}")

    plt.plot(cls.freq,cls.ampl)
    plt.xlabel(cls.f_head)
    plt.ylabel(cls.a_head)
    plt.show()

    return cls

def max_interp(x, oldx, oldy):
    ys = np.zeros_like(x, dtype=float)
    for ox, oy in zip(oldx, oldy):
        p = np.argmin(np.abs(ox - x))
        ys[p] = max(oy, ys[p])
    # plt.plot(x, ys, label='new')
    # plt.plot(oldx, oldy, label='old')
    # plt.legend()
    # plt.xlim(0, max(x))
    # plt.title("Max IP")
    # plt.show()

    return ys

def transform_spectrum(spec, linespeed):

    old_freqs = spec.freq
    print("Linespeed: ", linespeed)
    ampls = spec.ampl

    newfreqs = old_freqs / linespeed

    old_freqs = old_freqs[1:]
    newfreqs = newfreqs[1:]
    ampls = ampls[1:]


    plt.plot(old_freqs, ampls, label='old')
    plt.plot(newfreqs, ampls, label='new')
    plt.legend()
    plt.show()

    interp_freqs = np.arange(2, 129, 2)
    print(interp_freqs)

    interp_ampl = max_interp(interp_freqs, newfreqs, ampls)
    plt.plot(interp_freqs, interp_ampl)
    plt.show()

    return interp_freqs, interp_ampl

def apply_denoising(arr, spec, modelfld):
    raise NotImplementedError

    return den_arr

def save_sxm(arr, outf, linespeed=1.0, date=None):
    raise NotImplementedError

def subsample_image(arr, spec, ims_size=256):
    raise NotImplementedError

    return arr, spec


if __name__=="__main__":
    infld = r"D:\Dateien\KI_Speicher\SampleSXM\Test"
    specra_fld = os.path.join(infld, 'spectra')
    model = r"D:\Dateien\KI_Speicher\SwinSTM_Denoise\Runs\swinir_sr_CL_H128_W8_full"
    outfld = os.path.join(infld, 'denoised')
    os.makedirs(outfld, exist_ok=True)

    for file in os.listdir(infld):
        print("File: ", file)
        if not os.path.isfile(os.path.join(infld, file)):
            continue
        arr, linespeed, date = load_file(os.path.join(infld, file))
        spec = load_closest_spectrum(specra_fld, date)

        spec = transform_spectrum(spec, linespeed)
        continue
        arr, spec = subsample_image(arr, spec, ims_size=256)
        denoised = apply_denoising(arr, spec, model)
        save_sxm(denoised, os.path.join(outfld, file), linespeed, date)