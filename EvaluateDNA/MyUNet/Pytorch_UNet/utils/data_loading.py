import logging
import random
from os import listdir
from os.path import splitext
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

cuda0 = torch.device('cuda:0')

f = open("defect.txt", "w")

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', image_num=None, noise_level=0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.noise_level = noise_level
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if image_num is not None:
            self.ids = self.ids[:image_num]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # name = self.ids[0]
        # mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        # print("MF: ")
        # print("Mask Dir: ", self.masks_dir)
        # print("Mask scheme: ", name + self.mask_suffix + '.*')
        # print(mask_file)
        # print(len(mask_file))

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            if not np.amax(img_ndarray) <= 1:
                img_ndarray = np.nan_to_num(img_ndarray)
                img_ndarray /= np.amax(img_ndarray)
                # plt.imshow(img_ndarray)
                # plt.show()

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename, allow_pickle=True))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))


        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        # except Exception as e:
        #     print(f"Error for image {img_file}")
        #     print(e)
        #     return self.__getitem__(idx+1)





        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size} -> {img_file} {mask_file}'


         #plt.imshow(np.array(img))
         #plt.title(np.array(img).shape)
         #plt.show()
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        # if np.amin(mask) < 0:
        #     # plt.imshow(mask)
        #     # plt.title(str(mask_file[0]).split("\\")[-1])
        #     # plt.show()
        #     f = open("defect.txt", "a")
        #     f.write(str(mask_file[0]) + "\n")
        #     f.close()
        #     return self.__getitem__(idx + 1)

        # Augmentation
        if True:
            if random.random() < 0.5:
            #    print("UD")
                img = img[:,::-1, :]
                mask = mask[::-1, :]
            if random.random() < 0.5:
             #   print("LR")
                img = img[:,:, ::-1]
                mask = mask[:, ::-1]
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(img[0])
            # axs[1].imshow(mask)
            # plt.title("After")
            # plt.show()
            # print(img.shape)
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(img[0])
            img_noise = img + np.random.normal(0, np.random.uniform(0, self.noise_level), img.shape)
            # axs[1].imshow(img_noise[0])
            # plt.title(img.shape)
            # axs[2].imshow(img_noise[1])
            # plt.show()
            # img_noise = np.clip(img_noise, 1, None)
            img_noise -= np.amin(img_noise)
            img_noise /= np.amax(img_noise)
             #axs[2].imshow(img_noise[0])
             #plt.show()
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(img[0])
            # axs[1].imshow(mask)
            # plt.title("After Noise")
            # plt.show()



        return {
            'image': torch.as_tensor(img_noise, dtype=torch.float32).contiguous(), # ToDo: cuda?
            'mask': torch.as_tensor(mask.copy(),  dtype=torch.long).contiguous()
        }


class BasicDatasetOLD(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', image_num=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if image_num is not None:
            self.ids = self.ids[:image_num]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # name = self.ids[0]
        # mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        # print("MF: ")
        # print("Mask Dir: ", self.masks_dir)
        # print("Mask scheme: ", name + self.mask_suffix + '.*')
        # print(mask_file)
        # print(len(mask_file))

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename, allow_pickle=True))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))


        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        # except Exception as e:
        #     print(f"Error for image {img_file}")
        #     print(e)
        #     return self.__getitem__(idx+1)





        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size} -> {img_file} {mask_file}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        # if np.amin(mask) < 0:
        #     # plt.imshow(mask)
        #     # plt.title(str(mask_file[0]).split("\\")[-1])
        #     # plt.show()
        #     f = open("defect.txt", "a")
        #     f.write(str(mask_file[0]) + "\n")
        #     f.close()
        #     return self.__getitem__(idx + 1)

        # Augmentation
        if True:
            if random.random() < 0.5:
                #    print("UD")
                img = img[:, ::-1, :]
                mask = mask[::-1, :]
            if random.random() < 0.5:
                #   print("LR")
                img = img[:, :, ::-1]
                mask = mask[:, ::-1]
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(img[0])
            # axs[1].imshow(mask)
            # plt.title("After")
            # plt.show()
            # print(img.shape)
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(img[0])
            img[0, :, :] += np.random.normal(0, 0.0005, img[0].shape)
            img = np.clip(img, 0, None)            # axs[1].imshow(img_noise[0])
            # plt.title(img.shape)
            # axs[2].imshow(img_noise[1])
            # plt.show()
            # img_noise = np.clip(img_noise, 1, None)
            # img_noise -= np.amin(img_noise)
            # img_noise /= np.amax(img_noise)
             #axs[2].imshow(img_noise[0])
             #plt.show()
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(img[0])
            # axs[1].imshow(mask)
            # plt.title("After Noise")
            # plt.show()

        return {
            'image': torch.as_tensor(img, dtype=torch.float32).contiguous(), # ToDo: cuda?
            'mask': torch.as_tensor(mask.copy(),  dtype=torch.long).contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


class EvalDataset(BasicDataset):
    def __init__(self, images_dir: str, scale: float = 1.0):
        super().__init__(images_dir, images_dir, scale, "")
        self.images_dir = Path(images_dir)
        self.masks_dir = None
        self.mask_suffix = ""

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = self.load(img_file[0])

        img = self.preprocess(img, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(0 * img.copy()).long().contiguous()
        }

class EvalDatasetOLD(BasicDatasetOLD):
    def __init__(self, images_dir: str, scale: float = 1.0):
        super().__init__(images_dir, images_dir, scale, "")
        self.images_dir = Path(images_dir)
        self.masks_dir = None
        self.mask_suffix = ""

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = self.load(img_file[0])

        img = self.preprocess(img, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(0 * img.copy()).long().contiguous()
        }