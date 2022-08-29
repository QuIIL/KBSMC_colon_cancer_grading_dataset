import glob
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


####


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def print_data_count(label_list):
    count = []
    for i in range(5):
        count.append(label_list.count(i))
    count.append(len(label_list))
    return count


class DatasetSerial(data.Dataset):
    """get image by index
    """

    def __init__(self, pair_list, img_transform=None, target_transform=None, two_crop=False):
        self.pair_list = pair_list

        self.img_transform = img_transform
        self.target_transform = target_transform
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]
        image = pil_loader(path)

        # # image
        if self.img_transform is not None:
            img = self.img_transform(image)
        else:
            img = image

        return img, target



def prepare_colon_tma_data(
        data_root_dir='/media/trinh/Data0/data0/patches_data/KBSMC/Colon/colon_tma/COLON_PATCHES_1024/KBSMC_colon_tma_cancer_grading_512'):
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    set_tma01 = load_data_info('%s/tma01/*.jpg' % data_root_dir)
    set_tma02 = load_data_info('%s/tma02/*.jpg' % data_root_dir)
    set_tma03 = load_data_info('%s/tma03/*.jpg' % data_root_dir)
    set_tma04 = load_data_info('%s/tma04/*.jpg' % data_root_dir)
    set_tma05 = load_data_info('%s/tma05/*.jpg' % data_root_dir)
    set_tma06 = load_data_info('%s/tma06/*.jpg' % data_root_dir)
    set_wsi01 = load_data_info('%s/wsi01/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi02 = load_data_info('%s/wsi02/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi03 = load_data_info('%s/wsi03/*.jpg' % data_root_dir)  # benign exclusively

    train_set = set_tma01 + set_tma02 + set_tma03 + set_tma05 + set_wsi01
    valid_set = set_tma06 + set_wsi03
    test_set = set_tma04 + set_wsi02

    # print dataset detail
    train_label = [train_set[i][1] for i in range(len(train_set))]
    val_label = [valid_set[i][1] for i in range(len(valid_set))]
    test_label = [test_set[i][1] for i in range(len(test_set))]

    print(print_data_count(train_label))
    print(print_data_count(val_label))
    print(print_data_count(test_label))
    return train_set, valid_set, test_set


def prepare_colon_wsi_data(data_root_dir='./KBSMC_colon_45wsis_cancer_grading_512 (Test 2)'):
    """ List all the images and their labels
        return train_set, valid_set, test_set 2
    """

    def load_data_info_from_list(data_dir, path_list):
        file_list = []
        for WSI_name in path_list:
            pathname = glob.glob(f'{data_dir}/{WSI_name}/*/*.png')
            file_list.extend(pathname)
            label_list = [int(file_path.split('_')[-1].split('.')[0]) - 1 for file_path in file_list]
        print(Counter(label_list))
        list_out = list(zip(file_list, label_list))
        return list_out

    wsi_list = ['wsi_001', 'wsi_002', 'wsi_003', 'wsi_004', 'wsi_005', 'wsi_006', 'wsi_007', 'wsi_008', 'wsi_009',
                'wsi_010', 'wsi_011', 'wsi_012', 'wsi_013', 'wsi_014', 'wsi_015', 'wsi_016', 'wsi_017', 'wsi_018',
                'wsi_019', 'wsi_020', 'wsi_021', 'wsi_022', 'wsi_023', 'wsi_024', 'wsi_025', 'wsi_026', 'wsi_027',
                'wsi_028', 'wsi_029', 'wsi_030', 'wsi_031', 'wsi_032', 'wsi_033', 'wsi_034', 'wsi_035', 'wsi_090',
                'wsi_092', 'wsi_093', 'wsi_094', 'wsi_095', 'wsi_096', 'wsi_097', 'wsi_098', 'wsi_099', 'wsi_100']

    test_set = load_data_info_from_list(data_root_dir, wsi_list)
    return test_set


def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]  # gray to RGB heatmap
                aux = (aux * 255).astype('unint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size



