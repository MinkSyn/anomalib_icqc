import os

from PIL import Image
from torch.utils.data import Dataset

from tool import is_image_file
from const import AnomalyID
from torchvision import transforms


class PatchCoreDataset(Dataset):
    """ Loader dataset from root path for train & test

    Args:
        root (str): path of dataset
        split (str): recognize between train or test
        icqc2ano (dict(list(str))): definition classes of PatchCore
        
    Return:
        list[tuple(Tensor, str, int)]: dataset contains (image, path, target)
    """

    def __init__(
        self,
        root,
        split = None,
        icqc2ano = None,
        transforms= None,
    ):
        super().__init__()
        self.root = root
        self.icqc2ano = icqc2ano
        self.transforms = transforms

        if split is not None:
            assert split in ['train', 'test']

        self.samples = self.get_dataset(split)

    def get_dataset(self, split):
        if split == 'train':
            image_paths = []
            for file in os.listdir(os.path.join(self.root)):
                filename = os.fsdecode(file)
                if is_image_file(filename):
                    path = os.path.join(os.path.join(self.root), filename)
                    image_paths.append((path, AnomalyID['normal'].value))

        elif (split == 'test') and (self.icqc2ano is not None):
            lst_normal = self.icqc2ano['normal']
            lst_abnormal = self.icqc2ano['abnormal']
            if lst_normal == []:
                normal_paths = []
            else:
                normal_paths = self.get_image_test(lst_normal,
                                                   AnomalyID['normal'].value)
            abnormal_paths = self.get_image_test(lst_abnormal,
                                                 AnomalyID['abnormal'].value)
            image_paths = normal_paths + abnormal_paths
        return image_paths

    def get_image_test(self, lst_class,
                       id_class):
        image_paths = []
        for cls_quality in lst_class:
            for name_folder in os.listdir(self.root):
                if cls_quality in name_folder:
                    path_quality = os.path.join(self.root, name_folder)
                    for file in os.listdir(path_quality):
                        filename = os.fsdecode(file)
                        if is_image_file(filename):
                            path = os.path.join(path_quality, filename)
                            image_paths.append((path, id_class))
        return image_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, path, target
