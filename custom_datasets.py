import os, re


from torchvision.datasets import ImageFolder
from utils import path2age

class ImageFolderWithAges(ImageFolder):
    """Custom dataset that includes face image age. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, pat, pos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pat = pat
        self.pos = pos
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithAges, self).__getitem__(index)
        age = path2age(self.imgs[index][0], self.pat, self.pos)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (age,))
        return tuple_with_path


class ImageFolderWithAgeGroup(ImageFolder):
    """Custom dataset that includes face image age group, categorized by [0-12, 13-18, 19-25,
    26-35, 36-45, 46-55, 56-65, >= 66]. Extends torchvision.datasets.ImageFolder
    """
    def __init__(self, pat, pos, cutoffs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pat = pat
        self.pos = pos
        self.cutoffs = cutoffs

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithAgeGroup, self).__getitem__(index)
        age = path2age(self.imgs[index][0], self.pat, self.pos)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (self.find_group(age),))
        return tuple_with_path

    def find_group(self, age):
        for i, cut in enumerate(self.cutoffs):
            if age <= cut:
                return i
        return len(self.cutoffs)
