
import cv2
import numpy
import torch

class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_folder, mask_folder, preprocess=None, transform=None):
        self.df = df
        self.image_folder, self.mask_folder = image_folder, mask_folder
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        image_path, mask_path = "{}/{}.jpg".format(self.image_folder, item.iloc[0]), "{}/{}.tif".format(self.mask_folder, item.iloc[0])
        image, mask = cv2.cvtColor(cv2.imread(image_path), code=cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(mask_path), code=cv2.COLOR_BGR2RGB)

        if self.preprocess is not None:
            preprocessed = self.preprocess(image=image, mask=mask)
            image, mask = preprocessed["image"], preprocessed["mask"]
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask