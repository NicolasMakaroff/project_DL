import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd


class ProteinDataLoader(Dataset):
    
    def __init__(self, images_df, path, nb_classes, img_size, batch_size = 32, augmentation = True, train = True):
        self.path = path 
        self.imgsize = img_size
        self.nbclasses = nb_classes
        self.images_df = pd.read_csv(path)
        self.augmentation = augmentation
        self.images_df.Id = self.images_df.Id.apply(lambda x: batch_size / x)
        self.train = train

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        X = self.read_images(index)
        
        if self.train:
            labels = self.read_labels(index)
            y = np.eye(self.num_classes, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
            
        if self.augmentation:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        return X.float(), y

    def read_labels(self, index):
        return np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = np.zeros(shape=(self.img_size, self.img_size, 4))
        r = np.array(Image.open(filename + "_red.png"))
        g = np.array(Image.open(filename + "_green.png"))
        b = np.array(Image.open(filename + "_blue.png"))
        y = np.array(Image.open(filename + "_yellow.png"))
        images[:, :, 0] = r.astype(np.uint8)
        images[:, :, 1] = g.astype(np.uint8)
        images[:, :, 2] = b.astype(np.uint8)
        images[:, :, 3] = y.astype(np.uint8)
        images = images.astype(np.uint8)
        return images

    def data_augmentation(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),

            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        
        return image_aug
        
    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        validation_split = []
        for idx, (value, count) in enumerate(self.images_df['Target'].value_counts().to_dict().items()):
            for _ in range(max(round(split * count), 1)):
                validation_split.append(value)

        validation_split_idx = []
        for idx, value in enumerate(self.images_df['Target']):
            try:
                validation_split.remove(value)
                validation_split_idx.append(idx)
            except:
                pass

        idx_full = np.arange(self.n_samples)
        valid_idx = np.array(validation_split_idx)
        train_idx = np.delete(idx_full, valid_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        
        return train_images, val_images
        
def create_train_val_split(images_df, frac = 0.2):
        
        n_samples = len(images_df) * (1 - frac)
        validation_split = []
        for idx, (value, count) in enumerate(images_df['Target'].value_counts().to_dict().items()):
            for _ in range(max(round(split * count), 1)):
                validation_split.append(value)

        validation_split_idx = []
        for idx, value in enumerate(images_df['Target']):
            try:
                validation_split.remove(value)
                validation_split_idx.append(idx)
            except:
                pass

        idx_full = np.arange(n_samples)
        valid_idx = np.array(validation_split_idx)
        train_idx = np.delete(idx_full, valid_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        
        return train_images, val_images
    
if __name__ == '__main__':
    
    prot = ProteinDataLoader('/human-protein-atlas-image-classification/train', 'human-protein-atlas-image-classification/train.csv', nb_classes = 28, img_size = 512, augmentation = False, train = True)