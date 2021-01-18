from torchvision import datasets, transforms

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms as T
from imgaug import augmenters as iaa
import pandas as pd
import pathlib
from data import *

class_names = [
"Nucleoplasm", 
"Nuclear membrane",   
"Nucleoli",   
"Nucleoli fibrillar center" ,  
"Nuclear speckles",
"Nuclear bodies",
"Endoplasmic reticulum",   
"Golgi apparatus",
"Peroxisomes",
"Endosomes",
"Lysosomes",
"Intermediate filaments",   
"Actin filaments",
"Focal adhesion sites",   
"Microtubules",
"Microtubule ends",   
"Cytokinetic bridge",   
"Mitotic spindle",
"Microtubule organizing center",  
"Centrosome",
"Lipid droplets",   
"Plasma membrane",   
"Cell junctions", 
"Mitochondria",
"Aggresome",
"Cytosol",
"Cytoplasmic bodies",   
"Rods & rings" 
]


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0) 
        np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
        
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

class ProteinDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle, validation_split, num_workers, num_classes, img_size, training=True):
        self.images_df = pd.read_csv(csv_path)
        self.num_classes = num_classes
        self.dataset = ProteinDataset(self.images_df, data_dir, num_classes, img_size, not training, training)
        self.n_samples = len(self.dataset)
        super(ProteinDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

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

        return train_sampler, valid_sampler

class ProteinDataset(Dataset):
    def __init__(self, images_df, base_path, num_classes, img_size, augument=True, training=True):
        base_path = pathlib.Path(base_path)
        self.img_size = img_size
        self.num_classes = num_classes
        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.training = training

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        X = self.read_images(index)
        if self.training:
            labels = self.read_labels(index)
            y = np.eye(self.num_classes, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(),T.Resize((225, 225)), T.ToTensor()])(X)
        return X.float(), y

    def read_labels(self, index):
        return np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = np.zeros(shape=(self.img_size, self.img_size, 4))
        r = np.array(data_transforms(Image.open(filename + "_red.png")))
        g = np.array(data_transforms(Image.open(filename + "_green.png")))
        b = np.array(data_transforms(Image.open(filename + "_blue.png")))
        y = np.array(data_transforms(Image.open(filename + "_yellow.png")))
        images[:, :, 0] = r.astype(np.uint8)
        images[:, :, 1] = g.astype(np.uint8)
        images[:, :, 2] = b.astype(np.uint8)
        images[:, :, 3] = y.astype(np.uint8)
        images = images.astype(np.uint8)
        return images

    def augumentor(self, image):
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
    
    
if __name__ == '__main__':
    dictio = {'data_dir': "./human-protein-atlas-image-classification/train",
            'csv_path': "./human-protein-atlas-image-classification/train.csv",
            'img_size': 512,
            'batch_size': 1,
            'shuffle': True,
            'validation_split': 0.15,
            'num_workers': 0,
            'num_classes': 28}
    data_loader = ProteinDataLoader(**dictio)
    from data import *
    
    data_loader = data_transforms(data_loader)
    def display_image(image, ax):
        [a.axis('off') for a in ax]
        r, g, b, y = image
        ax[0].imshow(r,cmap='Reds')
        ax[0].set_title('Microtubules')
        ax[1].imshow(g,cmap='Greens')
        ax[1].set_title('Protein of Interest')
        ax[2].imshow(b,cmap='Blues')
        ax[2].set_title('Nucleus')
        ax[3].imshow(y,cmap='Oranges') 
        ax[3].set_title('Endoplasmic Reticulum')
        return ax

    from PIL import Image
    # Get a batch of training data
    # inputs contains 4 images because batch_size=4 for the dataloaders
    inputs, classes = next(iter(data_loader))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    fig, ax = plt.subplots(figsize=(15,5),nrows=1, ncols=4)
    display_image(out, ax);
    
    plt.show()