import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import os
import numpy as np
from PIL import Image



class flare_dataloader(object):
    def __init__(self, opt, transforms):
        dataset  = flare_data(opt, transforms)

        shuffle_dataset = True
        random_seed = 42
        test_split = 0.2
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, sampler = train_sampler)
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, sampler = test_sampler)

    def __call__(self):
        return self.train_loader, self.test_loader


class flare_data(Dataset):
    def __init__(self, opt, transforms):
        super(flare_data, self).__init__()

        self.data_path = opt.data_path
        self.image_names = os.listdir(self.data_path)
        self.transform = transforms

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.combined_image = Image.open(os.path.join(self.data_path, self.image_names[idx]))
        self.combined_image = np.array(self.combined_image)

        (h, w, c) = self.combined_image.shape

        self.flare_free_image = self.combined_image[:, :w//2, :]
        self.flare_image = self.combined_image[:, w//2:, :]

        if self.transform:
            self.flare_free_image = self.transform(self.flare_free_image)
            self.flare_image = self.transform(self.flare_image)

        return self.flare_image, self.flare_free_image


