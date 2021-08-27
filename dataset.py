import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
# from scipy.ndimage import gaussian_filter



class XRayDataset(torch.utils.data.Dataset):
    # Classe que preprocessa i torna una sample del dataset (la sample idx) i la seva label/target.

    def __init__(self, image_folder):
        root = os.getcwd()

        self.image_folder = root + image_folder
        self.images = os.listdir(self.image_folder+'images/')
        self.targets = pd.read_csv(self.image_folder+'targets', dtype=object, names=['image', 'target'])
        self.size = 299



    # Agafar una mostra del dataset
    def __getitem__(self, idx):

        image_file = self.images[idx]

        # PREPROCESS USUAL
        image = Image.open((self.image_folder + '/images/' + image_file))
        image = image.convert(mode="L")
        # image = gaussian_filter(image, sigma=10)
        image = np.array(image)

        # Normalitzem la imatge
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean)/std

        # Convertim la imatge a tensor
        image = torch.Tensor(image).reshape(1, self.size, self.size)  # 1 perquè és només un canal (blanc i negre)

        # Buscar el target
        target = self.targets.iloc[idx]['target']  # 2
        target = int(target)
        # target = torch.Tensor(target)
        return image, target

    def __len__(self):
        return len(self.images)

class XRayDatasetResnet(torch.utils.data.Dataset):
    # Classe que preprocessa i torna una sample del dataset (la sample idx) i la seva label/target.

    def __init__(self, image_folder, preprocess):
        root = os.getcwd()
        self.preprocess = preprocess
        self.image_folder = root + image_folder
        self.images = os.listdir(self.image_folder+'images/')
        self.targets = pd.read_csv(self.image_folder+'targets', dtype=object, names=['image', 'target'])
        self.size = 294 # resize per resnet101


    # Agafar una mostra del dataset
    def __getitem__(self, idx):

        image_file = self.images[idx]

        # preprocess per resnet101
        image = Image.open((self.image_folder + '/images/' + image_file))
        image = image.convert(mode="RGB")
        image = self.preprocess(image)

        # Buscar el target
        target = self.targets.iloc[idx]['target']  # 2
        target = int(target)
        # target = torch.Tensor(target)
        return image, target

    def __len__(self):
        return len(self.images)

class UNetTrain(torch.utils.data.Dataset):
    # Classe que preprocessa i torna una sample del dataset (la sample idx) i la seva label/target.

    def __init__(self, image_folder):
        root = os.getcwd()

        self.image_folder = root + image_folder
        self.images = os.listdir(self.image_folder+'images/')
        self.targets = os.listdir(self.image_folder+'images/')
        self.size = 299


    # Agafar una mostra del dataset
    def __getitem__(self, idx):
        image_file = self.images[idx]

        image = Image.open((self.image_folder + '/images/' + image_file))
        image = image.convert(mode="L")
        image = np.array(image)

        # Normalitzem la imatge
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean)/std

        # Convertim la imatge a tensor
        image = torch.Tensor(image).reshape(1, self.size, self.size)  # 1 perquè és només un canal (blanc i negre)

        # Buscar el target
        target_file = self.targets[idx]
        target = Image.open((self.image_folder + '/images/' + target_file))
        target = target.resize((194, 194))
        target = target.convert(mode="L")
        target = np.array(target)

        mean = np.mean(target)
        std = np.std(target)
        target = (target - mean) / std
        target = torch.Tensor(target).reshape(1, 194, 194)

        return image, target

    def __len__(self):
        return len(self.images)