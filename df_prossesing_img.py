#!/usr/bin/env python
# coding: utf-8

# In[28]:


import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class MyDataset(Dataset):
    def __init__(self, image_folder, transpose=True):
        self.image_folder = image_folder
        self.transpose = transpose
        
        # Create a list of labels and image file paths
        self.labels = []
        self.image_files = []
        for label in os.listdir(image_folder):
            label_folder = os.path.join(image_folder, label)
            if os.path.isdir(label_folder):
                for image_file in os.listdir(label_folder):
                    if image_file.endswith('.png'):
                        self.labels.append(label)
                        self.image_files.append(os.path.join(label_folder, image_file))
                        
        # Create a label encoder
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load the image
        image = cv2.imread(self.image_files[index])
        
        # Pre-process the image
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)
        image = image / 255.0
        
        # Transpose the image if necessary
        if self.transpose:
            image = np.transpose(image, (2, 0, 1))
            
        # Load the label
        label = self.labels[index]
        return image, label


# In[ ]:




