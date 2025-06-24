"""Custom dataset implementation for UAV detection."""
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Custom dataset for UAV vs background classification."""
    
    def __init__(self, uav_dir, background_dir, transform=None):
        """
        Initialize dataset.
        
        Args:
            uav_dir (str): Directory containing UAV images
            background_dir (str): Directory containing background images
            transform: Optional transform to be applied to images
        """
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir)]
        self.background_images = [os.path.join(background_dir, img) 
                                 for img in os.listdir(background_dir)]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label