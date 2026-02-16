import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def get_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

class SummerWinterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform

        self.trainA_path = os.path.join(root_dir, "trainA")
        self.trainB_path = os.path.join(root_dir, "trainB")

        self.trainA_images = os.listdir(self.trainA_path)
        self.trainB_images = os.listdir(self.trainB_path)

    def __len__(self):
        return max(len(self.trainA_images),
                   len(self.trainB_images))

    def __getitem__(self, index):
        # Summer Image
        img_A = Image.open(
            os.path.join(
                self.trainA_path,
                self.trainA_images[index % len(self.trainA_images)]
            )
        ).convert("RGB")

        # Winter Image (randomly sampled)
        img_B = Image.open(
            os.path.join(
                self.trainB_path,
                random.choice(self.trainB_images)
            )
        ).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}
    
def get_dataloader(root_dir,
                   image_size=256,
                   batch_size=1,
                   shuffle=True):

    transform = get_transforms(image_size)

    dataset = SummerWinterDataset(
        root_dir=root_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

    return dataloader