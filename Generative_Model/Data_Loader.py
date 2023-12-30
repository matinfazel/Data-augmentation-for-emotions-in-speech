import glob
import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.utils import make_grid,save_image
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor,step,target, num_images=25, size=(1, 28, 28)):

    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(f"/content/drive/MyDrive/Gan_Results/Comparison/{target}/Step{step}.png")  # Save the image
    plt.show()

def save_tensor_image(image_tensor,step,target):
    tmp = torch.squeeze(image_tensor)
    tmp = (tmp + 1) / 2
    save_image(tmp,f"/content/drive/MyDrive/Gan_Results/Fake_images/{target}/img_{step}.png")
    
load_shape = (640,480)
target_shape = (640,480)


transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])


dataset = ImageFolder("/content/drive/MyDrive/Spectogram/English/", transform=transform)
def FilteredDataset(dataset, target_class):
        # Filter the dataset based on selected classes
        classes = {
              'Angry':0,
              'Happy':1,
              'Neutral':2,
              'Sad':3,
              'Surprise':4
          }
        filtered_indices = [idx for idx in range(len(dataset)) if dataset[idx][1] == classes[target_class]]
        return torch.utils.data.Subset(dataset, filtered_indices)
