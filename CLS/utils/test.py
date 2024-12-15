import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


img = Image.open('../../datasets/flowers102/images/image_00003.jpg')
img = transforms.ToTensor()(img)
img = transforms.Resize(224)(img)

plt.imshow(img.permute(1, 2, 0))
plt.show()