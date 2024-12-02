from scipy.io import loadmat
import os

# 加载.mat文件

image_path = './images'
images_dir = os.listdir(image_path)
print(max(images_dir), min(images_dir))
data = loadmat('./labels/imagelabels.mat')

labels = data['labels'][0]
with open('./labels/flower_labels.txt', 'w') as f:
    for label in labels:
        f.write(str(label) + '\n')

print(len(labels))

# 打印加载的数据
print(data)