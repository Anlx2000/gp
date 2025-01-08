import torch, cv2
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)
laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)
avgpool = torch.tensor([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)

sharpen = torch.tensor([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)


sharpen = torch.tensor([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

def conv_operator(filename, kernel, in_channels=1):
    if in_channels == 1:
        img = np.expand_dims(cv2.imread(filename, 0), 2)    # gray
    elif in_channels == 3:
        img = cv2.imread(filename, 1)                        # bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        exit()

    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    y = F.conv2d(x, kernel.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
    y = y.squeeze(0).numpy().transpose(1, 2, 0)

    return img, y

def plt_show(windowsname, img, channels=1):
    plt.figure(windowsname)
    if channels ==1:
        plt.imshow(img, cmap='gray')
    elif channels == 3:
        plt.imshow(img, )
    else:
        exit()

    plt.axis('on')
    plt.show()

if __name__=="__main__":
    img_name = '/Users/anlx/研究生/毕设/CLS/corruption_imgs/fog.png'

    img, y = conv_operator(img_name, sobel_x, 3)
    plt_show("input img", img, 3)
    plt_show("sobel_x", y)

    _, y = conv_operator(img_name, sobel_y, 1)
    plt_show("sobel_y", y)

    _, y = conv_operator(img_name, laplace, 3)
    plt_show("laplace", y)

    _, y = conv_operator(img_name, avgpool, 3)
    plt_show("avgpool", y)

    _, y = conv_operator(img_name, sharpen, 3)
    plt_show("sharpen", y)

