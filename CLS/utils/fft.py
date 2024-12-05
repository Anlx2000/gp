import cv2
import numpy as np

def process_image(img, kernel_size=10):
    # 确保输入为单通道图像
    if img.ndim != 2:  # 如果不是单通道，抛出错误
        raise ValueError("输入图像必须是单通道图像")

    # 初始化高频和低频分量
    img_hp = []
    img_lp = []

    # 傅里叶变换
    f = np.fft.fft2(img)

    # 中心化频谱图
    fshift = np.fft.fftshift(f)

    # 构造高通和低通滤波器
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - kernel_size:crow + kernel_size, ccol - kernel_size:ccol + kernel_size] = 1

    # 对频谱图进行滤波
    fshift_hp = fshift * (1 - mask)
    fshift_lp = fshift * mask

    # 对高频分量进行傅里叶逆变换
    f_ishift_hp = np.fft.ifftshift(fshift_hp)
    img_hp_channel = np.fft.ifft2(f_ishift_hp/255)
    img_hp = np.abs(img_hp_channel)

    # 对低频分量进行傅里叶逆变换
    f_ishift_lp = np.fft.ifftshift(fshift_lp)
    img_lp_channel = np.fft.ifft2(f_ishift_lp/255)
    img_lp = np.abs(img_lp_channel)

    return img_hp, img_lp  # 返回高频和低频分量

# 读入图像
img = cv2.imread('./datasets/flowers102/images/image_00001.jpg', cv2.IMREAD_GRAYSCALE)

# 调用函数处理图像
img_hp, img_lp = process_image(img)

# 显示原图、低频分量和高频分量
cv2.imshow('Original', img)
cv2.imshow('Lowpass', img_lp)
cv2.imshow('Highpass', img_hp)

cv2.imshow('combined', img_hp + img_lp)

# 等待按下任意按键退出
cv2.waitKey(0)
cv2.destroyAllWindows()