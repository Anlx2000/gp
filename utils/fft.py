import cv2
import numpy as np

# 读入图像
img = cv2.imread('./datasets/flowers102/images/image_00001.jpg', cv2.IMREAD_GRAYSCALE)

# 傅里叶变换
f = np.fft.fft2(img)

# 中心化频谱图
fshift = np.fft.fftshift(f)

# 构造高通和低通滤波器
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
size = 20  # 修改这个值来调整mask的大小
mask = np.zeros((rows, cols), np.uint8)
mask[crow - size:crow + size, ccol - size:ccol + size] = 1

# 对频谱图进行滤波
fshift_hp = fshift * (1 - mask)
fshift_lp = fshift * mask

# 对高频分量进行傅里叶逆变换
f_ishift_hp = np.fft.ifftshift(fshift_hp)
img_hp = np.fft.ifft2(f_ishift_hp/255)
img_hp = np.abs(img_hp)
print(img_hp)

# 对低频分量进行傅里叶逆变换
f_ishift_lp = np.fft.ifftshift(fshift_lp)
img_lp = np.fft.ifft2(f_ishift_lp/255)
img_lp = np.abs(img_lp)

# 显示原图、低频分量和高频分量
cv2.imshow('Original', img)
cv2.imshow('Lowpass', img_lp)
cv2.imshow('Highpass', img_hp)


cv2.imshow('combined', img_hp + img_lp)


# 等待按下任意按键退出
cv2.waitKey(0)
cv2.destroyAllWindows()