import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_frequency_spectrum(img):
    """
    获取图像的频谱
    """
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # 傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    return magnitude_spectrum, fshift

def analyze_frequency_bands(spectrum, fshift):
    """
    分析不同频段的能量分布
    """
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # 调整频段划分，增加高频范围
    r_ultra_low = min(h, w) // 128   # 超低频（中心1/64）
    r_very_low = min(h, w) // 64    # 极低频（1/64-1/32）
    r_low = min(h, w) // 32         # 低频（1/32-1/16）
    r_mid_low = min(h, w) // 16      # 中低频（1/16-1/8）
    r_mid_high = min(h, w) // 8     # 中高频（1/8-1/6）
    r_high = min(h, w) // 4         # 高频（1/6-1/4）
    
    # 创建频段掩码
    y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    dist_from_center = np.sqrt(x*x + y*y)
    
    # 计算各频段的能量
    ultra_low_mask = dist_from_center <= r_ultra_low                           # 超低频（中心1/64）
    very_low_mask = (dist_from_center > r_ultra_low) & (dist_from_center <= r_very_low)  # 极低频（1/64-1/32）
    low_mask = (dist_from_center > r_very_low) & (dist_from_center <= r_low)  # 低频（1/32-1/16）
    mid_low_mask = (dist_from_center > r_low) & (dist_from_center <= r_mid_low)  # 中低频（1/16-1/8）
    mid_high_mask = (dist_from_center > r_mid_low) & (dist_from_center <= r_mid_high)  # 中高频（1/8-1/6）
    high_mask = (dist_from_center > r_mid_high) & (dist_from_center <= r_high)  # 高频（1/6-1/4）
    very_high_mask = dist_from_center > r_high                                # 超高频（>1/4）
    
    energy = {
        'ultra_low': np.abs(fshift[ultra_low_mask]).mean(),
        'very_low': np.abs(fshift[very_low_mask]).mean(),
        'low': np.abs(fshift[low_mask]).mean(),
        'mid_low': np.abs(fshift[mid_low_mask]).mean(),
        'mid_high': np.abs(fshift[mid_high_mask]).mean(),
        'high': np.abs(fshift[high_mask]).mean(),
        'very_high': np.abs(fshift[very_high_mask]).mean()
    }
    
    return energy, (ultra_low_mask, very_low_mask, low_mask, mid_low_mask, mid_high_mask, high_mask, very_high_mask)

def compare_frequency_spectrums(original_path, corruption_dir):
    """
    比较原始图像和各种噪声图像的频谱
    """
    # 读取原始图像
    original = cv2.imread(original_path)
    print(original_path)
    original_spectrum, original_fshift = get_frequency_spectrum(original)
    original_energy, masks = analyze_frequency_bands(original_spectrum, original_fshift)
    
    # 读取所有噪声图像
    corruption_images = {}
    corruption_energies = {}
    for img_name in os.listdir(corruption_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(corruption_dir, img_name)
            img = cv2.imread(img_path)
            corruption_name = img_name.split('.')[0]
            spectrum, fshift = get_frequency_spectrum(img)
            corruption_images[corruption_name] = spectrum
            corruption_energies[corruption_name], _ = analyze_frequency_bands(spectrum, fshift)
    
    # 创建频谱图
    n_images = len(corruption_images) + 1
    rows = (n_images + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4*rows))
    axes = axes.ravel()
    
    # 显示原始图像的频谱
    im = axes[0].imshow(original_spectrum, cmap='gray')
    axes[0].set_title('Original Frequency Spectrum')
    axes[0].axis('off')
    
    # 显示各种噪声图像的频谱
    for i, (name, spectrum) in enumerate(corruption_images.items(), 1):
        axes[i].imshow(spectrum, cmap='gray')
        axes[i].set_title(f'{name} Frequency Spectrum')
        axes[i].axis('off')
    
    if n_images % 2 == 1:
        fig.delaxes(axes[-1])
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.savefig('./frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制频段能量对比图
    plt.figure(figsize=(12, 6))
    bands = ['ultra_low', 'very_low', 'low', 'mid_low', 'mid_high', 'high', 'very_high']
    x = np.arange(len(bands))
    width = 0.8 / (len(corruption_images) + 1)
    
    # 绘制原始图像的频段能量
    plt.bar(x, [original_energy[band] for band in bands], width, 
            label='Original', alpha=0.8)
    
    # 绘制各种噪声图像的频段能量
    for i, (name, energy) in enumerate(corruption_energies.items(), 1):
        plt.bar(x + i*width, [energy[band] for band in bands], width,
                label=name, alpha=0.8)
    
    plt.xlabel('Frequency Bands')
    plt.ylabel('Energy')
    plt.title('Frequency Band Energy Distribution')
    plt.xticks(x + width*len(corruption_images)/2, bands)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./frequency_bands_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印频段差异分析
    print("\n频段差异分析:")
    for name, energy in corruption_energies.items():
        print(f"\n{name} vs Original:")
        for band in bands:
            diff = energy[band] - original_energy[band]
            percent = (diff / original_energy[band]) * 100
            print(f"{band:>4} 频段差异: {diff:>8.2f} ({percent:>+6.1f}%)")

def adjust_frequency_energy(img, target_energy_ratio=1.0, band_mask=None):
    """
    调整指定频段的能量
    
    参数:
        img: 输入图像
        target_energy_ratio: 目标能量比例
        band_mask: 频段掩码
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # 调整指定频段的能量
    if band_mask is not None:
        fshift[band_mask] *= target_energy_ratio
    
    # 逆变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 归一化到[0, 255]
    img_back = ((img_back - img_back.min()) * (255.0 / (img_back.max() - img_back.min()))).astype(np.uint8)
    
    return img_back

def compare_and_adjust_frequency(original_path, corruption_dir, output_dir='./adjusted_imgs'):
    """
    比较并调整频域能量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始图像
    original = cv2.imread(original_path)
    original_spectrum, original_fshift = get_frequency_spectrum(original)
    original_energy, original_masks = analyze_frequency_bands(original_spectrum, original_fshift)
    
    # 读取噪声图像并调整能量
    for img_name in os.listdir(corruption_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(corruption_dir, img_name)
            img = cv2.imread(img_path)
            corruption_name = img_name.split('.')[0]
            
            # 获取频谱
            spectrum, fshift = get_frequency_spectrum(img)
            energy, masks = analyze_frequency_bands(spectrum, fshift)
            
            # 调整每个频段的能量
            adjusted_img = img.copy()
            for band, mask in zip(['ultra_low', 'very_low', 'low', 'mid_low', 'mid_high', 'high', 'very_high'], 
                                original_masks):
                target_ratio = original_energy[band] / (energy[band] + 1e-10)  # 避免除零
                adjusted_img = adjust_frequency_energy(adjusted_img, target_ratio, mask)
            
            # 保存调整后的图像
            output_path = os.path.join(output_dir, f'adjusted_{corruption_name}.png')
            cv2.imwrite(output_path, adjusted_img)
            
            # 显示对比图
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(132)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.imshow(original, cmap='gray')  # 直接显示灰度图像
            plt.title(f'Corrupted ({corruption_name})')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
            plt.title('Adjusted')
            plt.axis('off')
            
            plt.savefig(os.path.join(output_dir, f'comparison_{corruption_name}.png'))
            plt.close()

if __name__ == '__main__':
    original_path = '../datasets/flowers102/images/image_00001.jpg'
    corruption_dir = './corruption_imgs'
    
    # 运行频谱分析和能量调整
    compare_frequency_spectrums(original_path, corruption_dir)
    compare_and_adjust_frequency(original_path, corruption_dir) 