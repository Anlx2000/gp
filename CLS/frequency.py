import cv2    
import numpy as np      
    
img = cv2.imread("/Users/anlx/研究生/毕设/CLS/corruption_imgs/snow.png", 0)    
gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)    
dst = cv2.convertScaleAbs(gray_lap)    
    
cv2.imshow('laplacian',dst)    
cv2.waitKey(0)    
cv2.destroyAllWindows()  