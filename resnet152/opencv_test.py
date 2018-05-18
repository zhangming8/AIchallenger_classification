# -*- coding: utf-8 -*-
import numpy as np
import cv2

#img = cv2.imread('1.jpg',0) #以灰度读取
img = cv2.imread('1.jpg')
height, width = img.shape[:2]
print(img.shape, img.size, img.dtype)

# 调整图片大小
img_resize = cv2.resize(img,(width//2,height//2))
cv2.imshow('img_resize', img_resize)
height, width = img_resize.shape[:2]

#%% 旋转图片
## 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放比例
## 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
#M = cv2.getRotationMatrix2D((width//2,height//2), 20, 0.9) #计算旋转矩阵M
#dst = cv2.warpAffine(img_resize, M, (width,height)) #通过旋转矩阵M旋转img_resize，(width,height)为旋转后的是图像的（宽，高）。应该记住的是图像的宽对应的是列数，高对应的是行数。
#while 1:
#    cv2.imshow('img_rot', dst)
#    if cv2.waitKey(1)&0xFF==27:
#        break
#cv2.destroyAllWindows()

#%% 仿射变换（仿射变换时原图所有的平行线在结果图中仍然平行，矩形变为平行四边形）
#需要从原图像中找到三个点以及他们在输出图像中的位置
#pts1 = np.float32([[50,50],[200,50],[50,200]])
#pts2 = np.float32([[10,100],[200,50],[100,250]])
#M = cv2.getAffineTransform(pts1,pts2)
#dst = cv2.warpAffine(img_resize,M,(width,height))
#while 1:
#    cv2.imshow('fangshebianhuan',dst)
#    if cv2.waitKey(1)&0xFF==27:
#        break
#cv2.destroyAllWindows()

#%% 透视变换（
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img_resize,M,(300,300))
while 1:
    cv2.imshow('toushibianhuan',dst)
    if cv2.waitKey() ==27 & 0xFF:
        break
cv2.destroyAllWindows()

#img = cv2.resize(img,(300,400))
#cv2.imshow('pic1',img)
#k = cv2.waitKey(0) & 0xFF
#if k == 27: #k==27为esc键
#    cv2.destroyAllWindows()
#elif k == ord('s'): #当按下字母s时保存新图片
#    cv2.imwrite('new_1.jpg',img)
#    cv2.destroyAllWindows()
    
#cap = cv2.VideoCapture(0)
#while(True):
#    ret, frame = cap.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
#    cv2.imshow('video',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#cap.release()
#cv2.destroyAllWindows()

#height,width=img.shape[:2]
#res=cv2.resize(img,(width//2,height//2),interpolation=cv2.INTER_CUBIC)
#while(1):
#    cv2.imshow('res',res)
#    cv2.imshow('img',img)
#    if cv2.waitKey(1) & 0xFF == 27:
#        break
#cv2.destroyAllWindows()