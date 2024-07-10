from email.mime import image
import cv2
from matplotlib import axis
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("lena.jpg")
img_blur=cv2.bilateralFilter(img,d=15,sigmaColor=75,sigmaSpace=75)

sp_radius_list=[10,20,30,40]
cl_radius_list=[5,20,50,70]
max_lv=1

img_segmented=[]

for sp_radius,cl_radius in zip(sp_radius_list,cl_radius_list):
    img_segmented.append(cv2.pyrMeanShiftFiltering(img_blur,sp_radius,cl_radius,maxLevel=max_lv))

fig,axes=plt.subplots(1,5,figsize=(6,4),sharex=True,sharey=True)
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("original")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(img_segmented[0], cv2.COLOR_BGR2RGB))
axes[1].set_title(f"segmented sp:{sp_radius_list[0]},cl:{cl_radius_list[0]}")
axes[1].axis("off")

for i in range(2,5):
    axes[i].imshow(cv2.cvtColor(img_segmented[i-1], cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"segmented sp:{sp_radius_list[i-1]},cl:{cl_radius_list[i-1]}")
    axes[i].axis("off")

plt.show()