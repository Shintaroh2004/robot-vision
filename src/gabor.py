from turtle import title
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gabor(img,ksize,sigma,theta,lambd,gamma,psi):
    kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lambd,gamma,psi)
    filtered_img=cv2.filter2D(img,cv2.CV_32F,kernel)
    return filtered_img,kernel

def bin_and_invert(img,thresh):
    _, bin_img=cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
    bin_img=cv2.bitwise_not(bin_img)
    return bin_img

def plot_img(orig,filt_imgs,kernels,titles):
    num_filters=len(filt_imgs)
    plt.figure(figsize=(20,10))
    plt.subplot(3,num_filters+1,1)
    plt.imshow(orig,cmap="gray")
    plt.title("original")
    plt.axis('off')

    for i,(img,kernel,title) in enumerate(zip(filt_imgs,kernels,titles)):
        plt.subplot(3,num_filters+1,i+2)
        plt.imshow(kernel,cmap="gray")
        plt.title(f"Gabor theta={title}")
        plt.axis("off")
        plt.subplot(3,num_filters+1,num_filters+i+3)
        plt.imshow(img,cmap="gray")
        plt.title(f"filterd theta={title}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gray_img=cv2.imread("hotei.jpg",cv2.IMREAD_GRAYSCALE)

    ksize=20 #検出される斜方成分が小さいほど増えた
    sigma=3.5 #下げると検出されるされる斜方成分が減った
    lambd=9.0 #フィルターの周波数が大きいほど減る
    gamma=0.5 #フィルターの幅が変化する
    psi=0
    angles=[70,100,120,165]
    thresh=180

    filt_imgs=[]
    kernels=[]
    titles=[]

    for angle in angles:
        theta=np.deg2rad(angle)
        filt_img,gabor_kernel=gabor(gray_img,ksize,sigma,theta,lambd,gamma,psi)
        bin_img=bin_and_invert(filt_img,thresh)
        filt_imgs.append(bin_img)
        kernels.append(gabor_kernel)
        titles.append(f"{angle}°")

    plot_img(gray_img,filt_imgs,kernels,titles)