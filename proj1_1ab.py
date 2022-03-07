# -*- coding: utf-8 -*-
"""
Author Sharmitha Ganesan

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import cv2


def FastFourier():
    
    cap = cv2.VideoCapture('1tagvideo.mp4')
    count = 0
    
    while True:
        
        ret, frame = cap.read()
        if ret is None:
            break
        
        count+=1
        if count == 100:
            break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    y1 = fft.fft2(gray)
    
    y2 = fft.fftshift(y1)
    
    (wt, ht) = gray.shape
    half_w, half_h = int(wt/2), int(ht/2)
    
    # high pass filter
    n = 3
    y2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
    
    y3 = fft.ifftshift(y2)
    
    y4 = fft.ifft2(y3)
    
    y = np.uint8(np.abs(y4))
    
    cv2.imshow('Image',y)
    edged = cv2.Canny(y, 90, 700)

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow('Tag Square and AR TAG',frame)
    
    #plt.imshow(gray)
    #plt.show()
    #plt.imshow(np.log(1+np.abs(y1)))
    #plt.show()
    plt.imshow(np.log(1+np.abs(y2)))
    plt.show()
    #plt.imshow(np.log(1+np.abs(y3)))
    #plt.show()
    plt.imshow(np.abs(y4))
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def orientation():
    img = cv2.imread('reference.png') #make sure the image format is written correctly
    
    gr =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    width, height = gr.shape
    
    #anti-clockwise
    a = gr[int(height/3.2)][int(height/3.2)]
    b = gr[int(height/1.4545)][int(height/3.2)]
    c = gr[int(height/1.4545)][int(height/1.4545)]
    d = gr[int(height/3.2)][int(height/1.4545)]
    
    e = gr[int(height/2.285)][int(height/2.285)] 
    f = gr[int(height/1.777)][int(height/2.285)] 
    g = gr[int(height/1.777)][int(height/1.777)] 
    h = gr[int(height/2.285)][int(height/1.777)] 
    
    orient = str()
    data = []
    
    if f > 200:#MSB
        f = 1
    else:
        f = 0
    if g > 200:
        g = 1
    else:
        g = 0
    if h > 200:
        h = 1
    else:
        h = 0
    if e > 200:#LSB
        e = 1
    else:
        e = 0
        
    #print('INNER GRID : ',e,f,g,h)
    
    #print('CORNERS : ',a,b,c,d)
    
    if c > 200:
        orient = '0 Degrees'
        data.append(f)
        data.append(g)
        data.append(h)
        data.append(e)
    elif b > 200:
        orient = '270 Degrees'
        data.append(e)
        data.append(f)
        data.append(g)
        data.append(h)
    elif a > 200:
        orient = '180 Degrees'
        data.append(h)
        data.append(e)
        data.append(f)
        data.append(g)
    elif d > 200:
        orient = '90 Degrees'
        data.append(g)
        data.append(h)
        data.append(e)
        data.append(f) 
        
    cv2.imshow('Tag',gr)
    
    print('ORIENTATION : ', orient)
    
    print('AR TAG ID : ', data)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

FastFourier() #problem1A
orientation() #problem1B