# -*- coding: utf-8 -*-
"""
Author Sharmitha Ganesan

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import cv2
import imutils
import math

cap = cv2.VideoCapture('1tagvideo.mp4')
out = cv2.VideoWriter('Testudo.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))

# FFT to subract the background to get the tag
def Fourier(gray):
    
        y1 = fft.fft2(gray)
        
        y2 = fft.fftshift(y1)
        
        (w, h) = gray.shape
        half_w, half_h = int(w/2), int(h/2)
        
        # high pass filter
        n = 5
        y2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0

        y3 = fft.ifftshift(y2)
        
        y4 = fft.ifft2(y3)
        
        y = np.uint8(np.abs(y4))
        
        return y
# Finding the corners of the tag from world co-ordinates
def contours(y):
    edged = cv2.Canny(y, 90, 300)
    contours = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
   
    #Finding the corners of the tag
    for cnt in contours:
        perimeter = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.09*perimeter,True)
        
        if cv2.contourArea(cnt) < 7000 and cv2.contourArea(cnt) > 500 and len(approx) == 4:
            
            apx = approx
            apx = apx.reshape(4,2)
            
            corn = np.zeros((4,2))
            
            add_cnt = apx.sum(axis=1)
            
            corn[0] = apx[np.argmin(add_cnt)]
            corn[2] = apx[np.argmax(add_cnt)]
            
            diff_cnt = np.diff(apx,axis=1)
            
            corn[1] = apx[np.argmin(diff_cnt)]
            corn[3] = apx[np.argmax(diff_cnt)]
           
            return corn
    
    
# Detecting the ID and Orientation
def orientation(newatag):
    
    gr =  cv2.cvtColor(newatag, cv2.COLOR_BGR2GRAY)
    
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
        
    #cv2.imshow('Tag',gr)
    
    return orient,data




def homography(cornlist,wlist):
    Alist = []
    for i in range(len(cornlist)):
        u, v = cornlist[i][0],cornlist[i][1]
        X, Y = wlist[i][0],wlist[i][1]
        Alist.append([X , Y , 1 , 0 , 0 , 0 , - X * u , - Y * u , - u])
        Alist.append([0 , 0 , 0 , X , Y , 1 , - X * v , - Y * v , - v]) 
    A = np.array(Alist)
  
    U, sigma, VT = np.linalg.svd(A)
    
    v= VT.T
    
    rv = v[:,8]/v[8][8]
    rv = rv.reshape((3,3))
    
    return rv

# Reorienting the Tag
def reorient():
    
    atag = [[0,0],[0,160],[160,160],[160,0]]
    newatag = np.zeros((160,160,3))
    #sh = newatag.shape
    #A = np.zeros(shape=(8,9))
    Alist = []
    for i in range(4):
        u, v = atag[i][0],atag[i][1]
        X, Y = corn[i][0],corn[i][1]
        Alist.append([X , Y , 1 , 0 , 0 , 0 , - X * u , - Y * u , - u])
        Alist.append([0 , 0 , 0 , X , Y , 1 , - X * v , - Y * v , - v]) 
        A = np.array(Alist)
    U, sigma, VT = np.linalg.svd(A)
    
    v= VT.T
    
    rv = v[:,8]/v[8][8]
    rv = rv.reshape((3,3))
    
    #r = homography(atag,reclist)
    
    rv_inv = np.linalg.inv(rv)
    
    for i in range(160):
        for j in range(160):
            wcoors=np.array([i,j,1])
            C = np.dot(rv_inv,wcoors)
            C = C/C[2]
            if (640 > C[0] > 0) and (480 > C[1] > 0):
                newatag[i][j] = input_img[int(C[1])][int(C[0])]
    
    newatag = newatag.astype(np.uint8)
    
    return newatag
    

while True:
    
   
    ret, frame = cap.read()
    
    if ret == True:
        
        input_img = cv2.resize(frame,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC) 
        b1 = input_img.copy()   
        b2 = input_img.copy()
        b3 = input_img.copy() 
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) 
        
        y = Fourier(gray)
        
        corn = contours(y)
        if corn is not None:
            cornlist = corn.tolist()
            cornlist2 = corn.tolist().copy()
            wlist = list()
            
            newatag = reorient()
            if newatag is not None:
                
                # Calling the Orientation Function
                Orientation,ID = orientation(newatag)
                
                if Orientation and ID is not None:
                    print('ID : ', ID)
                    print('Orientation : ', Orientation)
                    
                    #cv2.imshow('Frame',b)
                    
                    if Orientation == '270 Degrees':
                        wlist = [[500,0],[0,0],[0,500],[500,500]]
                        
                    elif Orientation == '0 Degrees':
                        wlist = [[0,0],[0,500],[500,500],[500,0]]
                        
                    elif Orientation == '90 Degrees':
                        wlist = [[0,500],[500,500],[500,0],[0,0]]
                        
                    elif Orientation == '180 Degrees':
                        wlist = [[500,500],[500,0],[0,0],[0,500]]
                    
                    rv1 = homography(cornlist,wlist)
                
                    #Placing Testudo on the Tag
                    img2 = cv2.imread('testudo.png')
                    img2 = cv2.resize(img2,(500,500),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    test = img2.shape
                    
                    new_img = np.zeros((test[0],test[1]))
                    
                    new_coor = []
                    
                    for i in range(test[0]):
                        for j in range(test[1]):
                            coor = np.array([[i],[j],[1]])
                            q = np.dot(rv1,coor)
                            new_x = q[0]/q[2]
                            new_y = q[1]/q[2]
                            pixel = img2[i][j]
                            new_coor.append([int(new_x),int(new_y),pixel])
                    
                    for i in range(len(new_coor)):
                        if (640 > new_coor[i][0] > 0) and (480 > new_coor[i][1] > 0):
                                input_img[new_coor[i][1]][new_coor[i][0]] = new_coor[i][2]
                    
                    cv2.imshow('TESTUDO',input_img)
                    
                    out.write(input_img)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):             
            break
    else:
        break
    
cv2.destroyAllWindows()   