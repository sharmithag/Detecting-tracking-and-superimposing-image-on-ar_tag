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


out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'XVID'), 15, (640,480))

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
    
    edged = cv2.Canny(y, 90, 600)
        
    contours = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
   
    #Finding the corners of the tag
    for cnt in contours:
        perimeter = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.09*perimeter,True)
        
        if cv2.contourArea(cnt) < 3000 and cv2.contourArea(cnt) > 500 and len(approx) == 4:
            
            apx = approx
            apx = apx.reshape(4,2)
            corn = np.zeros((4,2))
            sum_cnt = apx.sum(axis=1)
            corn[0] = apx[np.argmin(sum_cnt)]
            corn[2] = apx[np.argmax(sum_cnt)]
            diff_cnt = np.diff(apx,axis=1)
            corn[1] = apx[np.argmin(diff_cnt)]
            corn[3] = apx[np.argmax(diff_cnt)]
            break
    return corn

#Calculating Homography Matrix
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

# Projecting the CUBE onto the AR TAG by calculating the Projection Matrix
def projection(rv5,frame,points):
    
    k = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])

    bnew = np.dot(np.linalg.inv(k), rv5)
    
    b1 = bnew[:, 0].reshape(3, 1)
    b2 = bnew[:, 1].reshape(3, 1)
    
    r3 = np.cross(bnew[:, 0], bnew[:, 1])
    b3 = bnew[:, 2].reshape(3, 1)
    
    L = 2 / (np.linalg.norm((np.linalg.inv(k)).dot(b1)) + np.linalg.norm((np.linalg.inv(k)).dot(b2)))
    
    r1 = L * b1
    r2 = L * b2
    r3 = (r3 * L * L).reshape(3, 1)
    t = L * b3
    r = np.concatenate((r1, r2, r3, t), axis=1)
    
    P = np.dot(k, r)
    cs = np.dot(P,points.T)
    
    i1, j1, k1 = cs[:,0]
    i2, j2, k2 = cs[:,1]
    i3, j3, k3 = cs[:,2]
    i4, j4, k4 = cs[:,3]
    i5, j5, k5 = cs[:,4]
    i6, j6, k6 = cs[:,5]
    i7, j7, k7 = cs[:,6]
    i8, j8, k8 = cs[:,7]

    #Drawing lines through the co-ordinates
    cv2.line(frame,( int(i1/k1), int(j1/k1)),( int(i2/k2), int(j2/k2)), (255,0,0), 2)
    cv2.line(frame,( int(i2/k2), int(j2/k2)),(  int(i3/k3), int(j3/k3)), (255,0,0), 2)
    cv2.line(frame,( int(i3/k3), int(j3/k3)),( int(i4/k4), int(j4/k4)), (255,0,0), 2)
    cv2.line(frame,( int(i4/k4), int(j4/k4)),( int(i1/k1), int(j1/k1)), (255,0,0), 2)

    cv2.line(frame,( int(i1/k1), int(j1/k1)),( int(i5/k5), int(j5/k5)), (255,0,0), 2)
    cv2.line(frame,( int(i2/k2), int(j2/k2)),( int(i6/k6), int(j6/k6)), (255,0,0), 2)
    cv2.line(frame,( int(i3/k3), int(j3/k3)),( int(i7/k7), int(j7/k7)), (255,0,0), 2)
    cv2.line(frame,( int(i4/k4), int(j4/k4)),( int(i8/k8), int(j8/k8)), (255,0,0), 2)

    cv2.line(frame,( int(i5/k5), int(j5/k5)),( int(i6/k6), int(j6/k6)), (255,0,0), 2)
    cv2.line(frame,( int(i6/k6), int(j6/k6)),( int(i7/k7), int(j7/k7)), (255,0,0), 2)
    cv2.line(frame,( int(i7/k7), int(j7/k7)),( int(i8/k8), int(j8/k8)), (255,0,0), 2)
    cv2.line(frame,( int(i8/k8), int(j8/k8)),( int(i5/k5), int(j5/k5)), (255,0,0), 2)
    
    return frame
    
 

while True:
    
    ret, frame = cap.read()
    
    if ret == True:
        
        b = cv2.resize(frame,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC) 
        b1 = b.copy()   
        b2 = b.copy()
        b3 = b.copy() 
        gray = cv2.cvtColor(b1, cv2.COLOR_BGR2GRAY) 
        
        y = Fourier(gray)
        
        #corn = contours(y)
        try:
            corn = contours(y)
        except:
            continue
        
        cornlist = corn.tolist()
        
        points = np.float32([[0,0,0,1],[0,1,0,1],[1,1,0,1],[1,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,1],[1,0,1,1]])
        
        rv2 = homography(cornlist,points) 

        image = projection(rv2,b1,points)
        
        cv2.imshow('Cube',image)
        
        out.write(b1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):           
            break
    else:
        break  

cv2.waitKey(1)
cv2.destroyAllWindows()
