
"""
Author Sharmitha Ganesan
"""
import cv2

input = cv2.imread('D:\python\spyder\data\cityscape1.png')
gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray,0,0)
cv2.imshow('canny',edge)
cv2.imwrite('canny.jpg',edge)
cv2.waitKey(0)
cv2.destroyAllWindows()