import cv2
import numpy as np
from PIL import Image
import skimage 

from scipy import signal
import matplotlib.pyplot as plt
def getPolarPoint(img,m,n): # the idea is checking row by row to calculate how many white pixel in the image/ if no white pixel=> sum=0
  sumx = 0
  polar1 =0
  polar2 =m-1
  while sumx ==0: 
    sumx = np.sum(img[polar1])
    polar1+=1
  sumx = 0
  while sumx ==0: 
    sumx = np.sum(img[polar2])
    polar2-=1
  return polar1, polar2
def getFilterPolarPoint(img,m,n): # similar to getPolarPoint but applied for filter
  sumx = 0
  polar1 =-1
  polar2 =m-1
  while (sumx <1) and (polar1<=m-1): 
    sumx = img[polar1]
    polar1+=1

  sumx = 0
  while (sumx <1) and (polar2>=0): 
    sumx = img[polar2]
    polar2-=1
  return polar1, polar2

def getEdgedImg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sharper the image
    sharpFilter = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
    print(sharpFilter.shape)
    grayconv = signal.convolve2d(gray, sharpFilter, mode='same')
    graySharper = gray - grayconv
    graySharper = skimage.exposure.rescale_intensity(gray,out_range=(0,255)).astype(np.uint8)


    blur = cv2.GaussianBlur(graySharper,(5,5),0)
    edged = cv2.Canny(blur, 0, 100)

    kernel = np.ones((5, 5), np.uint8)
    dialation = cv2.dilate(edged, kernel, iterations = 5)
    blur = cv2.GaussianBlur(dialation ,(11,11),0)
    edged = cv2.Canny(blur, 0, 100)
    
    return edged
# img = cv2.imread('image_3.jpg')
# img = cv2.resize(img,[1000,1000],cv2.INTER_CUBIC) # scaling
# edged = getEdgedImg(img)
# plt.imshow(edged,cmap='gray')
# plt.show()

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # img = cv2.resize(img,[1000,1000],cv2.INTER_CUBIC) # scaling
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()