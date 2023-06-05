# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(mmcv.__version__)
print(get_compiling_cuda_version())
print(get_compiler_version())

# Check mmocr installation
import mmocr
print(mmocr.__version__)

# Visualize the results
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import skimage 
from scipy import signal
from mmocr.apis import MMOCRInferencer,TextDetInferencer


def TextRecognition(miniImg):
    kie = MMOCRInferencer(det='DBNet', rec='SAR')
    output = kie(miniImg, show=False)
    _polylines = output['predictions'][0]['det_polygons']
    words = output['predictions'][0]['rec_texts']
    acc = output['predictions'][0]['rec_scores']
    mask = np.zeros_like(miniImg)+255
    for i,pts in enumerate(_polylines):
        if (acc[i] < 0.8):
            continue
        pts = [int(x) for x in pts]
        points = np.array([pts[0:2],pts[2:4],pts[4:6],pts[6:8]])
        org = [np.min(points,axis=0)[0]+2,np.max(points,axis=0)[1]-2]
        word = words[i]
        mask = cv2.polylines(mask,[points],isClosed=True,color=[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
        mask = cv2.putText(mask, word, org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    return mask

def TextDetection(img):
    model = TextDetInferencer(model='TextSnake', device='cuda')
    output = model(img, show=True)
    _polylines = output['predictions'][0]['polygons'] # => the output will the the contours of image
    mask = np.zeros_like(img)+255

    for pts in _polylines:

        n = len(pts)
        points = np.array(pts,dtype=int).reshape(int(n/2),2)
        x,y,w,h = cv2.boundingRect(points)
        miniMask = np.zeros((h*15,w*3,3))
        miniMask[5:h+5,5:w+5,:] = img[y:(y+h),x:x+w,:]
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        miniMask = TextRecognition(miniMask)
        mask[y:(y+h),x:x+w,:] = miniMask[5:h+5,5:w+5,:]

    return mask

image = cv2.imread('image_3.jpg')
image[:,:,0] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image[:,:,1] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image[:,:,2] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(image.shape)
mask = TextRecognition(image)
mask2 = TextDetection(image)
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.subplot(1, 3, 3)
plt.imshow(mask2)
plt.show()
# # mask = flattenImage(image)
# img = cv2.resize(image,[700,1000],cv2.INTER_CUBIC) # scaling
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # sharper the image
# # sharpFilter = np.array([[0, 1, 0],
# #                         [1, -4, 1],
# #                         [0, 1, 0]])
# # print(sharpFilter.shape)
# # grayconv = signal.convolve2d(gray, sharpFilter, mode='same')
# # graySharper = gray - grayconv
# # graySharper = skimage.exposure.rescale_intensity(graySharper,out_range=(0,255)).astype(np.uint8)
# graySharper = skimage.exposure.rescale_intensity(gray)
# print(np.max(graySharper),',',np.min(graySharper))

# #dialation and find edge of the image, here I use lare size filter kernel so it will get the outside frame better
# # # Creating kernel
# kernel = np.ones((10, 10), np.uint8)
# blur = cv2.GaussianBlur(graySharper,(5,5),0)
# edged = cv2.Canny(blur, 0, 100)

# # plt.subplot(1, 2, 1)
# # plt.imshow(image)
# # plt.subplot(1, 2, 2)
# # plt.imshow(graySharper,cmap='gray')
# # plt.show()




