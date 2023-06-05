import numpy as np
import cv2
import stitching
import unwrap_labels
import os 
import pandas as pd
from PIL import Image
import skimage 
from scipy import signal
import matplotlib.pyplot as plt
from operator import itemgetter
import shutil
import torch, torchvision
import mmdet
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmocr.apis import MMOCRInferencer,TextDetInferencer
import stitching
from stitching.image_handler import ImageHandler
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.seam_finder import SeamFinder
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.timelapser import Timelapser
from stitching.seam_finder import SeamFinder
from stitching.cropper import Cropper
from stitching.blender import Blender
import matplotlib.pyplot as plt

class StitchingImage:
    def __init__(self,listname) -> None:
        self.listname = listname
    def getStiching(self):    
        weir_imgs = self.listname
        img_handler = ImageHandler()
        img_handler.set_img_names(weir_imgs)
        medium_imgs = list(img_handler.resize_to_medium_resolution())
        low_imgs = list(img_handler.resize_to_low_resolution(medium_imgs))
        final_imgs = list(img_handler.resize_to_final_resolution())
        original_size = img_handler.img_sizes[0]
        medium_size = img_handler.get_image_size(medium_imgs[0])
        low_size = img_handler.get_image_size(low_imgs[0])
        final_size = img_handler.get_image_size(final_imgs[0])
        #----------find features-----------
        
        finder = FeatureDetector()
        features = [finder.detect_features(img) for img in medium_imgs]
        keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])
        #----------Match features----------

        matcher = FeatureMatcher()
        matches = matcher.match_features(features)

        all_relevant_matches = matcher.draw_matches_matrix(medium_imgs, features, matches, conf_thresh=1.5, 
                                                        inliers=True, matchColor=(0, 255, 0))
        #---------select subset------------

        subsetter = Subsetter()
        dot_notation = subsetter.get_matches_graph(img_handler.img_names, matches)
        indices = subsetter.get_indices_to_keep(features, matches)

        medium_imgs = subsetter.subset_list(medium_imgs, indices)
        low_imgs = subsetter.subset_list(low_imgs, indices)
        final_imgs = subsetter.subset_list(final_imgs, indices)
        features = subsetter.subset_list(features, indices)
        matches = subsetter.subset_matches(matches, indices)

        img_names = subsetter.subset_list(img_handler.img_names, indices)
        img_sizes = subsetter.subset_list(img_handler.img_sizes, indices)
        img_handler.img_names, img_handler.img_sizes = img_names, img_sizes
        #---------estimate camera parameters------------


        camera_estimator = CameraEstimator()
        camera_adjuster = CameraAdjuster()
        wave_corrector = WaveCorrector()

        #----------refine camera-----------
        cameras = camera_estimator.estimate(features, matches)
        cameras = camera_adjuster.adjust(features, matches, cameras)
        cameras = wave_corrector.correct(cameras)
        #COMPOSTING
        #-----wrap------

        warper = Warper()
        warper.set_scale(cameras)

        #----wrap low resolution-------------
        low_sizes = img_handler.get_low_img_sizes()
        camera_aspect = img_handler.get_medium_to_low_ratio()      # since cameras were obtained on medium imgs

        warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
        warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
        low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
        #----wrap final resolution-------------
        final_sizes = img_handler.get_final_img_sizes()
        camera_aspect = img_handler.get_medium_to_final_ratio()    # since cameras were obtained on medium imgs

        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
        warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
        final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)
        #------crop----------------


        cropper = Cropper()
        mask = cropper.estimate_panorama_mask(warped_low_imgs, warped_low_masks, low_corners, low_sizes)
        lir = cropper.estimate_largest_interior_rectangle(mask)
        plot = lir.draw_on(mask, size=2)
        low_corners = cropper.get_zero_center_corners(low_corners)
        rectangles = cropper.get_rectangles(low_corners, low_sizes)

        plot = rectangles[1].draw_on(plot, (0, 255, 0), 2)  # The rectangle of the center img
        overlap = cropper.get_overlap(rectangles[1], lir)
        plot = overlap.draw_on(plot, (255, 0, 0), 2)
        intersection = cropper.get_intersection(rectangles[1], overlap)
        plot = intersection.draw_on(warped_low_masks[1], (255, 0, 0), 2)
        cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

        cropped_low_masks = list(cropper.crop_images(warped_low_masks))
        cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
        low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

        lir_aspect = img_handler.get_low_to_final_ratio()  # since lir was obtained on low imgs
        cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
        cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
        final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

        timelapser = Timelapser('as_is')
        timelapser.initialize(final_corners, final_sizes)

        for img, corner in zip(cropped_final_imgs, final_corners):
            timelapser.process_frame(img, corner)
        #-------seam mask------------------


        seam_finder = SeamFinder()

        seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
        seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]

        seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(cropped_final_imgs, seam_masks)]
        #-----Exposure Error Compensation----------

        compensator = ExposureErrorCompensator()

        compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

        compensated_imgs = [compensator.apply(idx, corner, img, mask) 
                            for idx, (img, mask, corner) 
                            in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]
        #------blending----------------

        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()
        return panorama

class ImageExtracter():
    def __init__(self, img, cylinder = False) :
        self.imgs = img
        self.cylinder = cylinder
        self.percent_points = []
        self.img_highResolution = cv2.resize(img,[1000,1500],cv2.INTER_CUBIC)
        self.img_lowResolution = cv2.resize(img,[500,500],cv2.INTER_CUBIC) # scaling
    def Processing1Img(self,imgInput):
        img = imgInput.copy()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = 0   #hue
        h_max = 179 #hue
        s_min = 0   #sat
        s_max = 255 #sat
        v_min = 90  #val
        v_max = 255 #val
        lower= np.array([h_min,s_min,v_min])
        upper= np.array([h_max,s_max,v_max])
        mask = cv2.inRange(imgHSV,lower,upper)
        kernel = np.ones((5, 5), np.uint8)
        #diagnoise
        dialation = cv2.dilate(mask, kernel, iterations = 1)  
        erodesion = cv2.erode(dialation,kernel, iterations= 6)
        diagnoised = cv2.dilate(erodesion, kernel, iterations = 5)    
        blur = cv2.GaussianBlur(diagnoised,(5,5),0)
        edged = cv2.Canny(blur, 0, 100)
        return edged

    def GetPointfromImage(self,outside, img):
        # finding 2 polar point (y coordinate)
        # (because the bottle have cyclindrical shape 
        # so the line will be bending and have the highest/ lowest point
        img2 = np.zeros_like(img) 
        xp,yp1,w,h = cv2.boundingRect(outside)
        yp2 = yp1+h
        #finding 4 point of rectangle
        perimeter = cv2.arcLength(outside, True)
        approx = cv2.approxPolyDP(outside, 0.05 * perimeter, True)
        for point in approx:
            x, y = point[0]
            cv2.circle(img2, (x, y), 3, (255, 255, 255), -1)
        p1,p2,p3,p4 = approx.reshape(4,2)
        listpoints =[p1,p2,p3,p4]
        listpoints = sorted(listpoints, key=itemgetter(0))
        p1,p2 = sorted([listpoints[0],listpoints[1]], key=itemgetter(1))
        p3,p4 = sorted([listpoints[2],listpoints[3]], key=itemgetter(1))
        pup = [(int)((p1[0]+p3[0])/2) , yp1]
        pdown = [(int)((p2[0]+p4[0])/2) , yp2]
        cv2.circle(img2, (pup[0], yp1), 3, (0, 255, 0), -1)
        cv2.circle(img2, (pdown[0], yp2), 3, (0, 0, 255), -1)
        """
                In both cases points represent figure below:

            |        |                  |        |
            |    up   |                  A    B    C
            | /    \ |                  | /    \ |
            1        3                  |        |
            |        |                  |        |
            |        |       OR         |        |
            |        |                  |        |
            2        4                  F        D
            | \    / |                  | \    / |
            |   do   |                  |   E    |
            |        |                  |        |
        """
        #A,B,C,D,E,F,G
        return img2,p1,pup,p3,p4,pdown,p2
    
    def FindLabelRegion(self):
        edged = self.Processing1Img(self.img_lowResolution)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filterOut = np.zeros_like(self.img_lowResolution) # for flatten cylinder
        frameCnt =[]
        maxArea = 0
        for cnt in contours:
            centerPoint = np.mean(cnt)
            if (cv2.arcLength(cnt, True) >500) and (centerPoint > 200) and (cv2.contourArea(cnt) >20000) and (cv2.contourArea(cnt)>maxArea):
                    filterOut = np.zeros_like(self.img_lowResolution)
                    frameCnt.append( cnt )
                    maxArea = cv2.contourArea(cnt)
                    print(cv2.arcLength(cnt, True) )
                    print('centerPoint:',centerPoint)
                    cv2.fillPoly(filterOut,pts=[cnt],color=(255,255,255))   
        try:
            Filtercontours = frameCnt[-1]
        except:
            Filtercontours = 0
        try:
            img2,pA,pB,pC,pD,pE,pF = self.GetPointfromImage(outside=Filtercontours,img = filterOut)
            pixel_points=[pA,pB,pC,pD,pE,pF]
            for point in pixel_points:
                point[0]= point[0]*2 # 1000/500
                point[1]= point[1]*3
                self.percent_points.append(point)
            return img2
        except Exception as e:
            raise Exception("The image is unqualified")
            
    def ProcessingImage(self):
            if (self.cylinder):
                try:
                    unwrapper = unwrap_labels.LabelUnwrapper(src_image=self.img_highResolution, pixel_points=self.percent_points)
                    dst_image = unwrapper.unwrap()
                    self.ProcessedOutput = dst_image
                except:
                    print('please take another images dataset')
            else:
                pA,pB,pC,pD,pE,pF = self.percent_points
                width = 520
                height = 520
                pt1 = np.float32([pA, pC, pF, pD])
                pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                matrix = cv2.getPerspectiveTransform(pt1, pt2)
                dst_image = cv2.warpPerspective(self.img_highResolution, matrix, (width, height))
                #croping the region
                dst_image = dst_image[10:510, 10:510,:]
                self.ProcessedOutput = dst_image
                

def ConnectData(imgs,cylinder=False):
    if (cylinder):
        if os.path.exists("image_unwrapped"):
            # remove if exists
            shutil.rmtree("image_unwrapped")
        listname = []
        stitcher = stitching.Stitcher()
        os.mkdir('image_unwrapped')
        for i,img in enumerate(imgs):
            name = f"image_unwrapped\D{i}.jpg"
            plt.imsave(name, img)
            listname.append(name)
        sticher = StitchingImage(listname)
        panorama = sticher.getStiching()
        shutil.rmtree("image_unwrapped")
        return panorama
    elif (not cylinder):
        concatenateImg= np.concatenate(imgs, axis=1)
        return concatenateImg

class OCR_extracter():
    def __init__(self,img):
        self.colorImg = img.copy()
        self.grayImg = np.zeros_like(img)
        self.grayImg[:,:,0] = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.grayImg[:,:,1] = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.grayImg[:,:,2] = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.dict_words = []
    def TextRecognition(self):
        kie = MMOCRInferencer(det='DBNet', rec='SAR')
        output = kie(self.grayImg, show=False)
        _polylines = output['predictions'][0]['det_polygons']
        words = output['predictions'][0]['rec_texts']
        acc = output['predictions'][0]['rec_scores']
        mask = np.zeros_like(self.grayImg)+255
        for i,pts in enumerate(_polylines):
            if (acc[i] < 0.8):
                continue
            pts = [int(x) for x in pts]
            points = np.array([pts[0:2],pts[2:4],pts[4:6],pts[6:8]])
            org = [np.min(points,axis=0)[0]+2,np.max(points,axis=0)[1]-2]
            word = words[i]
            dict_word = {'coordinate:':points.tolist(),'word:':word}
            self.dict_words.append(dict_word)
            mask = cv2.polylines(mask,[points],isClosed=True,color=[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
            mask = cv2.putText(mask, word, org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        return mask
    
    def TextDetection(self):
        model = TextDetInferencer(model='TextSnake', device='cuda')
        AIoutput = model(self.grayImg, show=False)
        _polylines = AIoutput['predictions'][0]['polygons'] # => the output will the the contours of image
        shapes = np.zeros_like(self.grayImg)
        output = self.grayImg.copy()
        for cnt in _polylines:
            n = len(cnt)
            cnt = np.array(cnt,dtype=int).reshape(int(n/2),2)
            x,y,w,h = cv2.boundingRect(cnt)
            shapes = cv2.rectangle(shapes,(x,y),(x+w,y+h),(255,255,255),-1)
        alpha = 0.5
        mask = shapes<125
        output[mask] = cv2.addWeighted(self.colorImg, alpha, shapes, 1 - alpha, 0)[mask]
        return output
    
    def ExtractToCSVfile(self):
        df = pd.DataFrame.from_dict(self.dict_words) 
        df.to_csv (r'data.csv', index=False, header=True)

