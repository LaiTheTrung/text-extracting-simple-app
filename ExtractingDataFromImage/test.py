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
weir_imgs = ["image_unwrapped\D0.jpg","image_unwrapped\D1.jpg","image_unwrapped\D2.jpg","image_unwrapped\D3.jpg","image_unwrapped\D4.jpg","image_unwrapped\D5.jpg","image_unwrapped\D6.jpg","image_unwrapped\D7.jpg","image_unwrapped\D8.jpg"]
sticher = StitchingImage(weir_imgs)
panorama = sticher.getStiching()
plt.imshow(panorama)
plt.show()