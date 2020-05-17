import glob
import time
import datetime
import cv2
from controller.ObjectDetectionController import ObjectDetectionController

IMAGES_IN_PATH = 'assets/images/*'
IMAGES_OUT_PATH = 'output/images/'
object_detection_controller = ObjectDetectionController()


def detect_on_images():
    paths = glob.glob(IMAGES_IN_PATH)
    images = [cv2.imread(path) for path in paths]
    for image, path in zip(images, paths):
        start_time = time.time()
        detections = object_detection_controller.detect(image)
        end_time = time.time()
        inference_time = datetime.timedelta(seconds=end_time - start_time)
        print("Image: " + path)
        print("Inference Time: ", inference_time)
        result = object_detection_controller.draw_bounding_boxes(image, detections)
        filename = path.split("\\")[-1]
        cv2.imwrite(IMAGES_OUT_PATH + filename, result)


if __name__ == '__main__':
    detect_on_images()
