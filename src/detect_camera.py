import cv2
from controller.ObjectDetectionController import ObjectDetectionController

object_detection_controller = ObjectDetectionController()


def process_frame(frame):
    detections = object_detection_controller.detect(frame)
    return object_detection_controller.draw_bounding_boxes(frame, detections)


def detect_on_camera():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open camera")
        return

    while True:
        ret, frame = capture.read()
        result = process_frame(frame)
        cv2.imshow('pedestrian detection', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_on_camera()
