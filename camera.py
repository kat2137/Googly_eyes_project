from picamera2 import Picamera2, Preview
from pprint import pprint
from datetime import datetime
import time
import cv2


DEFAULT_FPS = 5
FACEDETECTION_FILTER_PATH = './filters/haarcascade_frontalface_default.xml'

def create_camera() -> Picamera2:
    camera = Picamera2()
    camera_config = camera.create_preview_configuration()
    camera.configure(camera_config)
    return camera

def capture_frame(camera: Picamera2):
    ts = datetime.utcnow().timestamp()
    return camera.capture_file(f"./test_images/{ts}.jpeg")
    

def capture_video_frames(
    camera: Picamera2, 
    fps: int = DEFAULT_FPS
):
    if fps == 0:
        raise Exception("Cannot have 0 frames per second!")

    camera.start()
    while True:
        capture_frame(camera)
        time.sleep(1/fps)


face_detector = cv2.CascadeClassifier(FACEDETECTION_FILTER_PATH)


if __name__ == '__main__':
    camera = create_camera()
    capture_video_frames(camera)
