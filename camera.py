from picamera2 import Picamera2, Preview, MappedArray
from pprint import pprint
from datetime import datetime
import time
import cv2
from typing import Sequence
from dataclasses import dataclass
import numpy as np
from PIL import Image



DEFAULT_FPS = 5
FACEDETECTION_FILTER_PATH = './filters/haarcascade_frontalface_default.xml'

@dataclass
class Face:
    x: int
    y: int
    width: int
    height: int

@dataclass
class Dimensions: 
    height: int
    width: int


def create_camera() -> Picamera2:
    camera = Picamera2()
    camera_config = camera.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"},
                                      lores={"size": (320, 240), "format": "YUV420"})
    camera.configure(camera_config)
    return camera

def get_camera_dimensions(camera: Picamera2) -> Dimensions:
    (width, height) = camera.stream_configuration("main")["size"]
    return Dimensions(height=height, width=width)

def capture_frame(camera: Picamera2) -> np.ndarray:
    return camera.capture_array("main")
    
def save_image(arr: np.ndarray) -> None:
    ts = datetime.utcnow().timestamp()
    img = Image.fromarray(arr)
    img.save(f'./test_images/{ts}.png')



def capture_video_frames(
    camera: Picamera2, 
    fps: int = DEFAULT_FPS
) -> None:
    if fps == 0:
        raise Exception("Cannot have 0 frames per second!")

    camera.start()
    camera_dimensions = get_camera_dimensions(camera)
    while True:
        captured_array = capture_frame(camera)
        save_image(captured_array)
        gray_array = cv2.cvtColor(captured_array, cv2.COLOR_BGR2GRAY)
        save_image(gray_array)
        faces = face_detector.detectMultiScale(gray_array, 1.1, 3)
        print("FACE FOUND" if len(faces) > 0 else "NO FACE")
        time.sleep(1/fps)


# def draw_faces(request, faces: Sequence[Face]):
#     with MappedArray(request, "main") as m:
#             for f in faces:
#                 (x, y, w, h) = [c * n // d for c, n, d in zip(f, (w0, h0) * 2, (w1, h1) * 2)]
#                 cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0))


if __name__ == '__main__':
    face_detector = cv2.CascadeClassifier(FACEDETECTION_FILTER_PATH)
    camera = create_camera()
    dimensions = get_camera_dimensions(camera)
    print(dimensions)
    capture_video_frames(camera)
