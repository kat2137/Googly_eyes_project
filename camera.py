from picamera2 import Picamera2, Preview, MappedArray
from pprint import pprint
from datetime import datetime
import time
import cv2
from typing import Sequence
from dataclasses import dataclass
import numpy as np
from PIL import Image
from functools import total_ordering

DEFAULT_FPS = 5
FACEDETECTION_FILTER_PATH = './filters/haarcascade_frontalface_default.xml'

@dataclass
@total_ordering
class Face:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> int:
        return self.width * self.height
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.width == other.width and self.height == other.height
    
    def __lt__(self, other):
        return self.size < other.size


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

def detect_faces(face_detector, gray_array: np.ndarray) -> Sequence[Face]:
    faces = face_detector.detectMultiScale(gray_array, 1.05, 3)
    if not len(faces): return []
    print('FACE FOUND')
    return [Face(x, y, width, height) for (x, y, width, height) in faces]

def draw_faces(image_array: np.ndarray, faces: Sequence[Face]) -> np.ndarray:
    for idx, face in enumerate(sorted(faces, reverse=True)):
        if idx == 0:
            cv2.rectangle(image_array, (face.x, face.y), (face.x + face.width, face.y + face.height), (0 , 255, 0, 0))
        else: 
            cv2.rectangle(image_array, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0, 0))


def capture_video_frames(
    camera: Picamera2, 
    fps: int = DEFAULT_FPS
) -> None:
    if fps == 0:
        raise Exception("Cannot have 0 frames per second!")

    camera.start()
    while True:
        captured_array = capture_frame(camera)
        gray_array = cv2.cvtColor(captured_array, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_detector, gray_array)
        draw_faces(captured_array, faces)
        if faces:
            save_image(captured_array)
        time.sleep(1/fps)


if __name__ == '__main__':
    face_detector = cv2.CascadeClassifier(FACEDETECTION_FILTER_PATH)
    camera = create_camera()
    dimensions = get_camera_dimensions(camera)
    capture_video_frames(camera)
