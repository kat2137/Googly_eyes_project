#!/usr/bin/env python3
from picamera2 import Picamera2
import time
import cv2
from typing import Sequence
from dataclasses import dataclass
import numpy as np
import smbus
import os

DEFAULT_FPS = 10
FACEDETECTION_FILTER_PATH = './filters/haarcascade_frontalface_default.xml'
PCA9685_ADDRESS = 0x40
SERVO_MIN = 205
SERVO_MAX = 410

current_pan = 90
current_tilt = 90

servo_test = True

def setup_pca9685(bus):
    bus.write_byte_data(PCA9685_ADDRESS, 0x00, 0x00) # Reset
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDRESS, 0x00, 0x10) # Sleep
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDRESS, 0xFE, 121)  # Set prescaler
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDRESS, 0x00, 0x00) # Wake
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDRESS, 0x00, 0xA1) # Enable auto-increment
    print("PCA9685 Servo Driver Initialized.")

def move_servo(bus, channel, angle):
    angle = max(0, min(180, angle))
    pulse = int(SERVO_MIN + (angle / 180.0) * (SERVO_MAX - SERVO_MIN))
    reg = 0x06 + 4 * channel
    bus.write_i2c_block_data(PCA9685_ADDRESS, reg, [0, 0, pulse & 0xFF, pulse >> 8])

@dataclass(order=True)
class Face:
    size: int
    x: int
    y: int
    width: int
    height: int

def detect_faces(face_detector, gray_array: np.ndarray) -> Sequence[Face]:
    faces = face_detector.detectMultiScale(gray_array, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if not len(faces): return []
    return sorted([Face(size=w*h, x=x, y=y, width=w, height=h) for (x, y, w, h) in faces], reverse=True)


def adjust_servos(bus, face: Face, frame_dims):
    global current_pan, current_tilt
    face_center_x = face.x + face.width // 2
    face_center_y = face.y + face.height // 2
    error_x = face_center_x - frame_dims[0] // 2
    error_y = face_center_y - frame_dims[1] // 2
    
    p_gain_pan = 0.8
    p_gain_tilt = 0.3
    
    current_pan -= -error_x * p_gain_pan
    current_tilt += error_y * p_gain_tilt
    
    current_pan = max(0, min(180, current_pan))
    current_tilt = max(55, min(125, current_tilt))
    
    move_servo(bus, 0, current_pan)
    move_servo(bus, 1, current_tilt)

def test_servos(bus):
                move_servo(bus, 0, 120)
                move_servo(bus, 1, 120)
                time.sleep(0.5)
                move_servo(bus, 0, 60)
                move_servo(bus, 1, 60)
                time.sleep(0.5)
                move_servo(bus, 0, 90)
                move_servo(bus, 1, 90)

def main_loop(camera, face_detector, bus):
    global current_pan, current_tilt, servo_test
    frame_dims = camera.stream_configuration("main")["size"]
    last_face_time = time.time()
    frame_count = 0
    
    

    while True:
        try:
            test_frame = camera.capture_array("main")
            rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite("corrected_test_frame.jpg", rgb_frame)
            
            captured_array = camera.capture_array("main")
            frame_count += 1
                
            gray_array = cv2.cvtColor(captured_array, cv2.COLOR_BGR2GRAY)

            if servo_test:
                 test_servos(bus)
                 servo_test = False

            # editable detection parameters
            faces = face_detector.detectMultiScale(
                gray_array, 
                scaleFactor=1.1, 
                minNeighbors=3,  
                minSize=(30, 30),  
            )
            
            print(f"Frame {frame_count}: Detected {len(faces)} faces")
            
            if len(faces) > 0:
                faces_objects = sorted([Face(size=w*h, x=x, y=y, width=w, height=h) for (x, y, w, h) in faces], reverse=True)
                last_face_time = time.time()
                adjust_servos(bus, faces_objects[0], frame_dims)
                print(f"Servo positions: Pan={current_pan:.1f}, Tilt={current_tilt:.1f}")
            elif time.time() - last_face_time > 3 and (int(current_pan) != 90 or int(current_tilt) != 90):
                print("No face detected for 3 seconds, re-centering.")
                current_pan, current_tilt = 90, 90
                move_servo(bus, 0, current_pan)
                move_servo(bus, 1, current_tilt)
                print(current_tilt)
                last_face_time = time.time()
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(0.1)
            continue
            
        time.sleep(1 / DEFAULT_FPS)


if __name__ == '__main__':

    # Debug: Check if classifier file exists
    print(f"Checking for classifier file: {FACEDETECTION_FILTER_PATH}")
    if not os.path.exists(FACEDETECTION_FILTER_PATH):
        print(f"ERROR: Classifier file not found at {FACEDETECTION_FILTER_PATH}")
        
        # Try alternative locations
        alt_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]
    
    print("Loading Haar Cascade classifier...")
    face_detector = cv2.CascadeClassifier(FACEDETECTION_FILTER_PATH)
    if face_detector.empty():
        print(f"FATAL: Could not load classifier at {FACEDETECTION_FILTER_PATH}")
        exit(1)
    else:
        print("Classifier loaded successfully!")

    print("Initializing Camera...")
    camera = Picamera2()
    
    # Debug: Print available camera modes
    print("Available camera modes:")
    for mode in camera.sensor_modes:
        print(f"  {mode}")
    
    config = camera.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
    print(f"Camera config: {config}")
    
    camera.configure(config)
    camera.start()
    time.sleep(2)  # Longer startup time
    
    print("Camera started. Taking test capture...")
    
    # Test capture
    try:
        test_frame = camera.capture_array("main")
        print(f"Test frame captured: Shape={test_frame.shape}, dtype={test_frame.dtype}")
        
        # Save test frame for debugging
        cv2.imwrite("test_frame.jpg", test_frame)
        print("Test frame saved as test_frame.jpg")
        
    except Exception as e:
        print(f"Error capturing test frame: {e}")
        camera.stop()
        exit(1)

    bus = None
    try:
        bus = smbus.SMBus(1)
        setup_pca9685(bus)
        
        print("Centering servos...")
        move_servo(bus, 0, current_pan)
        move_servo(bus, 1, current_tilt)
        time.sleep(1)

        test_servos(bus)

        print("\nStarting Face Tracking ")
        main_loop(camera, face_detector, bus)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        camera.stop()
        print("Camera stopped.")
        if bus:
            print("Returning servos to center.")
            move_servo(bus, 0, 90)
            move_servo(bus, 1, 90)
        print("Goodbye!")