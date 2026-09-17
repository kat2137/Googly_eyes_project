import machine
from machine import Pin, PWM
import time 

servoPin = PWM(Pin(16))
servoPin.freq(50)

def servo(degrees):
    if degrees > 180: degrees = 180
    if degrees < 0: degrees = 0

    maxDuty = 9000
    minDuty = 1000
    newDuty = minDuty + (maxDuty - minDuty) * (degrees / 180)
    servoPin.duty_u16(int(newDuty))

while True:
    # Sweep from 0 to 65 degrees quickly
    for degree in range(0, 66, 5):  # step of 5 for faster movement
        servo(degree)
        time.sleep(0.01)
        print("forward — " + str(degree))

    for degree in range(65, -1, -5):  # back down in steps of 5
        servo(degree)
        time.sleep(0.01)
        print("reverse — " + str(degree))
