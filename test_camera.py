from andor_camera import AndorCamera
import time
import matplotlib.pyplot as plt

camera = AndorCamera()
camera.coolCamera(-70)
temp, _ = camera.getCameraTemp()
while temp > -70:
    temp, _ = camera.getCameraTemp()
    print(f"Temp is {temp}")
    time.sleep(1)

camera.setReadMode(mode=4)
camera.setAcquisitionMode(1)
camera.setImage(1, 1, 1, 512, 1, 512)
camera.setShutter(1, 1, 0, 0)
camera.setExposureTime(0.2)
time.sleep(0.1)
image = camera.getImage()
camera.setShutter(1, 2, 0, 0)
plt.imshow(image)
plt.show()