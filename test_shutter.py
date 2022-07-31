import andor_camera
import numpy as np
import time

camera = andor_camera.AndorCamera()
camera.SetPreAmpGain(1)
camera.SetVSSpeed(1)
camera.SetADChannel(0)
camera.SetEMCCDGain(0)
camera.SetAcquisitionMode(1) # single scan
camera.SetNumberAccumulations(1)
camera.SetReadMode(3) # single track
camera.SetSingleTrack(284, 309, 1) # metrix ROI to capture
camera.SetExposureTime(0)
camera.SetTriggerMode(0)
camera.GetAcquisitionTimings()
camera.SetCoolerMode(1)
camera.CoolerON()
camera.SetTemperature(-80)


time.sleep(1)
camera.SetShutter(1, 0, 0, 0)
time.sleep(1)
camera.StartAcquisition()
camera.WaitForAcquisition()
time.sleep(1)
image_data = np.asarray(camera.GetAcquiredData([]))
print(f"delta = {max(image_data) - min(image_data)}")

time.sleep(1)
camera.SetShutter(1, 1, 0, 0)
time.sleep(1)
camera.StartAcquisition()
camera.WaitForAcquisition()
time.sleep(1)
image_data = np.asarray(camera.GetAcquiredData([]))
print(f"delta = {max(image_data) - min(image_data)}")
