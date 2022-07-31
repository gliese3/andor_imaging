import numpy as np
from andor_camera_dll import AndorCameraDll 


class AndorCamera(): 
    def __init__(self):
        self.camera = AndorCameraDll()
        
        
    def getCameraTemp(self):
        temp, status = self.camera.getTemperature()
        return temp, status
    
    
    def setPreAmpGain(self, index):
        self.camera.setPreAmpGain(index)
    
    
    def setEMCCDGain(self, gain):
        self.camera.setEMCCDGain(gain)
        
        
    def setAcquisitionMode(self, mode):
        self.camera.setAcquisitionMode(mode)
        
    
    def setReadMode(self, mode):
        self.camera.setReadMode(mode)
        
    
    def setTriggerMode(self, mode):
        self.camera.setTriggerMode(mode)
        
        
    def setImage(self, hbin, vbin, hstart, hend, vstart, vend):
        self.camera.setImage(hbin, vbin, hstart, hend, vstart, vend)
        
        
    def coolCamera(self, temp):
        self.camera.setCoolerMode(1)
        self.camera.coolerOn()
        self.camera.setTemperature(temp)
        
    
    def getImage(self):
        self.camera.startAcquisition()
        self.camera.waitForAcquisition()
        image = self.camera.getAcquiredData()
        return image
    

    def setShutter(self, typ, mode, closingtime, openingtime):
        self.camera.setShutter(typ, mode, closingtime, openingtime)
        
        
    def setExposureTime(self, exp_time): # in sec
        self.camera.setExposureTime(exp_time)
        
    
    def coolerOff(self):
        self.camera.coolerOff()
        
        
    def shutDown(self):
        self.camera.shutDown()
        
    
    def abortAcquisition(self):
        self.camera.abortAcquisition()
        
        