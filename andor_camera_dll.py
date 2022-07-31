import ctypes as ct
import numpy as np


class AndorCameraDll(): 
    def __init__(self):

        # list of error codes
        self.ERROR_CODE = {
            20001: "DRV_ERROR_CODES",
            20002: "DRV_SUCCESS",
            20003: "DRV_VXNOTINSTALLED",
            20006: "DRV_ERROR_FILELOAD",
            20007: "DRV_ERROR_VXD_INIT",
            20010: "DRV_ERROR_PAGELOCK",
            20011: "DRV_ERROR_PAGE_UNLOCK",
            20013: "DRV_ERROR_ACK",
            20024: "DRV_NO_NEW_DATA",
            20026: "DRV_SPOOLERROR",
            20034: "DRV_TEMP_OFF",
            20035: "DRV_TEMP_NOT_STABILIZED",
            20036: "DRV_TEMP_STABILIZED",
            20037: "DRV_TEMP_NOT_REACHED",
            20038: "DRV_TEMP_OUT_RANGE",
            20039: "DRV_TEMP_NOT_SUPPORTED",
            20040: "DRV_TEMP_DRIFT",
            20050: "DRV_COF_NOTLOADED",
            20053: "DRV_FLEXERROR",
            20066: "DRV_P1INVALID",
            20067: "DRV_P2INVALID",
            20068: "DRV_P3INVALID",
            20069: "DRV_P4INVALID",
            20070: "DRV_INIERROR",
            20071: "DRV_COERROR",
            20072: "DRV_ACQUIRING",
            20073: "DRV_IDLE",
            20074: "DRV_TEMPCYCLE",
            20075: "DRV_NOT_INITIALIZED",
            20076: "DRV_P5INVALID",
            20077: "DRV_P6INVALID",
            20083: "P7_INVALID",
            20089: "DRV_USBERROR",
            20091: "DRV_NOT_SUPPORTED",
            20099: "DRV_BINNING_ERROR",
            20990: "DRV_NOCAMERA",
            20991: "DRV_NOT_SUPPORTED",
            20992: "DRV_NOT_AVAILABLE"
        }

        self.RET_VALES = {
            "DRV_SUCCESS" : 20002,
        }

        self.path_to_dll = "lib/atmcd64d_legacy.dll" # path to dll file
        self.lib = ct.CDLL(self.path_to_dll)
        cam_total = self.getAvailableCameras()
        print("Number of availiable cameras : ", cam_total)
        if cam_total == 0:
            print("No cameras found")
            return

        # init SDK
        self.initialize()

        # check resolution
        self.x_res, self.y_res = self.getDetector()
        print("Camera resolution:", f"x : {self.x_res}", f"y : {self.y_res}", sep="\n", end="\n")
        
        print("Camera is ready.")


    def getAvailableCameras(self):
        """
        This function returns the total number of Andor cameras currently installed.
        It is possible to call this function before any of the cameras are initialized.
        
        """
        cam_total = ct.c_long()
        ret_val =  self.lib.GetAvailableCameras(ct.byref(cam_total))
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"getAvailableCameras() error | error code: {ret_val}"           
        return cam_total.value
    
    
    def getDetector(self):
        """
        This function returns the size of the detector in pixels.
        The horizontal axis is taken to be the axis parallel to the readout register.
        
        """
        x_pixels = ct.c_int()
        y_pixels = ct.c_int()
        ret_val =  self.lib.GetDetector(ct.byref(x_pixels), ct.byref(y_pixels))
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"getDetector() error | error code: {ret_val}"           
        return x_pixels.value, y_pixels.value


    def initialize(self):
        """
        This function will initialize the Andor SDK System. As part of the initialization procedure on
        some cameras (i.e. Classic, iStar and earlier iXion) the DLL will need access to a
        DETECTOR.INI which contains information relating to the detector head, number pixels,
        readout speeds etc. If your system has multiple cameras then see the section Controlling
        multiple cameras
        
        """
        #! check this function. Page 211.
        dir_ = ct.c_char()
        ret_val =  self.lib.Initialize(ct.byref(dir_))
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"initialize() error | error code: {ret_val}"           


    def coolerOn(self):
        """
        Switches ON the cooling. On some systems the rate of temperature change is controlled
        until the temperature is within 3°C of the set value. Control is returned immediately to the
        calling application.
        
        """
        ret_val =  self.lib.CoolerON()
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"coolerOn() error | error code: {ret_val}"


    def coolerOff(self):
        """
        Switches OFF the cooling. The rate of temperature change is controlled in some models
        until the temperature reaches 0°C. Control is returned immediately to the calling
        application.
        
        NOTE:
            1. When the temperature control is switched off the temperature of the sensor is gradually
            raised to 0°C to ensure no thermal stresses are set up in the sensor.
            2. When closing down the program via ShutDown you must ensure that the temperature of the
            detector is above -20°C, otherwise calling ShutDown while the detector is still cooled will
            cause the temperature to rise faster than certified.
        
        """
        ret_val =  self.lib.CoolerOFF()
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"CoolerOFF() error | error code: {ret_val}"


    def setCoolerMode(self, mode):
        """
        This function determines whether the cooler is switched off when
        the camera is shutdown.
        
        mode values:
            1 : Temperature is maintained on ShutDown
            0 : Returns to ambient temperature on ShutDown
        
        """
        mode = ct.c_int(mode)
        ret_val =  self.lib.SetCoolerMode(mode)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"SetCoolerMode() error | error code: {ret_val}"
        

    def getTemperature(self):
        """
        This function returns the temperature of the detector to the nearest degree.
        It also gives the status of cooling process.
        
        """
        codes = {
            20075 : "System not initialized",
            20072 : "Acquisition in progress",
            20013 : "Unable to communicate with card",
            20034 : "Temperature is OFF",
            20036 : "Temperature has stabilized at set point",
            20037 : "Temperature has not reached set point",
            20040 : "Temperature had stabilised but has since drifted",
            20035 : "Temperature reached but not stabilized",
        }
        temp = ct.c_int()        
        ret_val =  self.lib.GetTemperature(ct.byref(temp))
        return temp.value, codes[ret_val]


    def setTemperature(self, temp):
        """
        This function will set the desired temperature of the detector. To turn the cooling ON and
        OFF use the CoolerON and CoolerOFF function respectively.
        
        """
        temp = ct.c_int(temp)        
        ret_val =  self.lib.SetTemperature(temp)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setTemperature() error | error code: {ret_val}"
        

    def setAcquisitionMode(self, mode=1):
        """
        This function will set the acquisition mode to be used on the next StartAcquisition.
        
        mode values:
                1 : Single Scan
                2 : Accumulate
                3 : Kinetics
                4 : Fast Kinetics
                5 : Run till abort
        
        """
        mode = ct.c_int(mode)        
        ret_val =  self.lib.SetAcquisitionMode(mode)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setAcquisitionMode() error | error code: {ret_val}"


    def setADChannel(self, channel):
        """
        This function will set the AD channel to one of the possible A-Ds of the system. This AD
        channel will be used for all subsequent operations performed by the system.
        
        """
        channel = ct.c_int(channel)
        ret_val = self.lib.SetADChannel(channel)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setADChannel() error | error code: {ret_val}"
        

    def setExposureTime(self, exp_time):
        """
        This function will set the exposure time to the nearest valid value not 
        less than the given value. The actual exposure time used is obtained by
        GetAcquisitionTimings.
        
        exp_time should be in seconds.
        
        """
        exp_time = ct.c_float(exp_time)        
        ret_val =  self.lib.SetExposureTime(exp_time)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setExposureTime() error | error code: {ret_val}"


    def setReadMode(self, mode):
        """
        This function will set the readout mode to be used on the
        subsequent acquisitions.
        
        mode values:
            0 : Full Vertical Binning
            1 : Multi-Track
            2 : Random-Track
            3 : Single-Track
            4 : Image
        
        """
        mode = ct.c_int(mode)
        ret_val =  self.lib.SetReadMode(mode)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setReadMode() error | error code: {ret_val}"
        self.read_mode = mode.value
        
        
    def setTriggerMode(self, mode=0):
        """
        This function will set the trigger mode that the camera will operate in.
        
        mode values:
            0 : Internal
            1 : External
            6 : External Start
            7 : External Exposure (Bulb)
            9 : External FVB EM (only valid for EM Newton models in FVB mode)
            10 : Software Trigger
            12 : External Charge Shifting
            
        """
        mode = ct.c_int(mode)
        ret_val =  self.lib.SetTriggerMode(mode)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setTriggerMode() error | error code: {ret_val}"


    def getAcquiredData(self):
        """
        This function will return the data from the last acquisition. The data are returned as long
        integers (32-bit signed integers). The “array” must be large enough to hold the complete
        data set.
        
        """
        #! must be corrected if needed
        
        if self.read_mode == 4:
            total_pixel_num = self.x_res * self.y_res
            arr = (ct.c_int32 * total_pixel_num)()
            ret_val =  self.lib.GetAcquiredData(ct.byref(arr),
                                                ct.c_ulong(total_pixel_num))
            assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"getAcquiredData() error | error code: {ret_val}"
            
            # now arr is a python 1-D array of size total_pixel_num
            # convert it to proper 2-D numpy array
            image = np.array(arr, dtype=np.int32).reshape(self.x_res, self.y_res)
            return image
        else:
            raise Exception("Only read mode = 4 is supported for now")


    def getStatus(self):
        """
        This function will return the current status of the Andor SDK system.
        This function should be called before an acquisition is started to ensure
        that it is IDLE and during an acquisition to monitor the process.
        
        """
        code = {
            20073 : "IDLE waiting on instructions",
            20074 : "Executing temperature cycle",
            20072 : "Acquisition in progress",
            20023 : "Unable to meet Accumulate cycle time",
            20022 : "Unable to meet Kinetic cycle time",
            20013 : "Unable to communicate with card",
            20018 : "Computer unable to read the data via the ISA slot at the required rate",
            20019 : "Computer unable to read data fast enough to stop camera memory going full",
            20026 : "Overflow of the spool buffer",
        }

        status = ct.c_int()
        ret_val =  self.lib.GetStatus(ct.byref(status))
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"getStatus() error | error code: {ret_val}"
        return code[status]
    

    def setPreAmpGain(self, index):
        """
        This function will set the pre amp gain to be used for subsequent acquisitions.
        The actual gain factor that will be applied can be found through a call
        to the GetPreAmpGain function. The number of Pre Amp Gains
        available is found by calling the GetNumberPreAmpGains function.
        
        """
        index = ct.c_int(index)
        ret_val =  self.lib.SetPreAmpGain(index)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setPreAmpGain() error | error code: {ret_val}"


    def shutDown(self):
        """
        This function will close the AndorMCD system down.
        
        """
        ret_val =  self.lib.ShutDown()
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"shutDown() error | error code: {ret_val}"


    def abortAcquisition(self):
        """
        This function aborts the current acquisition if one is active.
        
        """
        self.lib.AbortAcquisition()


    def startAcquisition(self):
        """
        This function starts an acquisition. The status of the acquisition can be
        monitored via GetStatus().
        
        """
        ret_val =  self.lib.StartAcquisition()
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"startAcquisition() error | error code: {ret_val}"
        
        
    def waitForAcquisition(self):
        """
        WaitForAcquisition can be called after an acquisition is started using StartAcquisition to
        put the calling thread to sleep until an Acquisition Event occurs. This can be used as a
        simple alternative to the functionality provided by the SetDriverEvent function, as all
        Event creation and handling is performed internally by the SDK library.
        Like the SetDriverEvent functionality it will use less processor resources than
        continuously polling with the GetStatus function. If you wish to restart the calling thread
        without waiting for an Acquisition event, call the function CancelWait.
        An Acquisition Event occurs each time a new image is acquired during an Accumulation,
        Kinetic Series or Run-Till-Abort acquisition or at the end of a Single Scan Acquisition.
        If a second event occurs before the first one has been acknowledged, the first one will be
        ignored. Care should be taken in this case, as you may have to use CancelWait to exit
        the function.
        
        """
        ret_val =  self.lib.WaitForAcquisition()
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"startAcquisition() error | error code: {ret_val}"
        

    def getAcquisitionTimings(self):
        """
        This function will return the current “valid” acquisition timing information. This function
        should be used after all the acquisitions settings have been set, e.g. SetExposureTime,
        SetKineticCycleTime and SetReadMode etc. The values returned are the actual times
        used in subsequent acquisitions. This function is required as it is possible to set
        the exposure time to 20ms, accumulate cycle time to 30ms and then set the readout mode 
        to full image. As it can take 250ms to read out an image it is not possible
        to have a cycle time of 30ms.
        
        """
        exposure   = ct.c_float()
        accumulate = ct.c_float()
        kinetic    = ct.c_float()
        ret_val = self.lib.GetAcquisitionTimings(ct.byref(exposure),
                                                 ct.byref(accumulate),
                                                 ct.byref(kinetic))
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"getAcquisitionTimings() error | error code: {ret_val}"
        return exposure.value, accumulate.value, kinetic.value


    def setIsolatedCropMode(self, active, cropheight, cropwidth, vbin, hbin):
        """
        This function effectively reduces the dimensions of the CCD by excluding some rows or
        columns to achieve higher throughput. In isolated crop mode iXon, Newton and iKon
        cameras can operate in either Full Vertical Binning or Imaging read modes. iDus can
        operate in Full Vertical Binning read mode only.
        
        """
        active      = ct.c_int(active)
        cropheight  = ct.c_int(cropheight)
        cropwidth   = ct.c_int(cropwidth)
        vbin        = ct.c_int(vbin)
        hbin        = ct.c_int(hbin)
        ret_val = self.lib.SetIsolatedCropMode(active, cropheight, cropwidth, vbin, hbin)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setIsolatedCropMode() error | error code: {ret_val}"
        

    def setShutter(self, typ, mode, closingtime, openingtime):
        """
        This function controls the behaviour of the shutter.
        The typ parameter allows the user to control the TTL signal output to an external shutter.
        The mode parameter configures whether the shutter opens & closes automatically
        (controlled by the camera) or is permanently open or permanently closed.
        
        int typ:
            0 : Output TTL low signal to open shutter
            1 : Output TTL high signal to open shutter
 
        int mode:
            0 : Fully Auto
            1 : Permanently Open
            2 : Permanently Closed
            4 : Open for FVB series
            5 : Open for any series
            
        int closingtime : Time shutter takes to close (milliseconds)
        int openingtime : Time shutter takes to open (milliseconds)
        
        """
        typ         = ct.c_int(typ)
        mode        = ct.c_int(mode)
        closingtime = ct.c_int(closingtime)
        openingtime = ct.c_int(openingtime)        
        ret_val = self.lib.SetShutter(typ, mode, closingtime, openingtime)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setShutter() error | error code: {ret_val}"


    def setImage(self, hbin, vbin, hstart, hend, vstart,vend):
        """
        This function will set the horizontal and vertical binning to be used
        when taking a full resolution image.
        
        """
        hbin    = ct.c_int(hbin)
        vbin    = ct.c_int(vbin)
        hstart  = ct.c_int(hstart)
        hend    = ct.c_int(hend)        
        vstart  = ct.c_int(vstart)  
        vend    = ct.c_int(vend)  
        ret_val = self.lib.SetImage(hbin, vbin, hstart, hend, vstart, vend)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setShutter() error | error code: {ret_val}"
        

    def setEMCCDGain(self, gain):
        """
        Allows the user to change the gain value. The valid range for the gain depends on what
        gain mode the camera is operating in. See SetEMGainMode to set the mode and
        GetEMGainRange to get the valid range to work with. To access higher gain values
        (>x300) see SetEMAdvanced.
        
        """
        gain = ct.c_int(gain)
        ret_val = self.lib.SetEMCCDGain(gain)
        assert ret_val == self.RET_VALES["DRV_SUCCESS"], f"setShutter() error | error code: {ret_val}"


########### Automation functions #################

    def Demo_CoolDown(self):
        '''
        Cool down the camera for a demo measurement
        '''
        Tset = -25
        self.SetCoolerMode(1)

        self.SetTemperature(Tset)
        self.CoolerON()

        while self.GetTemperature() != 'DRV_TEMP_STABILIZED':
            time.sleep(10)

    def Demo_ImagePrepare(self):
        '''
        Prepare the camera for a demo image measurement
        '''
        PreAmpGain = 0
        self.SetSingleImage()
        self.SetTriggerMode(0)
        self.SetShutter(1, 1, 0, 0)
        self.SetPreAmpGain(PreAmpGain)
        self.SetExposureTime(0.1)

    def Demo_ImageCapture(self):
        '''
        Perform the demo image measurement
        '''
        i = 0
        while i < 4:
            i += 1
            if verbose: print(self.GetTemperature())
            if verbose: print(self._temperature)
            if verbose: print("Ready for Acquisition")
            self.StartAcquisition()

            # Check for status
            while self.GetStatus() != 'DRV_IDLE':
                if verbose: print("Data not yet acquired, waiting 0.5s")
                time.sleep(0.5)

            data = []
            self.GetAcquiredData(data)
            self.SaveAsBmpNormalised("n%03g.bmp" % i)
            self.SaveAsBmp("%03g.bmp" % i)
            self.SaveAsTxt("%03g.txt" % i)

    def Demo_FVBPrepare(self):
        '''
        Prepare the camera for a demo image measurement
        '''
        PreAmpGain = 0
        self.SetSingleFVB()
        self.SetTriggerMode(0)
        self.SetShutter(1, 1, 0, 0)
        self.SetPreAmpGain(PreAmpGain)
        self.SetExposureTime(0.1)

    def Demo_FVBCapture(self):
        '''
        Perform the demo image measurement
        '''
        i = 0
        while i < 4:
            i += 1
            if verbose: print(self.GetTemperature())
            if verbose: print(self._temperature)
            if verbose: print("Ready for Acquisition")
            self.StartAcquisition()

            # Check for status
            while self.GetStatus() != 'DRV_IDLE':
                if verbose: print("Data not yet acquired, waiting 0.5s")
                time.sleep(0.5)

            data = []
            self.GetAcquiredData(data)
            self.SaveAsTxt("%03g.txt" % i)

#####################################################

