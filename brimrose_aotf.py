import serial
class Aotf():
    def __init__(self, 
                 port="COM5", # change port if needed
                 baudrate=115200,
                 timeout=0.005, 
                 bytesize=8,
                 parity="N",
                 stopbits=1):
        
        self.ser_port = serial.Serial(port=port,
                                    baudrate=baudrate,
                                    timeout=timeout,
                                    bytesize=bytesize,
                                    parity=parity,
                                    stopbits=stopbits)


        self.setCurrentChannel() # init process
        
        
    def setCurrentChannel(self, channel=0, port=0):
        """
        Set current channel and port. All following channel specific command
        will effectcurrent channel and port. First channel is 0. First port is 0.
        
        """
        self.ser_port.write(bytes(f"SCH,{channel},{port}\r", "utf-8"))
        res = self.ser_port.readline()
        if res.find(b"OK,0") == -1:
            raise Exception("setCurrentChannel : reply doesn't contain 'OK,0' string")
        
        
    def setWavelength(self, wave_len):
        """
        Set center wavelength of the AOTF passband, in nm.
        
        """
        assert 500 <= wave_len <= 900, "Bad wavelength value. It should be in range from 500 to 900 nm."
        self.ser_port.write(bytes(f"SWL,{wave_len}\r", "utf-8"))
        res = self.ser_port.readline()
        if res.find(b"OK,0") == -1:
            raise Exception("setWavelength() : reply doesn't contain 'OK,0' string")
        
        
    def getWavelength(self):
        """
        Get current center wavelength of the AOTF passband. EC will always be 0
        in the response.
        
        """
        self.ser_port.write(bytes(f"GWL\r", "utf-8"))
        res = self.ser_port.readline() # example: b'GWL\r,OK,0,600.000\r'
        ret_val = float(res.split(b",")[-1].strip().decode())
        return ret_val
    
    
    def setAmplitude(self, ampl):
        """
        Set the RF amplitude. It ranges from 0 to 4095.
        
        """
        assert 0 <= ampl <= 4095, "Bad amplitude value. It should be in range from 0 to 4095 nm."
        self.ser_port.write(bytes(f"SAM,{ampl}\r", "utf-8"))
        res = self.ser_port.readline()
        if res.find(b"OK,0") == -1:
            raise Exception("setAmplitude() : reply doesn't contain 'OK,0' string")
        
    
    def getAmplitude(self):
        """
        Get the RF amplitude. It ranges from 0 to 4095.
        
        """
        self.ser_port.write(bytes(f"GAM\r", "utf-8"))
        res = self.ser_port.readline() # example: b'GAM\r,OK,0,90\r'
        ret_val = int(res.split(b",")[-1].strip().decode())
        return ret_val

    def stopAotf(self):
        self.ser_port.close()
    