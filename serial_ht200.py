import serial



class MySerial:
    def __init__(self,com):
        self.rx_var_formated = []
        self.__baud = 9600
        self.__com = com
    def sendData(self, fase, tiempo):
        ser = serial.Serial(self.__com, self.__baud, timeout=0)
        print("pattern ...")
        gbtx = bytearray(714)
        gbtx[0] = 192
        gbtx[1] = 13        #// Direcion esclavo
        gbtx[2] = 129 #// apagar led
        gbtx[3] = 25
        gbtx[4] = 1
        gbtx[5] = 3
        gbtx[6] = fase #//fase
        gbtx[7] = tiempo #//tiempo
        gbtx[8] = 4
        gbtx[9] = 1
        gbtx[10] = 2
        gbtx[11] = 3
        gbtx[12] = 4
        gbtx[13] = 239
        ser.write(gbtx)
        ser.close() 


# ht200 = MySerial("COM3")
# ht200.sendData(2,15)