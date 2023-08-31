import socket
import tramas
import datetime



class MySocketHT200:
    def __init__(self):
        self.rx_var_formated = []
        self.__rx_var = bytearray(2048)
        self.__rx_num = 0
        self.__num = 11
        self.__ips_connected = []
        self.__port = 161

    def readPendingDatagrams(self, frame, ip_controller,timeout_request = 10):
        CheckSumCalc = 0
        CheckSumReceive = 0
        data_received = bytearray()
        flag_good_connection = False
        try:
            __udpsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            __udpsocket.settimeout(timeout_request)
            port = 13536
            while True:
                try:
                    __udpsocket.bind(('0.0.0.0', port))
                    __udpsocket.sendto(frame, (ip_controller, self.__port))
                    data_received,sender = __udpsocket.recvfrom(2048)
                    flag_good_connection = True 
                    break
                except OSError:
                    port += 1
                except __udpsocket.timeout:
                    flag_good_connection = False 
                    print("closing socket")
                    break
            __udpsocket.close()
            if flag_good_connection == False:
                return False
            # convertimos en una lista de enteros los valores recibidos por udp
            array_data_received = list(data_received)
            size = len(array_data_received)
            if array_data_received[size-3] == 0xDB and array_data_received[size-2] == 0xDC:
                dataEndPoint = size-4
                CheckSumReceive = 0xC0
                for i in range(1, dataEndPoint+1):
                    CheckSumCalc += array_data_received[i]
            elif array_data_received[size-3] == 0xDB and array_data_received[size-2] == 0xDD:
                dataEndPoint = size-4
                CheckSumReceive = 0xDB
                for i in range(1, dataEndPoint+1):
                    CheckSumCalc += array_data_received[i]
            else:
                dataEndPoint = size-3
                CheckSumReceive = array_data_received[size-2]
                for i in range(1, dataEndPoint+1):
                    CheckSumCalc += array_data_received[i]
            CheckSumCalc = (CheckSumCalc % 256)
            if CheckSumCalc != CheckSumReceive:
                print("checksum diferente")
                return False
            self.__rx_num = 0
            self.__num = 11
            while self.__num <= dataEndPoint:
                if array_data_received[self.__num] == 0xDB and array_data_received[self.__num+1] == 0xDC:
                    self.__rx_var[self.__rx_num] = 0xC0
                    self.__rx_num += 1
                    self.__num += 2
                elif array_data_received[self.__num] == 0xDB and array_data_received[self.__num+1] == 0xDD:
                    self.__rx_var[self.__rx_num] = 0xDB
                    self.__rx_num += 1
                    self.__num += 2
                else:
                    self.__rx_var[self.__rx_num] = array_data_received[self.__num]
                    self.__rx_num += 1
                    self.__num += 1
            return True
        except OSError:
            print("algo ocurrio mal")
            return False

    def getTime(self, ip):
        self.__rx_var
        if self.readPendingDatagrams(tramas.time_frame, ip):
            second = self.__rx_var[0]//16*10 + self.__rx_var[0] % 16  # segundo
            minute = self.__rx_var[1]//16*10 + self.__rx_var[1] % 16  # minuto
            hour = self.__rx_var[2]//16*10 + self.__rx_var[2] % 16  # hora
            week = self.__rx_var[3]  # semana
            date = self.__rx_var[4]//16*10 + \
                self.__rx_var[4] % 16  # día del mes
            month = self.__rx_var[5]//16*10 + self.__rx_var[5] % 16  # mes
            year = 2000 + self.__rx_var[6]//16 * \
                10 + self.__rx_var[6] % 16  # año
            time_controler = {
                "segundos": second,
                "minutos": minute,
                "hour": hour,
                "semana": week,
                "dia": date,
                "mes": month,
                "year": year,
            }
            return time_controler

    def getFases(self, ip):
        rx_var = self.__rx_var
        print('solicitando fases ....')
        if self.readPendingDatagrams(tramas.fases_frame, ip):
            PhaseSize = 32
            if 16 == rx_var[0] and self.__rx_num == PhaseSize * 16 + 1:
                data_list = []
                for i in range(16):
                    readpoint = PhaseSize * i + 1
                    Number = rx_var[readpoint]
                    Walk = rx_var[readpoint+1]
                    PedestrianClear = rx_var[readpoint+2]
                    MinimumGreen = rx_var[readpoint+3]
                    Passage = rx_var[readpoint+4]
                    Maximum1 = rx_var[readpoint+5]
                    Maximum2 = rx_var[readpoint+6]
                    YellowChange = rx_var[readpoint+7]
                    RedClear = rx_var[readpoint+8]
                    RedRevert = rx_var[readpoint+9]
                    AddedInitial = rx_var[readpoint+10]
                    MaximumInitial = rx_var[readpoint+11]
                    TimeBeforeReduction = rx_var[readpoint+12]
                    CarsBeforeReduction = rx_var[readpoint+13]
                    TimeToReduce = rx_var[readpoint+14]
                    ReduceBy = rx_var[readpoint+15]
                    MinimumGap = rx_var[readpoint+16]
                    DynamicMaxLimit = rx_var[readpoint+17]
                    DynamicMaxStep = rx_var[readpoint+18]
                    Startup = rx_var[readpoint+19]
                    Ring = rx_var[readpoint+20]
                    VehicleClear = rx_var[readpoint+21]
                    Options = rx_var[readpoint +
                                     22] | (rx_var[readpoint+23] << 8)
                    Concurrency = rx_var[readpoint+24] | (rx_var[readpoint+25] << 8) | (
                        rx_var[readpoint+26] << 16) | (rx_var[readpoint+27] << 24)
                    ReleasePhase = rx_var[readpoint+28] | (rx_var[readpoint+29] << 8) | (
                        rx_var[readpoint+30] << 16) | (rx_var[readpoint+31] << 24)

                    # creamos un diccionario con los datos
                    data_fase = {
                        'number': Number,
                        'walk': Walk,
                        'pedestrianClear': PedestrianClear,
                        'minimumGreen': MinimumGreen,
                        'passage': Passage,
                        'maximun1': Maximum1,
                        'maximun2': Maximum2,
                        'yellowchange': YellowChange,
                        'redclear': RedClear,
                        'RedRevert': RedRevert,
                        'AddedInitial': AddedInitial,
                        'MaximunInitial': MaximumInitial,
                        'TimeBeforeReduction': TimeBeforeReduction,
                        'carsbeforereduction': CarsBeforeReduction,
                        'timetoreduce': TimeToReduce,
                        'reduceby': ReduceBy,
                        'minimungap': MinimumGap,
                        'dynamimaxlist': DynamicMaxLimit,
                        'dynamicmaxstep': DynamicMaxStep,
                        'startup': Startup,
                        'ring': Ring,
                        'vehicleclear': VehicleClear,
                        'options': Options,
                        'concurrency': Concurrency,
                        'releasephase': ReleasePhase
                    }
                    data_list.append(data_fase)
                return (data_list)

    def getSecuencia(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.secuence_frame, ip):

            SequenceSize = (16 + 1) * 4 + 1
            list_secuencies = []
            table = 0
            if 16 == rx_var[0] and self.__rx_num == SequenceSize * 16 + 1:
                readpoint = 1
                for i in range(8):
                    table += 1
                    Num = rx_var[readpoint]
                    readpoint += 1
                    rings_secuency = []
                    for i in range(4):
                        RingNum = rx_var[readpoint]
                        fases_ring = []
                        readpoint += 1
                        for i in range(16):
                            fase = rx_var[readpoint]
                            readpoint += 1
                            fase_data = {
                                "id": "paso-{calculo}".format(calculo=i+1), "value": fase, 'ring': RingNum}
                            fases_ring.append(fase_data)
                        rings_secuency.append(fases_ring)
                    list_secuencies.append(
                        {"data": rings_secuency, "id": "seq-{}".format(table)})
                return list_secuencies

    def getSplit(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.split_frame, ip):
            SplitSize = 16 * 4 + 1
            split_list_total = []
            if 20 == rx_var[0] and self.__rx_num == SplitSize * 20 + 1:
                readpoint = 1
                index = 0
                for i in range(8):  # le dejamos en 1 para mostrar solo la tabla 1
                    index += 1
                    num = rx_var[readpoint]
                    split_list = []
                    readpoint += 1
                    for i in range(16):
                        fase = rx_var[readpoint]
                        readpoint += 1
                        time = rx_var[readpoint]
                        readpoint += 1
                        mode = rx_var[readpoint]
                        readpoint += 1
                        coord = rx_var[readpoint]
                        readpoint += 1
                        split_dict = {
                            'fase': fase,
                            'tiempo': time,
                            'mode': mode,
                            'coord': coord
                        }
                        split_list.append(split_dict)
                    split_list_total.append(
                        {"data": split_list, "id": "split-{}".format(index)})
                return split_list_total

    def getPattern(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.pattern_frame, ip):

            PatternSize = 7
            if 100 == rx_var[0] and self.__rx_num == PatternSize * 100 + 1:
                pattern_list = []
                for i in range(16):
                    readpoint = PatternSize * i + 1
                    Number = rx_var[readpoint]
                    CycleTime = rx_var[readpoint +
                                       1] | (rx_var[readpoint+2] << 8)
                    OffsetTime = rx_var[readpoint+3]
                    SplitNumber = rx_var[readpoint+4]
                    SequenceNumber = rx_var[readpoint+5]
                    WorkMode = rx_var[readpoint+6]
                    pattern_dict = {
                        'number': Number,
                        'cycletime': CycleTime,
                        'offsettime': OffsetTime,
                        'splitnumber': SplitNumber,
                        'sequencenumber': SequenceNumber,
                        'workmode': WorkMode,
                    }
                    pattern_list.append(pattern_dict)

                return pattern_list

    '''
    la funcion de  obtencion de patrones se debe decodificar los valores del objeto para poder mapear
    '''

    def getAccion(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.action_frame, ip):
            ActionSize = 4
            if 100 == rx_var[0] and self.__rx_num == ActionSize * 100 + 1:
                readpoint = 1
                action_list = []
                for i in range(16):
                    Num = rx_var[readpoint]
                    readpoint += 1
                    PatternNum = rx_var[readpoint]
                    readpoint += 1
                    Auxillary = rx_var[readpoint]
                    readpoint += 1
                    Special = rx_var[readpoint]
                    readpoint += 1
                    action_dict = {
                        'number': Num,
                        'patron': PatternNum,
                        'auxiliary': Auxillary,
                        'special': Special
                    }
                    action_list.append(action_dict)
                return action_list

    def getPlanes(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.plan_frame, ip):
            plansize = 73
            if 16 == rx_var[0] and self.__rx_num == (plansize * 16 + 1):
                readpoint = 1
                plan_total_list = []
                for i in range(16):  # le dejamos en 1 para obtener solo el primer plan
                    plan = rx_var[readpoint]
                    plan_list = []
                    readpoint += 1
                    for j in range(24):
                        num = j+1
                        hour = rx_var[readpoint]
                        readpoint += 1
                        minute = rx_var[readpoint]
                        readpoint += 1
                        accion = rx_var[readpoint]
                        readpoint += 1
                        plan_dict = {
                            'id': "num-{}".format(num),
                            'hour': hour,
                            'minute': minute,
                            'action': accion
                        }
                        plan_list.append(plan_dict)
                    plan_total_list.append(
                        {"data": plan_list, "id": "plan-{}".format(plan)})
                return plan_total_list

    def getScnedule(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.schedule_frame, ip):
            schedule_size = 9
            schedule_list = []
            for i in range(40):
                readpoint = schedule_size*i + 1
                number = rx_var[readpoint]
                m_1 = rx_var[readpoint+1]
                m_2 = rx_var[readpoint+2]
                month = m_1 | (m_2 << 8)
                day = rx_var[readpoint+3]
                byte1_date = rx_var[readpoint+4]
                byte2_date = rx_var[readpoint+5]
                byte3_date = rx_var[readpoint+6]
                byte4_date = rx_var[readpoint+7]
                date = byte1_date | (byte2_date << 8) | (
                    byte3_date << 16) | (byte4_date << 24)
                day_plan = rx_var[readpoint+8]
                schedule_dict = {
                    'number': number,
                    'day_plan': day_plan,
                    'month': month,
                    'day': day,
                    'date': date,
                    'd1': byte1_date,
                    'd2': byte2_date,
                    'd3': byte3_date,
                    'd4': byte4_date,
                    'm1': m_1,
                    'm2': m_2
                }
                schedule_list.append(schedule_dict)
            return schedule_list

    def getDeviceInfo(self, ip):
        rx_var = self.__rx_var
        print('se piden datos')
        if self.readPendingDatagrams(tramas.device_info_frame, ip,timeout_request=2):
            StrLen = 0
            temp = [0] * 64
            for i in range(0, 128, 2):
                if rx_var[i] != 0x00 or rx_var[i + 1] != 0x00:
                    temp[StrLen] = (rx_var[i] << 8) | rx_var[i + 1]
                    StrLen += 1
                else:
                    break
            pt = 0
            aux_lat = []
            latitud = 0
            for i in range(14):
                if rx_var[i+100]== 46:
                    pt = i
                aux_lat.append(rx_var[i+100]-48)
            if pt != 0:
                latitud = (aux_lat[pt-4]*10+aux_lat[pt-3])+ ((aux_lat[pt-2]*10)+aux_lat[pt-1]+(aux_lat[pt+1]*0.1)+(aux_lat[pt+2]*0.01)+(aux_lat[pt+3]*0.001)+(aux_lat[pt+4]*0.0001))/60
                if rx_var[100]== ord('0'):
                    latitud = latitud*-1
            aux_lng = []
            longitud = 0
            pt = 0
            for i in range(14):
                if rx_var[i+114]== 46:
                    pt = i
                aux_lng.append(rx_var[i+114]-48)
            if pt != 0:
                longitud = (aux_lng[pt-4]*10+aux_lng[pt-3])+ ((aux_lng[pt-2]*10)+aux_lng[pt-1]+(aux_lng[pt+1]*0.1)+(aux_lng[pt+2]*0.01)+(aux_lng[pt+3]*0.001)+(aux_lng[pt+4]*0.0001))/60
                if rx_var[114]== ord('0'):
                    longitud = longitud*-1
            manufacturerInfoStr = ''.join(
                [chr(temp[i]) for i in range(StrLen)])
            deviceinfo_dict = {'manufacturer': manufacturerInfoStr,'latitud':latitud,'longitud':longitud}
            return deviceinfo_dict

    def getBasicInfo(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.basic_info_frame, ip,timeout_request=2):
            mac_addr = bytearray([rx_var[i] for i in range(142, 148)])
            mac_addr = mac_addr.hex().upper()
            mac_addr = ':'.join([mac_addr[i:i+2] for i in range(0, 12, 2)])
            ip_server = "{oct_1}.{oct_2}.{oct_3}.{oct_4}".format(
                oct_1=rx_var[148], oct_2=rx_var[149], oct_3=rx_var[150], oct_4=rx_var[151])
            port_server = "{port_s}".format(
                port_s=(rx_var[152] << 8) | rx_var[153])
            zona_horaria = ((rx_var[156] << 16) | (
                rx_var[157] << 8) | rx_var[158])/3600.0
            tscNum = (rx_var[159] << 24) | (rx_var[160] << 16) | (
                rx_var[161] << 8) | rx_var[162]
            basicinfo_dict = {
                "mac_target": mac_addr,
                "ip_target": ip_server,
                "puerto_server": port_server,
                "zona_horaria": zona_horaria,
                "numero_dispositivo": tscNum
            }
            return basicinfo_dict

    def getUnit(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.unit_frame, ip):
            if self.__rx_num == 12:
                StartupFlash = rx_var[0]
                StartupAllRed = rx_var[1]
                AutomaticPedClear = rx_var[2]
                RedRevert = rx_var[3]
                BackupTime = rx_var[4] | (rx_var[5] << 8)
                FlowCycle = rx_var[6]
                FlashStatus = rx_var[7]
                Status = rx_var[8]
                GreenConflictDetectFlag = rx_var[9]
                RedGreenConflictDetectFlag = rx_var[10]
                RedFailedDetectFlag = rx_var[11]

                unit_dict = {
                    "StartupFlash": StartupFlash,
                    "StartupAllRed": StartupAllRed,
                    "AutomaticPedClear": AutomaticPedClear,
                    "RedRevert": RedRevert,
                    "BackupTime": BackupTime,
                    "FlowCycle": FlowCycle,
                    "FlashStatus": FlashStatus,
                    "Status": Status,
                    "GreenConflictDetectFlag": GreenConflictDetectFlag,
                    "RedGreenConflictDetectFlag": RedGreenConflictDetectFlag,
                    "RedFailedDetectFlag": RedFailedDetectFlag
                }
                return unit_dict
    def getUnitHT200(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.unit_frame, ip):
            if self.__rx_num == 14:
                StartupFlash = rx_var[0]
                StartupAllRed = rx_var[1]
                AutomaticPedClear = rx_var[2]
                RedRevert = rx_var[3]
                BackupTime = rx_var[4] | (rx_var[5] << 8)
                FlowCycle = rx_var[6]
                FlashStatus = rx_var[7]
                Status = rx_var[8]
                GreenConflictDetectFlag = rx_var[9]
                RedGreenConflictDetectFlag = rx_var[10]
                RedFailedDetectFlag = rx_var[11]
                GoiaSync = rx_var[12]
                ConexionSemaforos = rx_var[13]
                unit_dict = {
                    "StartupFlash": StartupFlash,
                    "StartupAllRed": StartupAllRed,
                    "AutomaticPedClear": AutomaticPedClear,
                    "RedRevert": RedRevert,
                    "BackupTime": BackupTime,
                    "FlowCycle": FlowCycle,
                    "FlashStatus": FlashStatus,
                    "Status": Status,
                    "GreenConflictDetectFlag": GreenConflictDetectFlag,
                    "RedGreenConflictDetectFlag": RedGreenConflictDetectFlag,
                    "RedFailedDetectFlag": RedFailedDetectFlag,
                    "GoiaSync":GoiaSync,
                    "SemaforoFlag":ConexionSemaforos
                }
                print(unit_dict)
                return unit_dict

    def getChannel(self, ip):
        rx_var = self.__rx_var
        channel_list = []
        if self.readPendingDatagrams(tramas.chanel_frame, ip):
            ChannelSize = 8
            if 16 == rx_var[0] and self.__rx_num == ChannelSize * 16 + 1:
                readpoint = 1
                for i in range(16):
                    Num = rx_var[readpoint]
                    readpoint += 1
                    ControlSource = rx_var[readpoint]
                    readpoint += 1
                    ControlType = rx_var[readpoint]
                    readpoint += 1
                    Flash = rx_var[readpoint]
                    readpoint += 1
                    Dim = rx_var[readpoint]
                    readpoint += 1
                    Position = rx_var[readpoint]
                    readpoint += 1
                    Direction = rx_var[readpoint]
                    readpoint += 1
                    CountdownID = rx_var[readpoint]
                    readpoint += 1
                    channel_dict = {
                        "number": Num,
                        "source": ControlSource,
                        "type": ControlType,
                        "flash": Flash,
                        "dim": Dim,
                        "position": Position,
                        "direction": Direction,
                        "countdown": CountdownID,
                    }
                    channel_list.append(channel_dict)

                return (channel_list)

    def getCoord(self, ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.coord_frame, ip):
            if self.__rx_num == 4:
                OperationalMode = rx_var[0]
                CorrectionMode = rx_var[1]
                MaximumMode = rx_var[2]
                ForceMode = rx_var[3]
                coord_dict = {
                    "OperationalMode": OperationalMode,
                    "CorrectionMode": CorrectionMode,
                    "MaximumMode": MaximumMode,
                    "ForceMode": ForceMode,
                }
                return coord_dict

    def getOverlap(self, ip):
        rx_var = self.__rx_var
        OverlapSize = 10
        overlap_list = []
        if self.readPendingDatagrams(tramas.overlap_frame, ip):
            if 16 == rx_var[0] and self.__rx_num == OverlapSize * 16 + 1:
                readpoint = 1
                for i in range(16):
                    Num = rx_var[readpoint]
                    readpoint += 1
                    Type = rx_var[readpoint]
                    readpoint += 1
                    TrailGreen = rx_var[readpoint]
                    readpoint += 1
                    TrailClear = rx_var[readpoint]
                    readpoint += 1
                    TrailYellow = rx_var[readpoint]
                    readpoint += 1
                    TrailRed = rx_var[readpoint]
                    readpoint += 1
                    overlapDict = {
                        "Num": Num,
                        "Type": Type,
                        "TrailGreen": TrailGreen,
                        "TrailClear": TrailClear,
                        "TrailYellow": TrailYellow,
                        "TrailRed": TrailRed,
                    }
                    overlap_list.append(overlapDict)
                return (overlap_list)
    
    def getErroresControlador(self,ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.error_frame, ip):
            readpoint = 1
            error_list = []
            for i in range(16):
                reg_error={
                    "year": rx_var[readpoint],
                    "month": rx_var[readpoint+1],
                    "date": rx_var[readpoint+2]//16*10+rx_var[readpoint+2]%16,
                    "day": rx_var[readpoint+3]//16*10 +rx_var[readpoint+3]%16,
                    "hour": rx_var[readpoint+4],
                    "minute": rx_var[readpoint+5],
                    "seconds": rx_var[readpoint+6]//16*10+rx_var[readpoint+6]%16,
                    "error": rx_var[readpoint+7],
                    "grupo":rx_var[readpoint+8],
                    "reserve":rx_var[readpoint+9],
                }
                readpoint+=10
                error_list.append(reg_error)
               

            #print(rx_var)
            return (error_list)

    def getWorkState(self,ip):
        rx_var = self.__rx_var
        if self.readPendingDatagrams(tramas.workstate_frame, ip):
            work_state ={
                "horario":rx_var[0],
                "plan":rx_var[1],
                "action":rx_var[2],
                "pattern":rx_var[3],
                "seq":rx_var[4],
                "split":rx_var[5],
                "modo":rx_var[6],
                "ring1_fase":rx_var[10],
                "ring1_duracion":rx_var[11],
                "ring1_remain":rx_var[12],
                "ring2_fase":rx_var[13],
                "ring2_duracion":rx_var[14],
                "ring2_remain":rx_var[15],
                "ring3_fase":rx_var[16],
                "ring3_duracion":rx_var[17],
                "ring3_remain":rx_var[18],
                "ring4_fase":rx_var[19],
                "ring4_duracion":rx_var[20],
                "ring4_remain":rx_var[21]
            }
                
            return (work_state)

    def setUnit(self, data, ip_controller):
        gbtx = bytearray(25)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 21
        temp_var = []
        num = 11
        temp_num = 12
        for i in data:
            # coegmos los datos de la api y los convertimos en una lista para posteriormente formatear y crear la trama udp
            temp_var.append(i)
        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        return self.enviarData(gbtx, ip_controller)
    def setUnitHT200(self, data, ip_controller):
            gbtx = bytearray(27)
            # trama normal para escritura
            gbtx[0] = 192
            gbtx[1] = 32
            gbtx[2] = 32
            gbtx[3] = 16
            gbtx[5] = 1
            gbtx[6] = 1
            gbtx[7] = 0
            gbtx[10] = 1
            # trama que especifica que se van a grabar los datos en unit
            gbtx[4] = 3
            gbtx[8] = 129
            gbtx[9] = 21
            temp_var = []
            num = 11
            temp_num = 14
            for i in data:
                # coegmos los datos de la api y los convertimos en una lista para posteriormente formatear y crear la trama udp
                temp_var.append(i)
            for i in range(temp_num):
                if temp_var[i] == 0xC0:
                    gbtx[num] = 0xDB
                    num += 1
                    gbtx[num] = 0xDC
                    num += 1
                elif temp_var[i] == 0xDB:
                    gbtx[num] = 0xDB
                    num += 1
                    gbtx[num] = 0xDD
                    num += 1
                else:
                    gbtx[num] = temp_var[i]
                    num += 1
            CheckSumCalc = 0
            for i in range(1, num):
                CheckSumCalc += gbtx[i]
            CheckSumCalc = (CheckSumCalc % 256)

            if CheckSumCalc == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif CheckSumCalc == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = CheckSumCalc
                num += 1
            gbtx[num] = 192  # frame tail
            return self.enviarData(gbtx, ip_controller)

    def setFases(self, data, ip_controller):
        gbtx = bytearray(526)
        print("fases ...")
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 7
        gbtx[11] = 16
        temp_var = []
        num = 12
        temp_num = 512
        for x in data:
            for i in x:
                # cogemos los datos de la api y los convertimos en una lista para posteriormente formatear y crear la trama udp
                temp_var.append(int(i))
        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        print("se envia fases correctamente")
        return self.enviarData(gbtx, ip_controller)

    def setSecuencias(self, data, ip_controller):
        gbtx = bytearray(1118)
        print("sec ...")
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 19
        gbtx[11] = 16
        temp_var = []
        num = 12
        temp_num = 1104
        for x in data:
            temp_var.append(int(x))

        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        print("se envia secuencias correctamente")
        return self.enviarData(gbtx, ip_controller)

    def setSplit(self, data, ip_controller):
        print("split ...")
        gbtx = bytearray(1314)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 20
        gbtx[11] = 20
        temp_var = []
        num = 12
        temp_num = 1300
        for x in data:
            temp_var.append(int(x))

        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx.append(0xDC)
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx.append(0xDD)
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        print("se envia split correctamente")
        return self.enviarData(gbtx, ip_controller)

    def setPattern(self, data, ip_controller):
        print("pattern ...")
        gbtx = bytearray(714)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 8
        gbtx[11] = 100
        temp_var = []
        num = 12
        temp_num = 700
        for x in data:
            temp_var.append(int(x))

        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        print("se envia pattern correctamente")
        return self.enviarData(gbtx, ip_controller)

    def setAction(self, data, ip_controller):
        print("action ...")
        gbtx = bytearray(414)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 18
        gbtx[11] = 100
        temp_var = []
        num = 12
        temp_num = 400
        for x in data:
            temp_var.append(int(x))
        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        print("se envia accion correctamente")
        return self.enviarData(gbtx, ip_controller)

    def setPlan(self, data, ip_controller):
        print("plan ...")
        gbtx = bytearray(1182)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 17
        gbtx[11] = 16
        temp_var = []
        num = 12
        temp_num = 1168
        for x in data:
            temp_var.append(int(x))
        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail

        return self.enviarData(gbtx, ip_controller)

    def setHorarios(self, data, ip_controller):
        gbtx = bytearray(374)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit
        gbtx[4] = 3
        gbtx[8] = 129
        gbtx[9] = 9
        gbtx[11] = 40
        temp_var = []
        num = 12
        temp_num = 360
        for x in data:
            temp_var.append(int(x))
        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)

        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail

        return self.enviarData(gbtx, ip_controller)

    def setChannel(self, data, ip_controller):
        print("canal ...")
        gbtx = bytearray(142)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[4] = 3
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[10] = 1
        # trama que especifica que se van a grabar los datos en unit

        gbtx[8] = 129
        gbtx[9] = 6
        gbtx[11] = 16
        temp_var = []
        num = 12
        temp_num = 128
        for x in data:
            temp_var.append(int(x))
        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)
        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        return self.enviarData(gbtx, ip_controller)

    def setTime(self, ip_controller):
        gbtx = bytearray(21)
        # trama normal para escritura
        gbtx[0] = 192
        gbtx[1] = 32
        gbtx[2] = 32
        gbtx[3] = 16
        gbtx[4] = 2
        gbtx[5] = 1
        gbtx[6] = 1
        gbtx[7] = 0
        gbtx[8] = 129
        gbtx[9] = 5
        gbtx[10] = 1
        now = datetime.datetime.now()
        seconds = str(now.second)
        minute = str(now.minute)
        hour = str(now.hour)
        day = 4
        date = str(now.day)
        month = str(now.month)
        year = str(now.year)[2:]
        __data = [
            int(seconds, 16),
            int(minute, 16),
            int(hour, 16),
            1,
            int(date, 16),
            int(month, 16),
            int(year, 16)
        ]
        y = ((__data[6] >> 4) & 0x0f)*10+(__data[6] & 0x0f)
        m = ((__data[5] >> 4) & 0x0f)*10+(__data[5] & 0x0f)
        d = ((__data[4] >> 4) & 0x0f)*10+(__data[4] & 0x0f)
        if m < 3:
            m = m+12
            y = y-1
        a = y/4
        b = (m+1)*13/5
        c = y+a+b+d-1
        c = c % 7
        __data[3] = c
        temp_var = []
        num = 11
        temp_num = len(__data)
        for x in __data:
            temp_var.append(int(x))
        for i in range(temp_num):
            if temp_var[i] == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif temp_var[i] == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = temp_var[i]
                num += 1
        CheckSumCalc = 0
        for i in range(1, num):
            CheckSumCalc += gbtx[i]
        CheckSumCalc = (CheckSumCalc % 256)
        if CheckSumCalc == 0xC0:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDC
            num += 1
        elif CheckSumCalc == 0xDB:
            gbtx[num] = 0xDB
            num += 1
            gbtx[num] = 0xDD
            num += 1
        else:
            gbtx[num] = CheckSumCalc
            num += 1
        gbtx[num] = 192  # frame tail
        print(list(gbtx))
        return True
        print(ip_controller)
        return self.enviarData(gbtx, ip_controller)

    def setBasicPlan(self, data_target, ip):

        if self.setFases(ip_controller=ip, data=data_target['fases']) == False:
            print("fases wrong")
            return False
        elif self.setSecuencias(ip_controller=ip, data=data_target['secuencias']) == False:
            print("secuencias wrong")
            return False
        elif self.setSplit(ip_controller=ip, data=data_target['split']) == False:
            print("split wrong")
            return False
        elif self.setPattern(ip_controller=ip, data=data_target['pattern']) == False:
            print("pattern wrong")
            return False
        elif self.setAction(ip_controller=ip, data=data_target['accion']) == False:
            print("accion wrong")
            return False
        elif self.setPlan(ip_controller=ip, data=data_target['plan']) == False:
            print("plan wrong")
            return False
        elif self.setChannel(ip_controller=ip, data=data_target['channel']) == False:
            print("channel wrong")
            return False
        else:
            return True
        
    def setControlManual(self, data_target, ip_controller):
            gbtx = bytearray(19)
            # trama normal para escritura
            gbtx[0] = 192
            gbtx[1] = 32
            gbtx[2] = 32
            gbtx[3] = 16
            gbtx[4] = 4
            gbtx[5] = 1
            gbtx[6] = 1
            gbtx[7] = 0
            gbtx[8] = 129
            temp_var = []
            num = 9
            temp_num = 8
            for x in data_target:
                temp_var.append(int(x))
            for i in range(temp_num):
                if temp_var[i] == 0xC0:
                    gbtx[num] = 0xDB
                    num += 1
                    gbtx[num] = 0xDC
                    num += 1
                elif temp_var[i] == 0xDB:
                    gbtx[num] = 0xDB
                    num += 1
                    gbtx[num] = 0xDD
                    num += 1
                else:
                    gbtx[num] = temp_var[i]
                    num += 1
            CheckSumCalc = 0
            for i in range(1, num):
                CheckSumCalc += gbtx[i]
            CheckSumCalc = (CheckSumCalc % 256)
            if CheckSumCalc == 0xC0:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDC
                num += 1
            elif CheckSumCalc == 0xDB:
                gbtx[num] = 0xDB
                num += 1
                gbtx[num] = 0xDD
                num += 1
            else:
                gbtx[num] = CheckSumCalc
                num += 1
            gbtx[num] = 192  # frame tail
            print(list(gbtx))
            print(ip_controller)
            return self.enviarData(gbtx, ip_controller)


    def enviarData(self, data, ip):
        __udpsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        flag = False
        port = 13536
        while True:
            try:
                __udpsocket.bind(('0.0.0.0', port))
                __udpsocket.settimeout(10)
                __udpsocket.sendto(data, (ip, self.__port))
                data_received, sender = __udpsocket.recvfrom(2048)
                trama_respuesta = list(data_received)
                if trama_respuesta[8] == 132:
                    flag = True
                else:
                    flag = False
                self.__ips_connected.append(sender)
                break
            except OSError:
                port += 1
        __udpsocket.close()

        return flag
