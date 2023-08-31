#tiempo
time_frame = bytearray()
time_frame.append(192)  #0
time_frame.append(32)  #1
time_frame.append(32) #2
time_frame.append(16) #3
time_frame.append(2) #4
time_frame.append(1) #5
time_frame.append(1) #6
time_frame.append(1) #7
time_frame.append(128) #8
time_frame.append(5) #9
time_frame.append(1) #10
time_frame.append(219) #11
time_frame.append(221) #12
time_frame.append(192) #13
#fases
fases_frame = bytearray()
fases_frame.append(192)  #0
fases_frame.append(32)  #1
fases_frame.append(32) #2
fases_frame.append(16) #3
fases_frame.append(3) #4
fases_frame.append(1) #5
fases_frame.append(1) #6
fases_frame.append(0) #7
fases_frame.append(128) #8
fases_frame.append(7) #9
fases_frame.append(1) #10
fases_frame.append(221) #11
fases_frame.append(192) #12
#search sequence

secuence_frame = bytearray() #0
secuence_frame.append(192) #0
secuence_frame.append(32)  #0
secuence_frame.append(32) #0
secuence_frame.append(16) #0
secuence_frame.append(3) #0
secuence_frame.append(1) #0
secuence_frame.append(1) #0
secuence_frame.append(0) #0
secuence_frame.append(128) #0
secuence_frame.append(19) #0
secuence_frame.append(1) #0
secuence_frame.append(233) #0
secuence_frame.append(192) #0

#search split
split_frame = bytearray()
split_frame.append(192) #0
split_frame.append(32) #0
split_frame.append(32) #0
split_frame.append(16) #0
split_frame.append(3) #0
split_frame.append(1) #0
split_frame.append(1) #0
split_frame.append(0) #0
split_frame.append(128) #0
split_frame.append(20) #0
split_frame.append(1) #0
split_frame.append(234) #0
split_frame.append(192) #0


#search pattern
pattern_frame = bytearray()
pattern_frame.append(192)  #0
pattern_frame.append(32)  #1
pattern_frame.append(32) #2
pattern_frame.append(16) #3
pattern_frame.append(3) #4
pattern_frame.append(1) #5
pattern_frame.append(1) #6
pattern_frame.append(0) #7
pattern_frame.append(128) #8
pattern_frame.append(8) #9
pattern_frame.append(1) #10
pattern_frame.append(222) #11
pattern_frame.append(192) #12


#basic_info 
basic_info_frame = bytearray()
basic_info_frame.append(192)  #0
basic_info_frame.append(32)  #1
basic_info_frame.append(32) #2
basic_info_frame.append(16) #3
basic_info_frame.append(2) #4
basic_info_frame.append(1) #5
basic_info_frame.append(1) #6
basic_info_frame.append(1) #7
basic_info_frame.append(128) #8
basic_info_frame.append(189) #9
basic_info_frame.append(1) #10
basic_info_frame.append(147) #11
basic_info_frame.append(192) #12

#device_info 
device_info_frame = bytearray()
device_info_frame.append(192)  #0
device_info_frame.append(32)  #1
device_info_frame.append(32) #2
device_info_frame.append(16) #3
device_info_frame.append(2) #4
device_info_frame.append(1) #5
device_info_frame.append(1) #6
device_info_frame.append(1) #7
device_info_frame.append(128) #8
device_info_frame.append(190) #9
device_info_frame.append(1) #10
device_info_frame.append(148) #11
device_info_frame.append(192) #12


#search horarios 
schedule_frame = bytearray()
schedule_frame.append(192)  #0
schedule_frame.append(32)  #1
schedule_frame.append(32) #2
schedule_frame.append(16) #3
schedule_frame.append(3) #4
schedule_frame.append(1) #5
schedule_frame.append(1) #6
schedule_frame.append(1) #7
schedule_frame.append(128) #8
schedule_frame.append(9) #9
schedule_frame.append(1) #10
schedule_frame.append(224) #11
schedule_frame.append(192) #12

#search plan 
plan_frame = bytearray()
plan_frame.append(192)  #0
plan_frame.append(32)  #1
plan_frame.append(32) #2
plan_frame.append(16) #3
plan_frame.append(3) #4
plan_frame.append(1) #5
plan_frame.append(1) #6
plan_frame.append(1) #7
plan_frame.append(128) #8
plan_frame.append(17) #9
plan_frame.append(1) #10
plan_frame.append(232) #11
plan_frame.append(192) #12

#search action
action_frame = bytearray()
action_frame.append(192)  #0
action_frame.append(32)  #1
action_frame.append(32) #2
action_frame.append(16) #3
action_frame.append(3) #4
action_frame.append(1) #5
action_frame.append(1) #6
action_frame.append(1) #7
action_frame.append(128) #8
action_frame.append(18) #9
action_frame.append(1) #10
action_frame.append(233) #11
action_frame.append(192) #12


#search unit
unit_frame = bytearray()
unit_frame.append(192)  #0
unit_frame.append(32)  #1
unit_frame.append(32) #2
unit_frame.append(16) #3
unit_frame.append(3) #4
unit_frame.append(1) #5
unit_frame.append(1) #6
unit_frame.append(0) #7
unit_frame.append(128) #8
unit_frame.append(21) #9
unit_frame.append(1) #10
unit_frame.append(235) #11
unit_frame.append(192) #12


#read chanel o grupos
chanel_frame = bytearray()
chanel_frame.append(192)  #0
chanel_frame.append(32)  #1
chanel_frame.append(32) #2
chanel_frame.append(16) #3
chanel_frame.append(3) #4
chanel_frame.append(1) #5
chanel_frame.append(1) #6
chanel_frame.append(0) #7
chanel_frame.append(128) #8
chanel_frame.append(6) #9
chanel_frame.append(1) #10
chanel_frame.append(220) #11
chanel_frame.append(192) #12

#read coord
coord_frame = bytearray()
coord_frame.append(192)  #0
coord_frame.append(32)  #1
coord_frame.append(32) #2
coord_frame.append(16) #3
coord_frame.append(3) #4
coord_frame.append(1) #5
coord_frame.append(1) #6
coord_frame.append(0) #7
coord_frame.append(128) #8
coord_frame.append(22) #9
coord_frame.append(1) #10
coord_frame.append(236) #11
coord_frame.append(192) #12

#read overlaap
overlap_frame = bytearray()
overlap_frame.append(192)  #0
overlap_frame.append(32)  #1
overlap_frame.append(32) #2
overlap_frame.append(16) #3
overlap_frame.append(3) #4
overlap_frame.append(1) #5
overlap_frame.append(1) #6
overlap_frame.append(0) #7
overlap_frame.append(128) #8
overlap_frame.append(23) #9
overlap_frame.append(1) #10
overlap_frame.append(237) #11
overlap_frame.append(192) #12


#registro de errores frame
error_frame = bytearray()
error_frame.append(192)  #0
error_frame.append(32)  #1
error_frame.append(32)  #2
error_frame.append(16)  #3
error_frame.append(3)  #4
error_frame.append(1)  #5
error_frame.append(1)  #6
error_frame.append(0)  #7
error_frame.append(128)  #8
error_frame.append(11)  #9
error_frame.append(255)  #10
error_frame.append(223)  #11
error_frame.append(192)  #12


#work state frame
workstate_frame = bytearray()
workstate_frame.append(192)  #0
workstate_frame.append(32)  #1
workstate_frame.append(32)  #2
workstate_frame.append(16)  #3
workstate_frame.append(2)  #4
workstate_frame.append(1)  #5
workstate_frame.append(1)  #6
workstate_frame.append(0)  #7
workstate_frame.append(128)  #8
workstate_frame.append(3)  #9
workstate_frame.append(1)  #10
workstate_frame.append(216)  #11
workstate_frame.append(192)  #12



#search ips frame
search_ips_frame = bytearray()
search_ips_frame.append(192)  #0
search_ips_frame.append(32)  #1
search_ips_frame.append(32) #2
search_ips_frame.append(16) #3
search_ips_frame.append(1) #4
search_ips_frame.append(0) #5
search_ips_frame.append(255) #6
search_ips_frame.append(255) #7
search_ips_frame.append(128) #8
search_ips_frame.append(191) #9
search_ips_frame.append(1) #10
search_ips_frame.append(143) #11
search_ips_frame.append(192) #12

