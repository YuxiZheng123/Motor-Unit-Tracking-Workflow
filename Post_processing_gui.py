import openhdemg.library as emg
import custom_function as cf

###
mvc=15 # mvc level

increase_rate=5 # Force increasing rate per second
steady_time=15 # constant force period length (Steady period length)

resting_time=2 # resting time length. Here we assume the resting time is edit to 2 second

# Caculate start and end steady samples
start_steady = resting_time*2000 + (mvc / increase_rate) * 2000
end_steady = start_steady + steady_time * 2000

# Load edited files
emgfile_1 = emg.askopenfile(filesource="OPENHDEMG")
emgfile_2 = emg.askopenfile(filesource="OPENHDEMG")

#Load tracking results
data=cf.load_tracking_results()

# Start to analyze peripheral MUs properties (MUAP and MUCV)
# TMSi 4_8_L grid order
custom_sorting_order_TMSi_4_8_L = [
    [31, 30, 29, 28, 27, 26, 25, 24],
    [23, 22, 21, 20, 19, 18, 17, 16],
    [15, 14, 13, 12, 11, 10,  9,  8],
    [ 7,  6,  5,  4,  3,  2,  1,  0],
]

# Sort EMG signals by the order of channels of first file
sorted_rawemg_1 = emg.sort_rawemg(
     emgfile_1,
     code="Custom order",
     orientation=180,
     dividebycolumn=True,
     custom_sorting_order=custom_sorting_order_TMSi_4_8_L,
 )
### Sort EMG signals by the order of channels of second file
sorted_rawemg_2 = emg.sort_rawemg(
     emgfile_2,
     code="Custom order",
     orientation=180,
     dividebycolumn=True,
     custom_sorting_order=custom_sorting_order_TMSi_4_8_L,
 )


# Open GUI for analysis
gui = cf.post_processing_gui(
     emgfile_1=emgfile_1,
     emgfile_2=emgfile_2,
     start_steady=start_steady,
     end_steady=end_steady,
     sorted_rawemg_1=sorted_rawemg_1,
     sorted_rawemg_2=sorted_rawemg_2,
     data=data,
     n_firings=[0,50],
     muaps_timewindow=50
 )




