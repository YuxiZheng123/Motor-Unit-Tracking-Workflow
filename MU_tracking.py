import openhdemg.library as emg
import custom_function as cf

###
emgfile_1 = emg.askopenfile(filesource="OPENHDEMG")
emgfile_2 = emg.askopenfile(filesource="OPENHDEMG")


custom_sorting_order_TMSi_4_8_L = [
    [31, 30, 29, 28, 27, 26, 25, 24],
    [23, 22, 21, 20, 19, 18, 17, 16],
    [15, 14, 13, 12, 11, 10,  9,  8],
    [ 7,  6,  5,  4,  3,  2,  1,  0],
]

tracking_res = emg.tracking(
    emgfile_1,
    emgfile_2,
    firings="all",
    derivation="sd",
    timewindow=50,
    threshold=0.7,
    matrixcode="Custom order",
    custom_sorting_order=custom_sorting_order_TMSi_4_8_L,
    orientation=180,
    filter=True,
    show=False,
    gui=False
)

cf.save_dataframe_as_csv(tracking_res, include_index=True)
