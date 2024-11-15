import openhdemg.library as emg
import custom_function as cf

###
mvc=15 # mvc level

increase_rate=5 # Force increasing rate per second
steady_time=15 # constant force period length (Steady period length)

resting_time=10 # resting time length. Here we assume the resting time is 10 second

emgfile = emg.askopenfile(filesource="OPENHDEMG")

emgfile = emg.filter_refsig(
        emgfile=emgfile,
        order=4,
        cutoff=15,
        )

emgfile=emg.sort_mus(emgfile)


gui = cf.pre_processing_gui(emgfile=emgfile,mvc=mvc,increase_rate=increase_rate,steady_time=steady_time,resting_time=resting_time)