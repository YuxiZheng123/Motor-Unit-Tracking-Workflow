import sys
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.simpledialog import askstring
import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import openhdemg.library as emg

###
def load_tracking_results():
    # Configure the root Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Open a file selection dialog
    file_path = askopenfilename(
        title="Select a Tracking results file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    # Check if a file was selected
    if not file_path:
        print("No file selected.")

    # Read the CSV file
    data = pd.read_csv(file_path)
    return data

def save_dataframe_as_csv(dataframe, include_index=False):
    # Initialize the Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Configure the file save dialog
    file_path = asksaveasfilename(
        title="Save File",  # Dialog title
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),  # Define file filters
        defaultextension=".csv"  # Default file extension
    )

    # Check if a file path was chosen
    if file_path:
        # Save the DataFrame as a CSV file
        dataframe.to_csv(file_path, index=include_index)
        print(f"File saved as {file_path}")

        # Try to open the file with the system's default program
        try:
            os.startfile(file_path)  # Only for Windows
        except AttributeError:
            # For MacOS and Linux, use 'open' or 'xdg-open'
            if 'darwin' in os.sys.platform:
                os.system(f'open "{file_path}"')
            else:  # 'linux' or 'linux2' or 'linux3'
                os.system(f'xdg-open "{file_path}"')

    else:
        print("File save cancelled.")

    # Clean up and exit Tkinter
    root.destroy()

############################
class post_processing_gui():

    def __init__(
        self,
        emgfile_1,
        emgfile_2,
        start_steady,
        end_steady,
        sorted_rawemg_1,
        sorted_rawemg_2,
        data,
        n_firings=[0, 50],
        muaps_timewindow=50,
        figsize=[25, 20],
        csv_separator="\t",
    ):
        # On start, compute the necessary information
        self.start_steady=start_steady
        self.end_steady=end_steady

        self.emgfile_1 = emgfile_1
        self.emgfile_2 = emgfile_2

        # We choose relative recruitment/derecuitment thresholds because our dynamometer could not tell us the value of force.
        self.rt_dert_1=emg.compute_thresholds(self.emgfile_1, event_= 'rt_dert', type_='rel')
        self.rt_dert_2=emg.compute_thresholds(self.emgfile_2, event_= 'rt_dert', type_='rel')
        # Caculate discharge rate
        self.dr_1=emg.compute_dr(self.emgfile_1, start_steady=self.start_steady, end_steady=self.end_steady, event_='rec_derec_steady')
        self.dr_2=emg.compute_dr(self.emgfile_2, start_steady=self.start_steady, end_steady=self.end_steady, event_='rec_derec_steady')
        
        ###
        self.dd_1 = emg.double_diff(sorted_rawemg_1)
        self.st_1 = emg.sta(
            emgfile=emgfile_1,
            sorted_rawemg=self.dd_1,
            firings=n_firings,
            timewindow=muaps_timewindow,
        )
        self.sta_xcc_1 = emg.xcc_sta(self.st_1)

        ####
        self.dd_2 = emg.double_diff(sorted_rawemg_2)
        self.st_2 = emg.sta(
            emgfile=emgfile_2,
            sorted_rawemg=self.dd_2,
            firings=n_firings,
            timewindow=muaps_timewindow,
        )
        self.sta_xcc_2 = emg.xcc_sta(self.st_2)

        self.figsize = figsize
        self.csv_separator = csv_separator
        self.data=data# Tracking information

        # After that, set up the GUI
        self.root = tk.Tk()
        self.root.title('MUs Properties Caculation')
        self.should_close = False  # Add an attribute to control closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Top frame
        top_frm = tk.Frame(self.root, padx=2, pady=2)
        top_frm.pack(side=tk.TOP, fill="both", expand=True)
        # Bottom frame
        bottom_frm = tk.Frame(self.root, padx=2, pady=2,height=200)
        bottom_frm.pack_propagate(False)
        bottom_frm.pack(side=tk.TOP, fill="x", expand=False)

        # Create inner frames
        # Top left
        top_left_frm = tk.Frame(top_frm, padx=2, pady=2)
        top_left_frm.pack(side=tk.TOP, anchor="nw", fill="x")
        # Bottom left
        self.bottom_left_frm = tk.Frame(top_frm, padx=2, pady=2)
        self.bottom_left_frm.pack(
            side=tk.TOP, anchor="nw", expand=True, fill="both",
        )
        self.small_bottom_left_frm_1 = tk.Frame(self.bottom_left_frm, padx=2, pady=2)
        self.small_bottom_left_frm_2 = tk.Frame(self.bottom_left_frm, padx=2, pady=2)
        
        # Use grid instead of pack to layout these two Frames, so weights can be assigned
        self.small_bottom_left_frm_1.grid(row=0, column=0, sticky="nsew")
        self.small_bottom_left_frm_2.grid(row=0, column=1, sticky="nsew")
        # Configure the grid for bottom_left_frm and assign equal weights
        self.bottom_left_frm.grid_rowconfigure(0, weight=1)  # Set weight of row 0 to 1
        self.bottom_left_frm.grid_columnconfigure(0, weight=1)  # Set weight of column 0 to 1
        self.bottom_left_frm.grid_columnconfigure(1, weight=1)  # Also set weight of column 1 to 1

        top_bottom_frm = tk.Frame(bottom_frm, padx=2, pady=2)
        top_bottom_frm.pack(side=tk.TOP, anchor="nw", fill="x")
        bottom_bottom_frm = tk.Frame(bottom_frm, padx=2, pady=2)
        bottom_bottom_frm.pack(side=tk.TOP, anchor="nw",expand=True, fill="both")

        # Label MU pair combobox
        mupair_label = ttk.Label(top_left_frm, text="MU pair number", width=15)
        mupair_label.grid(row=0, column=0, columnspan=1, sticky=tk.W)

        # Create a combobox to change MU pair
        self.all_pairs = list(range(len(self.data)))

        self.selectmu_cb = ttk.Combobox(
            top_left_frm,
            textvariable=tk.StringVar(),
            values=self.all_pairs,
            state='readonly',
            width=15,
        )
        self.selectmu_cb.grid(row=1, column=0, columnspan=1, sticky=tk.W)
        self.selectmu_cb.current(0)

        # gui_plot() takes one positional argument (self), but the bind()
        # method is passing two arguments: the event object and the function
        # itself. Use lambda to avoid the error.
        self.selectmu_cb.bind(
            '<<ComboboxSelected>>',
            lambda event: self.gui_plot(),
        )

        # Add empty column
        emp = ttk.Label(top_left_frm, text="", width=15)
        emp.grid(row=0, column=1, columnspan=1, sticky=tk.W)

        # Create the widgets to calculate CV
        # Label and combobox to select the matrix column
        col_label = ttk.Label(top_left_frm, text="Column", width=15)
        col_label.grid(row=0, column=2, columnspan=1, sticky=tk.W)

        self.columns = list(self.st_1[0].keys())

        self.col_cb = ttk.Combobox(
            top_left_frm,
            textvariable=tk.StringVar(),
            values=self.columns,
            state='readonly',
            width=15,
        )
        self.col_cb.grid(row=1, column=2, columnspan=1, sticky=tk.W)
        self.col_cb.current(0)

        # Label and combobox to select the matrix channels
        self.rows = list(range(len(list(self.st_1[0][self.columns[0]].columns))))

        start_label = ttk.Label(top_left_frm, text="From row", width=15)
        start_label.grid(row=0, column=3, columnspan=1, sticky=tk.W)

        self.start_cb = ttk.Combobox(
            top_left_frm,
            textvariable=tk.StringVar(),
            values=self.rows,
            state='readonly',
            width=15,
        )
        self.start_cb.grid(row=1, column=3, columnspan=1, sticky=tk.W)
        self.start_cb.current(0)

        self.stop_label = ttk.Label(top_left_frm, text="To row", width=15)
        self.stop_label.grid(row=0, column=4, columnspan=1, sticky=tk.W)

        self.stop_cb = ttk.Combobox(
            top_left_frm,
            textvariable=tk.StringVar(),
            values=self.rows,
            state='readonly',
            width=15,
        )
        self.stop_cb.grid(row=1, column=4, columnspan=1, sticky=tk.W)
        self.stop_cb.current(max(self.rows))

        # Button to start CV estimation
        self.ied = emgfile_1["IED"]
        self.fsamp = emgfile_1["FSAMP"]
        button_est = ttk.Button(
            top_left_frm,
            text="Estimate",
            command=self.compute_cv,
            width=15,
        )
        button_est.grid(row=1, column=5, columnspan=1, sticky="we")

        # Configure column weights
        for c in range(6):
            if c == 1:
                top_left_frm.columnconfigure(c, weight=20)
            else:
                top_left_frm.columnconfigure(c, weight=1)

        # Create a button to copy the dataframe to clipboard
        copy_btn = ttk.Button(
            top_bottom_frm,
            text="Copy Results",
            command=self.copy_to_clipboard,
            width=15,
        )
        
        copy_btn.grid(row=0, column=1, columnspan=1, sticky=tk.W)

        # Save button
        save_btn = ttk.Button(
        top_bottom_frm,
        text="Save Results",
        command=self.save_to_excel,  
        width=15
        )
        
        save_btn.grid(row=0, column=2, columnspan=1, sticky=tk.W)

        mu_1_list = self.data.iloc[:, [1]]
        mu_2_list = self.data.iloc[:, [2]]
        # Add text frame to show the results (only CV and RMS)
        self.res_df = pd.DataFrame(
            data=0.00,
            index=self.all_pairs,
            columns=["MU_Number_1","rel_RT_1","rel_DERT_1","DR_all_1", "CV_1", "RMS_1", "XCC_1","MU_Number_2","rel_RT_2","rel_DERT_2","DR_all_2", "CV_2", "RMS_2", "XCC_2", "Column", "From_Row", "To_Row"],
        )

        self.res_df['MU_Number_1'] = mu_1_list.values
        self.res_df['MU_Number_2'] = mu_2_list.values

        self.textbox = tk.Text(bottom_bottom_frm, width=100, padx=4, pady=4)
        self.textbox.pack(side=tk.TOP, expand=True, fill="both")
        
        self.textbox.insert(
            '1.0',
            self.res_df.loc[:, ["MU_Number_1","rel_RT_1","rel_DERT_1","DR_all_1", "CV_1", "RMS_1", "XCC_1","MU_Number_2","rel_RT_2","rel_DERT_2","DR_all_2", "CV_2", "RMS_2", "XCC_2"]].to_string(
                #float_format="{:.2f}".format
            ),
        )

        # Plot MU 0 while opening the GUI,
        # this will move the GUI in the background ??.
        self.gui_plot()

        # Bring back the GUI in the foreground
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Start the main loop
        self.root.mainloop()
        
    # Define functions necessary for the GUI
    # Use empty docstrings to hide the functions from the documentation.
    def gui_plot(self):
        # Plot the MUAPs used to estimate CV.

        # Get MU pair number. It should be list type
        mu_pair = [int(self.selectmu_cb.get())]

        selected_data = self.data.iloc[mu_pair, [1, 2]]  # Select specified rows for columns 2 and 3

        # Convert the selected columns to int
        mu_1 = int(selected_data.iloc[:, 0])

        mu_2 = int(selected_data.iloc[:, 1])

        # Get the figure
        fig_1 = emg.plot_muaps_for_cv(
            sta_dict=self.st_1[mu_1],
            xcc_sta_dict=self.sta_xcc_1[mu_1],
            showimmediately=False,
            figsize=self.figsize,
        )

        # If canvas already exists, destroy it
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        # Place the figure in the GUI
        self.canvas = FigureCanvasTkAgg(fig_1, master=self.small_bottom_left_frm_1)
        self.canvas.draw_idle()  # Await resizing
        self.canvas.get_tk_widget().pack(
            expand=True, fill="both", padx=0, pady=0,
        )
        plt.close(fig_1)

        fig_2 = emg.plot_muaps_for_cv(
            sta_dict=self.st_2[mu_2],
            xcc_sta_dict=self.sta_xcc_2[mu_2],
            showimmediately=False,
            figsize=self.figsize,
        )

        # If canvas already exists, destroy it
        if hasattr(self, 'canvas2'):
            self.canvas2.get_tk_widget().destroy()

        # Place the figure in the GUI
        self.canvas2 = FigureCanvasTkAgg(fig_2, master=self.small_bottom_left_frm_2)
        self.canvas2.draw_idle()  # Await resizing
        self.canvas2.get_tk_widget().pack(
            expand=True, fill="both", padx=0, pady=0,
        )
        plt.close(fig_2)

    def copy_to_clipboard(self):
        # Copy the dataframe to clipboard in csv format.

        self.res_df.to_clipboard(excel=True, sep=self.csv_separator)

    # Define the save_to_excel function
    def save_to_excel(self):
        # Open a file save dialog to choose the location and file name
        file_path = asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            # Save the DataFrame as an Excel file
            self.res_df.to_excel(file_path, index=False)
            print(f"Results have been saved to {file_path}")

    # Define functions for cv estimation
    def compute_cv(self):
        # Compute conduction velocity.
        # Get MU pair number. It should be list type
        mu_pair = [int(self.selectmu_cb.get())]
        mu_pair_int=int(self.selectmu_cb.get())
        selected_data = self.data.iloc[mu_pair, [1, 2]]  # Select specified rows for columns 2 and 3

        # Convert the selected columns to int
        mu_1 = int(selected_data.iloc[:, 0])
        mu_2 = int(selected_data.iloc[:, 1])
        mu_1_list=[int(selected_data.iloc[:, 0])]
        mu_2_list=[int(selected_data.iloc[:, 1])]

        rt_1_mu=self.rt_dert_1.iloc[mu_1_list,[0]]
        dert_1_mu=self.rt_dert_1.iloc[mu_1_list,[1]]

        rt_2_mu=self.rt_dert_2.iloc[mu_2_list,[0]]
        dert_2_mu=self.rt_dert_2.iloc[mu_2_list,[1]]

        dr_1_mu=self.dr_1.iloc[mu_1_list,[5]]
        dr_2_mu=self.dr_2.iloc[mu_2_list,[5]]

        self.res_df.loc[mu_pair_int, "rel_RT_1"] = rt_1_mu.values
        self.res_df.loc[mu_pair_int, "rel_DERT_1"] = dert_1_mu.values
        self.res_df.loc[mu_pair_int, "DR_all_1"] = dr_1_mu.values
        
        self.res_df.loc[mu_pair_int, "rel_RT_2"] = rt_2_mu.values
        self.res_df.loc[mu_pair_int, "rel_DERT_2"] = dert_2_mu.values
        self.res_df.loc[mu_pair_int, "DR_all_2"] = dr_2_mu.values

        # Get the muaps of the selected columns.
        sig_1 = self.st_1[mu_1][self.col_cb.get()]
        sig_2 = self.st_2[mu_2][self.col_cb.get()]
        col_list = list(range(int(self.start_cb.get()), int(self.stop_cb.get())+1))

        sig_1 = sig_1.iloc[:, col_list]
        sig_2 = sig_2.iloc[:, col_list]

        # Verify that the signal is correcly oriented
        if len(sig_1) < len(sig_1.columns):
            raise ValueError(
                "The number of signals exceeds the number of samples. Verify that each row represents a signal"
            )

        if len(sig_2) < len(sig_2.columns):
            raise ValueError(
                "The number of signals exceeds the number of samples. Verify that each row represents a signal"
            )

        # Estimate CV
        cv_1 = emg.estimate_cv_via_mle(emgfile=self.emgfile_1, signal=sig_1)
        cv_2 = emg.estimate_cv_via_mle(emgfile=self.emgfile_2, signal=sig_2)

        # Calculate RMS
        sig_1 = sig_1.to_numpy()
        rms_1 = np.mean(np.sqrt((np.mean(sig_1**2, axis=0))))

        sig_2 = sig_2.to_numpy()
        rms_2 = np.mean(np.sqrt((np.mean(sig_2**2, axis=0))))

        # Update the self.res_df and the self.textbox

        #self.res_df.loc[mu_pair_int, "MU_Number_1"] = mu_1
        self.res_df.loc[mu_pair_int, "CV_1"] = cv_1
        self.res_df.loc[mu_pair_int, "RMS_1"] = rms_1

        #self.res_df.loc[mu_pair_int, "MU_Number_2"] = mu_2
        self.res_df.loc[mu_pair_int, "CV_2"] = cv_2
        self.res_df.loc[mu_pair_int, "RMS_2"] = rms_2

        xcc_col_list = list(range(int(self.start_cb.get())+1, int(self.stop_cb.get())+1))
        xcc_1 = self.sta_xcc_1[mu_1][self.col_cb.get()].iloc[:, xcc_col_list].mean().mean()
        self.res_df.loc[mu_pair_int, "XCC_1"] = xcc_1

        xcc_2 = self.sta_xcc_2[mu_2][self.col_cb.get()].iloc[:, xcc_col_list].mean().mean()
        self.res_df.loc[mu_pair_int, "XCC_2"] = xcc_2


        self.res_df.loc[mu_pair_int, "Column"] = self.col_cb.get()
        self.res_df.loc[mu_pair_int, "From_Row"] = self.start_cb.get()
        self.res_df.loc[mu_pair_int, "To_Row"] = self.stop_cb.get()

        self.textbox.replace(
            '1.0',
            'end',
            self.res_df.loc[:, ["MU_Number_1","rel_RT_1","rel_DERT_1","DR_all_1", "CV_1", "RMS_1", "XCC_1","MU_Number_2","rel_RT_2","rel_DERT_2","DR_all_2", "CV_2", "RMS_2", "XCC_2"]].to_string(
                float_format="{:.2f}".format
            ),
        )

    def on_close(self):
        # Handle window close event
        print("Cleaning up and closing the GUI.")
        self.root.destroy()
        sys.exit(0)  


class pre_processing_gui():

    def __init__(
        self,
        emgfile,
        mvc,
        increase_rate,
        steady_time,
        resting_time,
        figsize=[25, 20],
        csv_separator="\t",
    ):
        # On start, compute the necessary information
        self.mvc=mvc
        self.increase_rate=increase_rate
        self.steady_time=steady_time
        self.resting_time=resting_time

        self.start_steady=self.resting_time*2000+((self.mvc)/increase_rate)*2000
        self.end_steady=self.start_steady+steady_time*2000

        self.emgfile = emgfile
        self.figsize = figsize
        self.csv_separator = csv_separator


        # After that, set up the GUI
        self.root = tk.Tk()
        self.root.title('MUs Properties Caculation')
        self.should_close = False  # Add an attribute to control closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Top frame
        top_frm = tk.Frame(self.root, padx=2, pady=2)
        top_frm.pack(side=tk.TOP, fill="both", expand=True)
        # Bottom frame
        bottom_frm = tk.Frame(self.root, padx=2, pady=2,height=300)
        bottom_frm.pack_propagate(False)
        bottom_frm.pack(side=tk.TOP, fill="x", expand=False)

        # Create inner frames
        # Top left
        top_top_frm = tk.Frame(top_frm, padx=2, pady=2)
        top_top_frm.pack(side=tk.TOP, anchor="nw", fill="x")
        # Bottom left
        self.bottom_top_frm = tk.Frame(top_frm, padx=2, pady=2)
        self.bottom_top_frm.pack(
            side=tk.TOP, anchor="nw", expand=True, fill="both",
        )
        self.small_bottom_top_frm_1 = tk.Frame(self.bottom_top_frm, padx=2, pady=2)
        self.small_bottom_top_frm_2 = tk.Frame(self.bottom_top_frm, padx=2, pady=2)
        
        # Use grid instead of pack to layout these two Frames, so weights can be assigned
        self.small_bottom_top_frm_1.grid(row=0, column=0, sticky="nsew")
        self.small_bottom_top_frm_2.grid(row=0, column=1, sticky="nsew")
        # Configure the grid for bottom_left_frm and assign equal weights
        self.bottom_top_frm.grid_rowconfigure(0, weight=1)  # Set weight of row 0 to 1
        self.bottom_top_frm.grid_columnconfigure(0, weight=1)  # Set weight of column 0 to 1
        self.bottom_top_frm.grid_columnconfigure(1, weight=1)  # Also set weight of column 1 to 1

        top_bottom_frm = tk.Frame(bottom_frm, padx=2, pady=2)
        top_bottom_frm.pack(side=tk.TOP, anchor="nw", fill="x")
        bottom_bottom_frm = tk.Frame(bottom_frm, padx=2, pady=2)
        bottom_bottom_frm.pack(side=tk.TOP, anchor="nw",expand=True, fill="both")


        # Resize button
        Resize_label = ttk.Label(top_top_frm, text="Resize", width=15)
        Resize_label.grid(row=0, column=0, columnspan=1, sticky=tk.W)

        self.button_resize = ttk.Button(
            top_top_frm,
            text="Resize",
            command=self.Resize,
            width=15,
        )
        self.button_resize.grid(row=1, column=0, columnspan=1, sticky=tk.W)

        # Remove offset button
        Rmove_offset_label = ttk.Label(top_top_frm, text="Remove offset", width=15)
        Rmove_offset_label.grid(row=0, column=1, columnspan=1, sticky=tk.W)

        button_remove_offset = ttk.Button(
            top_top_frm,
            text="Remove offset",
            command=self.remove_offset,
            width=15,
        )
        button_remove_offset.grid(row=1, column=1, columnspan=1, sticky=tk.W)

        # Create a combobox to change MU
        MU_label = ttk.Label(top_top_frm, text="MU number", width=15)
        MU_label.grid(row=0, column=2, columnspan=1, sticky=tk.W)

        self.all_mus = list(range(self.emgfile["NUMBER_OF_MUS"]))

        self.selectmu_cb = ttk.Combobox(
            top_top_frm,
            textvariable=tk.StringVar(),
            values=self.all_mus,
            state='readonly',
            width=15,
        )
        self.selectmu_cb.grid(row=1, column=2, columnspan=1, sticky=tk.W)
        self.selectmu_cb.current(0)

        # gui_plot() takes one positional argument (self), but the bind()
        # method is passing two arguments: the event object and the function
        # itself. Use lambda to avoid the error.
        self.selectmu_cb.bind(
            '<<ComboboxSelected>>',
            lambda event: self.gui_idr_plot(),
        )

        # Delete button
        delete_label = ttk.Label(top_top_frm, text="Delete", width=15)
        delete_label.grid(row=0, column=3, columnspan=1, sticky=tk.W)

        button_delete = ttk.Button(
            top_top_frm,
            text="Delete MU",
            command=self.delete_mu,
            width=15,
        )
        button_delete.grid(row=1, column=3, columnspan=1, sticky=tk.W)

        # Save button
        Save_label = ttk.Label(top_top_frm, text="Save EMG file", width=15)
        Save_label.grid(row=0, column=4, columnspan=1, sticky=tk.W)

        button_save_emgfile = ttk.Button(
            top_top_frm,
            text="Save EMG file",
            command=self.save_emgfile,
            width=15,
        )
        button_save_emgfile.grid(row=1, column=4, columnspan=1, sticky=tk.W)

        # Create a button to copy the dataframe to clipboard
        copy_btn = ttk.Button(
            top_bottom_frm,
            text="Copy accuracy results",
            command=self.copy_to_clipboard,
            width=20,
        )
        
        copy_btn.grid(row=0, column=1, columnspan=1, sticky=tk.W)

        # Save button
        save_btn = ttk.Button(
        top_bottom_frm,
        text="Save accuracy results",
        command=self.save_to_excel,  
        width=20
        )
        
        save_btn.grid(row=0, column=2, columnspan=1, sticky=tk.W)

        # Show decomposition accuraacy
        isi = emg.compute_covisi(self.emgfile, start_steady=self.start_steady, end_steady=self.end_steady, event_="steady")
        sil = self.emgfile['ACCURACY']
        sil.columns = ['SIL']

        self.res_df = pd.concat([isi, sil], axis=1)

        self.textbox = tk.Text(bottom_bottom_frm, width=100, padx=4, pady=4)
        self.textbox.pack(side=tk.TOP, expand=True, fill="both")
        
        self.textbox.insert(
            '1.0',
            self.res_df.to_string(
                #float_format="{:.2f}".format
            ),
        )

        # Plot MU 0 while opening the GUI,
        # this will move the GUI in the background ??.
        self.gui_idr_plot()
        self.gui_mupulse_plot()

        # Bring back the GUI in the foreground
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Start the main loop
        self.root.mainloop()
    
    def Resize(self):
        root3 =tk.Tk()
        root3.withdraw()  # Hide the Tkinter main window
        resize_input = askstring("Input", "Enter the start second you want to resize:")
        if resize_input is None:
            root3.destroy()  # Destroy the temporarily created window
            return  # Exit the function without removing the offset

        start = float(resize_input)*2000
        end = (self.resting_time*2000 + (self.mvc / self.increase_rate) * 2000 + (self.mvc / self.increase_rate) * 2000 
               + self.steady_time * 2000 + self.resting_time*2000 - float(resize_input)*2000)


        # Resize EMG file
        self.emgfile, start_1, end_1 = emg.resize_emgfile(self.emgfile, area=[start, end])
        self.button_resize.config(state=tk.DISABLED)
        self.gui_idr_plot()
        self.gui_mupulse_plot()

        self.start_steady=self.resting_time*2000-float(resize_input)*2000+(self.mvc/self.increase_rate)*2000
        self.end_steady=self.start_steady+self.steady_time*2000

        isi = emg.compute_covisi(self.emgfile, start_steady=self.start_steady, end_steady=self.end_steady, event_="steady")
        sil = self.emgfile['ACCURACY']
        sil.columns = ['SIL']

        self.res_df = pd.concat([isi, sil], axis=1)

        
        self.textbox.replace(
            '1.0',
            'end',
            self.res_df.to_string(
                #float_format="{:.2f}".format
            ),
        )
    
    def remove_offset(self):
        root2 =tk.Tk()
        root2.withdraw()  # Hide the Tkinter main window
        sample_input = askstring("Input", "Enter the number of sample to use for removing offset (2000 Samples/Second):")
        if sample_input is None:
            root2.destroy()  # Destroy the temporarily created window
            return  # Exit the function without removing the offset
        sample_input=int(sample_input)
        self.emgfile = emg.remove_offset(emgfile=self.emgfile, auto=sample_input)
        self.gui_idr_plot()
        self.gui_mupulse_plot()
        root2.destroy()

    def delete_mu(self):
        mu = int(self.selectmu_cb.get())
        self.emgfile = emg.delete_mus(self.emgfile, munumber=mu)
        # Update the motor units list
        self.all_mus = list(range(self.emgfile["NUMBER_OF_MUS"]))

        # Update the dropdown values
        self.selectmu_cb['values'] = self.all_mus
        self.selectmu_cb.current(0)
        self.gui_idr_plot()
        self.gui_mupulse_plot()

        
        isi = emg.compute_covisi(self.emgfile, start_steady=self.start_steady, end_steady=self.end_steady, event_="steady")
        sil = self.emgfile['ACCURACY']
        sil.columns = ['SIL']

        self.res_df = pd.concat([isi, sil], axis=1)
        
        self.textbox.replace(
            '1.0',
            'end',
            self.res_df.to_string(
                #float_format="{:.2f}".format
            ),
        )

    def save_emgfile(self):
        emg.asksavefile(self.emgfile, compresslevel=4)


    def gui_mupulse_plot(self):

        fig_mupulse=emg.plot_mupulses(self.emgfile,showimmediately=False, figsize=self.figsize,)
        # If canvas already exists, destroy it
        if hasattr(self, 'canvas1'):
            self.canvas1.get_tk_widget().destroy()

        # Place the figure in the GUI
        self.canvas1 = FigureCanvasTkAgg(fig_mupulse, master=self.small_bottom_top_frm_1)
        self.canvas1.draw_idle()  # Await resizing
        self.canvas1.get_tk_widget().pack(
            expand=True, fill="both", padx=0, pady=0,
        )
        plt.close(fig_mupulse)

    # Define functions necessary for the GUI
    # Use empty docstrings to hide the functions from the documentation.
    def gui_idr_plot(self):
        # Plot the MUAPs used to estimate CV.

        # Get MU number
        mu = int(self.selectmu_cb.get())

        # Get the figure
        fig_idr = emg.plot_idr(self.emgfile, munumber=mu,showimmediately=False, figsize=self.figsize,)

        # If canvas already exists, destroy it
        if hasattr(self, 'canvas2'):
            self.canvas2.get_tk_widget().destroy()

        # Place the figure in the GUI
        self.canvas2 = FigureCanvasTkAgg(fig_idr, master=self.small_bottom_top_frm_2)
        self.canvas2.draw_idle()  # Await resizing
        self.canvas2.get_tk_widget().pack(
            expand=True, fill="both", padx=0, pady=0,
        )
        plt.close(fig_idr)

    def copy_to_clipboard(self):
        # Copy the dataframe to clipboard in csv format.
        self.res_df.to_clipboard(excel=True, sep=self.csv_separator)

    # Define the save_to_excel function
    def save_to_excel(self):
        # Open a file save dialog to choose the save location and file name
        file_path = asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            # Save the DataFrame as an Excel file
            self.res_df.to_excel(file_path, index=False)
            print(f"Results have been saved to {file_path}")

    def on_close(self):
        # Handle window close event
        print("Cleaning up and closing the GUI.")
        self.root.destroy()
        sys.exit(0)  