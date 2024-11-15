# Motor Unit Tracking Workflow

This project is designed for motor unit tracking and includes the workflow for motor unit tracking, comprising pre-processing, tracking-processing, and post-processing. The project requires EMG data collected using the TMSi 4_8_L HD-EMG grid. Users need to complete motor unit decomposition before proceeding with this workflow. This program has been created and tested in Python 3.10.5; please ensure this version or a later one is installed on user's device. For more detailes, user could read User Guiline in this project.

## Scripts

- **Post_processing_gui.py**: This script requires the user to load the EMG file obtained after motor unit decomposition. The user needs to modify the code to set the MVC level, the force increasing rate, the constant force period length, and the resting time period used in experiment, to match their experimental setup and obtain accurate accuracy values displayed in the GUI. This script requires the user to load the EMG file obtained after motor unit decomposition. The user needs to modify the code to set the MVC level during the contraction experiment, the force increasing rate, the constant force period length, and the resting time period before the subject contracts, to match their experimental setup and obtain accurate accuracy values displayed in the GUI.
- **MU_tracking.py**: This script is used for motor unit tracking as well as displaying and saving the tracking result files. Users need to load two EMG files to be tracked.
- **Pre_processing_gui.py**: This script generates a GUI for data analysis following motor unit tracking. Users need to load the two EMG files used for motor unit tracking as well as the motor unit tracking result file. Additionally, users need to modify the code to set the MVC level, the force increasing rate, the constant force period length, and the resting time period used in experiment, to match their experimental setup and obtain correct results. Through this GUI, users can obtain various properties of each tracked motor unit from the two EMG files used for tracking, including the relative de/recruitment threshold, average discharge rate, CV (conduction velocity), RMS, and XCC (average cross-correlation coefficient of RMS and CV).
- **custom_function.py**: The script includes the definitions (def) and the GUI class required to execute other scripts. This is the most important script!!!

## Downloading the Repository

To get started with the project, user can download from GitHub (to easily copy website address below, view this .md file at the [GitHub repository](https://github.com/YuxiZheng123/Motor-Unit-Tracking-Workflow.git)):

### Download the Repository

1. Visit the [GitHub repository](https://github.com/YuxiZheng123/Motor-Unit-Tracking-Workflow.git).
2. Click on the green **Code** button.
3. Select **Download ZIP**.
4. Extract the ZIP file to desired directory.

## Setting Up the Environment

To avoid package conflicts and ensure a consistent environment, it is recommended to create and use a virtual environment. Follow these steps:

1. **Create a Virtual Environment**:
   Open a terminal and navigate to the project directory. Run the following command to create a virtual environment named `.venv`:

   ``` bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment** (or accept IDE prompt to enter environment):
   - On **Windows**:

     ``` bash
     .venv\Scripts\activate
     ```

   - On **macOS/Linux**:

     ```bash
     source .venv/bin/activate
     ```

3. **Install Required Packages**:
   After activating the virtual environment, install all dependencies using the `requirements.txt` file provided in the project directory (this may take a minute):

   ```bash
   pip install -r requirements.txt
    ```

## Run Program

Users can directly run the three scripts, Pre_processing_gui.py, MU_tracking.py, and Post_processing_gui.py, individually based on their needs.
