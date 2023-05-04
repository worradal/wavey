#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This file contains the code for the user interface.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "0.1.2"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__date__ = "May 04, 2023"

# built-in modules
from os import makedirs
from os.path import join, exists
import tkinter as tk
from tkinter.ttk import Progressbar
from tkinter import filedialog
import threading as th
# third-party modules


# project modules
from wavey.data import Data

'''Data types from Data class'''
RAMAN = Data.RAMAN
IR = Data.IR
UV_VIS = Data.UV_VIS


'''Create a window'''
window = tk.Tk()
window.title("Wavey")
window.geometry("500x500")

def make_progress_bar():
    '''Create a window for progress bar'''
    progress_bar_window = tk.Toplevel(window)
    progress_bar_window.title("Progress Bar")
    progress_bar_window.geometry("300x100")
    progress_bar_window.withdraw()
    '''progress bar'''
    progress_bar = Progressbar(progress_bar_window, orient="horizontal", length=200, mode="determinate",takefocus=True,maximum=100)
    progress_bar.pack()
    '''create a label for the progress bar'''
    progress_bar_label = tk.Label(progress_bar_window, text="Progress")
    progress_bar_label.pack()
    return progress_bar_window, progress_bar

def update_progress_bar(progress_bar, progress):
    '''Update the progress bar'''
    progress_bar['value'] = progress
    progress_bar.update()

def remove_progress_bar(progress_bar_window):
    '''Remove the progress bar'''
    progress_bar_window.destroy()

'''Propogate error to user'''
def error_window(error):
    error_window = tk.Toplevel(window)
    error_window.title("Error")
    error_window.geometry("300x100")
    error_window.withdraw()
    error_label = tk.Label(error_window, text=error)
    error_label.pack()
    error_window.deiconify()

'''create a button to run the program'''

def run(
        in_dir: str,
        out_dir:str,
        num_time_points: int,
        baseline_correction_method:str="None",
        weight_file:str="",
        start: int=0,
        end: int=-1,
        ftype: str=RAMAN
    ):

    try:
        if baseline_correction_method.lower() == "arpls":
            baseline_correction_configs = {
                'lambda':  100000,
                'stop_ratio': .000001, 
                'max_iters': 10000
            }

        ''' add async to the run function to make it run in the background'''
        progress_bar_window, progress_bar = make_progress_bar()
        progress_bar_window.deiconify()
        makedirs(out_dir, exist_ok=True)
        update_progress_bar(progress_bar, 10)
        original_data_fpath = join(out_dir, 'original_data.csv')
        transformed_data_fpath = join(out_dir, 'transformed_data.csv')
        phase_data_fpath = join(out_dir, 'phase_data.csv')
        spectral_data = Data(in_dir=in_dir, num_time_points=num_time_points, start=start, end=end,ftype=ftype)
        update_progress_bar(progress_bar, 25)
        print('Saving original data to ', original_data_fpath)
        spectral_data.save_to(original_data_fpath)

        if baseline_correction_method != "None":
            spectral_data.baseline_correct(
                method=baseline_correction_method, 
                configs=baseline_correction_configs)
            baseline_corrected_data_fpath = join(out_dir, 'baseline_corrected_data.csv')
            print('Saving baseline corrected data to ', baseline_corrected_data_fpath)
            spectral_data.save_to(baseline_corrected_data_fpath)
        update_progress_bar(progress_bar, 45)
        spectral_data.fourier_transform()
        update_progress_bar(progress_bar, 60)
        if exists(weight_file):
            spectral_data.weight(fpath=weight_file)
        
        spectral_data.save_phase_to(fpath=phase_data_fpath)
        update_progress_bar(progress_bar, 75)
        spectral_data.inverse_fourier_transform()
        update_progress_bar(progress_bar, 95)
        print('Saving transformed data to ', transformed_data_fpath)
        spectral_data.save_to(transformed_data_fpath)
        update_progress_bar(progress_bar, 100)
        remove_progress_bar(progress_bar_window)
    except Exception as e:
        error_window(str(e)+ " \nThis is most likely due to an incorrect input or selected file. \nPlease check your inputs and try again. Also make sure you do not have the files open in another program like Excel.")

'''threading the run command'''
def thread_run_cmd(
        in_dir: str,
        out_dir:str,
        num_time_points: int,
        baseline_correction_method:str="",
        weight_file:str="",
        start: int=0,
        end: int=-1,
        ftype: str=RAMAN
    ):
        t1 = th.Thread(target=run, args=(in_dir, out_dir, num_time_points, baseline_correction_method, weight_file, start, end, ftype))
        t1.start()

run_button = tk.Button(
    window,
    text="Run",
    command=lambda: thread_run_cmd(in_dir=text_diplay_input_dir.cget("text"),
        out_dir=text_diplay_output_dir.cget("text"),
        num_time_points=int(num_time_points.get()),
        baseline_correction_method=baseline_correction_method.get(),
        weight_file=text_diplay_wf.cget("text"),
        start=int(start.get()),
        end=int(end.get()),
        ftype=ftype.get())
)

def remove_run_button():
    run_button.pack_forget()

def pack_run_button():
    run_button.pack()


'''create a button to open a spectral files directory'''

text_diplay_input_dir = tk.Label(window, text="No Input Directory Chosen", font=("Arial", 12))

def open_dir_input():
    dir_path = tk.filedialog.askdirectory()
    in_dir = dir_path
    text_diplay_input_dir.config(text=in_dir)

def pack_open_dir_button():
    open_dir_button.pack()
open_dir_button = tk.Button(window, text="Spectra Data Directory", font=("Arial", 12), command=open_dir_input)

text_diplay_input_dir.pack()
pack_open_dir_button()

'''create a button to an output directory'''

text_diplay_output_dir = tk.Label(window, text="No Output Directory Chosen", font=("Arial", 12))
def open_dir_output():
    dir_path = tk.filedialog.askdirectory()
    out_dir = dir_path
    text_diplay_output_dir.config(text=out_dir)

text_diplay_output_dir.pack()
out_dir_button = tk.Button(window, text="Output Data Directory", font=("Arial", 12), command=open_dir_output)
out_dir_button.pack()

'''create a button to open a weighting file'''

text_diplay_wf = tk.Label(window, text="No Weighting File Chosen", font=("Arial", 12))
def open_file_weighting():
    file_path = filedialog.askopenfilename()
    weight_file = file_path
    text_diplay_wf.config(text=weight_file)
text_diplay_wf.pack()
open_file_button = tk.Button(window, text="Weight File", command=open_file_weighting)
open_file_button.pack()

'''create an input for the number of time points'''

num_time_points = tk.Entry(window)
num_time_points.insert(0, "60")
label = tk.Label(window, text="Number of Time Points",font=("Arial", 12))
label.pack()
num_time_points.pack()

'''create an input for the start frame'''

start = tk.Entry(window)
start.insert(0, "0")
label = tk.Label(window, text="Start Frame", font=("Arial", 12))
label.pack()
start.pack()

'''create an input for the end frame'''

end = tk.Entry(window)
end.insert(0, "-1")
label = tk.Label(window, text="End Frame", font=("Arial", 12))
label.pack()
end.pack()

'''create an enumeration for the baseline correction method'''

baseline_correction_method = tk.StringVar(window)
baseline_correction_method.set("None") # default value
label = tk.Label(window, text="Baseline Correction Method")
label.pack()
#TODO: make linked variables for the baseline correction method
w = tk.OptionMenu(window, baseline_correction_method, "None", "arpls") #, "polynomial")
w.pack()

'''create an enumeration for the file type'''

ftype = tk.StringVar(window)
label = tk.Label(window, text="File Type")
label.pack()
ftype.set(RAMAN) # default value

w = tk.OptionMenu(window, ftype, RAMAN, UV_VIS, IR)
w.pack()

'''Add run button to window'''
pack_run_button()
window.mainloop()
