import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter import *
import tkinter.font as font
import os
from EvaluationChain import evaluate_dataset_xy_allargs
import numpy as np
from visualize_pdf import create_pdf_report

# Parameters to be set: Threads, Input Path, OutputPath, Models, QualityMode
# Optional: YOLO conf thrsh, ss_thrsh, angle_thrsh

# Creating tkinter main window
# Exception printing
def full_stack():
    import traceback, sys
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]       # remove call of full_stack, the printed exception
                            # will contain the caught exception caller instead
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
         stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr

def execute():
    win = tk.Tk()
    win.title("DNA origami analysis")

    global yoloModel_filename, fitFile_filename, ssModel_filename, bilinearModel, yoloConf_Threshold,uNetConf_Threshold
    global cropsizeT_Threshold, filterCS, analyzeAngles, analyzeQuality
    ssModel_filename = "default"
    fitFile_filename = "default"
    yoloModel_filename = "default"
    bilinearModel = False
    yoloConf_Threshold = 0.7
    uNetConf_Threshold = 0.4
    cropsizeT_Threshold = 3.0
    filterCS = True
    analyzeAngles = True
    analyzeQuality = True
    def browseFiles_input():
        filename = filedialog.askdirectory(initialdir="/",
                                           title="Select a Folder", )
        input_filename = filename
        input_T.delete("1.0", "end")
        input_T.insert("1.0", input_filename)

    def browseFiles_output():
        filename = filedialog.askdirectory(initialdir="/",
                                           title="Select a Folder", )

        output_filename = filename
        output_T.delete("1.0", "end")
        output_T.insert("1.0", output_filename)

    def browseFiles_Fit():
        global fitFile_T, fitFile_filename

        filename = filedialog.askopenfilename(initialdir="/",
                                           title="Select a File", filetypes=[("Fit Parameters", "*.csv")])

        fitFile_filename = filename
        fitFile_T.delete("1.0", "end")
        fitFile_T.insert("1.0", fitFile_filename)

    def browseFiles_SS():
        global ssModel_T, ssModel_filename

        filename = filedialog.askopenfilename(initialdir="/",
                                           title="Select a File", filetypes=[("Pytorch Model", "*.pth")])

        ssModel_filename = filename
        ssModel_T.delete("1.0", "end")
        ssModel_T.insert("1.0", ssModel_filename)

    def browseFiles_Yolo():
        global yoloModel_T, yoloModel_filename

        filename = filedialog.askopenfilename(initialdir="/",
                                           title="Select a File", filetypes=[("YOLO Model", "*.pt")])

        yoloModel_filename = filename
        yoloModel_T.delete("1.0", "end")
        yoloModel_T.insert("1.0", yoloModel_filename)

    # Title Label

    # Creating scrolled text
    # area widget

    # Input
    label_input = Label(win,
                        text="Input Folder",
                        width=20, height=1,
                        fg="black", anchor=tk.E)
    label_input.grid(column=0, row=0)
    button_input = Button(win,
                          text="Select Input Folder",
                          command=browseFiles_input, width=20)
    button_input.grid(column=1, row=0)
    input_T = Text(win, height=1, width=40)
    input_T.insert(tk.END, "No File selected")
    input_T.grid(column=2, row=0)

    # Output
    label_output = Label(win,
                         text="Output Folder", width=20,
                         fg="black", anchor=tk.E)
    label_output.grid(column=0, row=1)
    button_output = Button(win,
                           text="Select Output Folder",
                           command=browseFiles_output, width=20)
    button_output.grid(column=1, row=1)
    output_T = Text(win, height=1, width=40)
    output_T.insert(tk.END, "No File selected")
    output_T.grid(column=2, row=1)

    # Threads
    label_threads = Label(win, width=20,
                          text="Threads", height=1,
                          fg="black", anchor=tk.E)
    label_threads.grid(column=0, row=2)
    threads_box = Spinbox(win, from_=1, to=os.cpu_count(), width=20)

    threads_box.grid(column=1, row=2)
    threads_box.delete(0, "end")
    threads_box.insert(0, os.cpu_count())

    # Angle Threshold
    label_angle = Label(win, width=20,
                        text="Angle Threshold (deg)",
                        height=1,
                        fg="black", anchor=tk.E)
    label_angle.grid(column=0, row=3)
    angleTXT = Spinbox(win, from_=1, to=45, width=20)

    angleTXT.grid(column=1, row=3)
    angleTXT.delete(0, "end")
    angleTXT.insert(0, 30)





    show_advanced=False
    advanced_windows = []

    def bilinearSS_ticked():
        global bilinearModel
        bilinearModel = bilinearSS_var.get()

    def filterCS_ticked():
        global filterCS
        filterCS = filterCS_var.get()

    def yoloSlider_changed(event):
        global yoloConf_Threshold, yoloConf_Slider
        yoloConf_Threshold = yoloConf_Slider.get()

    def uNetSlider_changed(event):
        global uNetConf_Threshold, uNetConf_Slider
        uNetConf_Threshold = uNetConf_Slider.get()

    def cropsizeTSlider_changed(event):
        global cropsizeT_Threshold, cropsizeT_Slider
        cropsizeT_Threshold = cropsizeT_Slider.get()

    def aQuality_ticked():
        global analyzeQuality, analyzeQuality_var
        analyzeQuality = analyzeQuality_var.get()

    def aAngles_ticked():
        global analyzeAngles, analyzeAngles_var
        analyzeAngles = analyzeAngles_var.get()


    # Advanced Options
    def advanced_options_ticked():
        global ssModel_label, ssModel_button, ssModel_T, ssModel_filename
        global fitFile_label, fitFile_button, fitFile_T, fitFile_filename
        global yoloModel_label, yoloModel_button, yoloModel_T, yoloModel_filename
        global bilinearSS_var, yoloConf_Threshold, yoloConf_Slider
        global uNetConf_Threshold, uNetConf_Slider
        global cropsizeT, cropsizeT_Threshold, cropsizeT_Slider
        global filterCS_button, filterCS_var, analyzeAngles_var, analyzeQuality_var

        if adv_opt_var.get():
            adv_button.config(state=tk.DISABLED)
            fitFile_label = Label(win,
                                  text="Fit Parameters File",
                                  width=20, height=1,
                                  fg="black", anchor=tk.E)
            fitFile_label.grid(column=0, row=6)
            fitFile_button = Button(win,
                                    text="Select File", width=20,
                                    command=browseFiles_Fit)
            fitFile_button.grid(column=1, row=6)
            fitFile_T = Text(win, height=1, width=40)
            fitFile_T.insert(tk.END, fitFile_filename)
            fitFile_T.grid(column=2, row=6)


            yoloModel_label = Label(win,
                                  text="YOLO Model File",
                                  width=20, anchor=tk.E, height=1,
                                  fg="black")
            yoloModel_label.grid(column=0, row=7)
            yoloModel_button = Button(win, width=20,
                                    text="Select File",
                                    command=browseFiles_Yolo)
            yoloModel_button.grid(column=1, row=7)
            yoloModel_T = Text(win, height=1, width=40)
            yoloModel_T.insert(tk.END, yoloModel_filename)
            yoloModel_T.grid(column=2, row=7)


            ssModel_label = Label(win,
                                text="U-Net Model File",
                                width=20, anchor=tk.E, height=1,
                                fg="black")
            ssModel_label.grid(column=0, row=8)
            ssModel_button = Button(win, width=20,
                                  text="Select File",
                                  command=browseFiles_SS)
            ssModel_button.grid(column=1, row=8)
            ssModel_T = Text(win, height=1, width=40)
            ssModel_T.insert(tk.END, ssModel_filename)
            ssModel_T.grid(column=2, row=8)

            bilinearSS_var = tk.BooleanVar(value=False)
            bilinearSS_button = tk.Checkbutton(win, text="Loaded Model is bilinear", variable=bilinearSS_var, onvalue=True,
                                        offvalue=False, command=bilinearSS_ticked)
            bilinearSS_button.grid(column=1, row=9)

            yoloConf_Label = Label(win,
                                text="YOLO confidence",
                                width=20, anchor=tk.E, height=1,
                                fg="black")
            yoloConf_var = tk.DoubleVar(value=yoloConf_Threshold)
            yoloConf_Label.grid(column=0, row=10)
            yoloConf_Slider = tk.Scale(win, from_=0.0, to=1.0, width=20, orient="horizontal", variable=yoloConf_var, command=yoloSlider_changed, resolution=0.001)
            yoloConf_Slider.grid(column=1, row=10)

            uNetConf_Label = Label(win,
                                   text="uNet threshold",
                                   width=20, anchor=tk.E, height=1,
                                   fg="black")
            uNetConf_var = tk.DoubleVar(value=uNetConf_Threshold)
            uNetConf_Label.grid(column=0, row=11)
            uNetConf_Slider = tk.Scale(win, from_=0.0, to=1.0, width=20, orient="horizontal", variable=uNetConf_var,
                                       command=uNetSlider_changed, resolution=0.001)
            uNetConf_Slider.grid(column=1, row=11)

            cropsizeT_Label = Label(win,
                                   text="Filter cropped size sigma",
                                   width=20, anchor=tk.E, height=1,
                                   fg="black")
            cropsizeT_var = tk.DoubleVar(value=cropsizeT_Threshold)
            cropsizeT_Label.grid(column=0, row=12)
            cropsizeT_Slider = tk.Scale(win, from_=0.0, to=10.0, width=20, orient="horizontal", variable=cropsizeT_var,
                                       command=cropsizeTSlider_changed, resolution=0.1)
            cropsizeT_Slider.grid(column=1, row=12)

            filterCS_var = tk.BooleanVar(value=True)
            filterCS_button = tk.Checkbutton(win, text="Filter cropped size", variable=filterCS_var,
                                               onvalue=True,
                                               offvalue=False, command=filterCS_ticked)
            filterCS_button.grid(column=1, row=13)

            analyzeQuality_var = tk.BooleanVar(value=True)
            analyzeQuality_button = tk.Checkbutton(win, text="Re-evaluate quality", variable=analyzeQuality_var,
                                             onvalue=True,
                                             offvalue=False, command=aQuality_ticked)
            analyzeQuality_button.grid(column=1, row=14)

            analyzeAngles_var = tk.BooleanVar(value=True)
            analyzeAngles_button = tk.Checkbutton(win, text="Analyze Angles", variable=analyzeAngles_var,
                                                   onvalue=True,
                                                   offvalue=False, command=aAngles_ticked)
            analyzeAngles_button.grid(column=1, row=15)

        win.update()

    adv_opt_var = tk.BooleanVar(value=False)
    adv_button = tk.Checkbutton(win, text="Show advanced options", variable=adv_opt_var, onvalue=True, offvalue=False, command=advanced_options_ticked)
    adv_button.grid(column=1, row=4)




    def start_run():
        threads = int(threads_box.get())
        angles = np.pi * float(angleTXT.get()) / 180
        print("Using Threads: ", threads)
        print("Angle Thrsh: ", angles)
        folder = input_T.get(1.0, "end-1c")
        resfolder = output_T.get(1.0, "end-1c")
        global filterCS_var, bilinearSS_var, analyzeAngles_var, analyzeQuality_var

        if not adv_opt_var.get():
            arguments = {"folder": folder,
                         "resfolder": resfolder,
                         "threads": threads,
                         "angle_thrsh": angles
                         }
        else:
            arguments = {"folder": folder,
                     "resfolder": resfolder,
                     "threads": threads,
                     "angle_thrsh": angles,
                     "yolo_conf_thrsh": yoloConf_Threshold,
                     "ss_thrsh": uNetConf_Threshold,
                     "crpsize_stds": cropsizeT_Threshold,
                     "filter_cropsize": filterCS_var.get(),
                     "bilinear_model": bilinearSS_var.get(),
                     "perform_angleAnalysis": analyzeAngles_var.get(),
                     "perform_qualityAnalysis": analyzeQuality_var.get(),
                     "labels": None,
                     "use_UNet": True
                     }

            yoloModel_filename = yoloModel_T.get(1.0, "end-1c")
            if yoloModel_filename != "default":
                arguments["yolo_model"] = yoloModel_filename

            ssModel_filename = ssModel_T.get(1.0, "end-1c")
            if ssModel_filename != "default":
                arguments["ss_model"] = ssModel_filename

            fitFile_filename = fitFile_T.get(1.0, "end-1c")
            if fitFile_filename != "default":
                arguments["fit_params"] = fitFile_filename


        try:
            avg, stdmwt, resf = evaluate_dataset_xy_allargs(**arguments)
            create_pdf_report(resfolder)
            os.startfile(os.path.join(resfolder, "Report.pdf"))
            tk.messagebox.showinfo(title="Evaluation finished", message=f"Evaluation finished!\nResult:\nAvg: {avg:.2f}nm, Std.deriv(Avg): {stdmwt:.2f}nm\nResults saved in {resf}")
        except Exception as e:
            tk.messagebox.showerror(title="Evaluation failed", message=f"Evaluation failed.\n"+full_stack())
            print(e)
        win.destroy()


    startfont = font.Font(weight="bold")
    button_start = Button(win, width=60,
                          text="Start Analysis",
                          command=start_run, font=startfont)
    button_start.grid(column=0, row=100, columnspan=3)

    # Placing cursor in the text area
    win.mainloop()

if __name__ == "__main__":
    while True:
        execute()
