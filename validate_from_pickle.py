import pyperclip
import My_functions as Mf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


import os

def find_folders(path):
    # Get the list of all files and folders in the specified path
    items = os.listdir(path)

    # Filter out only the folders
    folders = [os.path.join(path, item) for item in items if os.path.isdir(os.path.join(path, item))]

    return folders


experiment_number = 1
the_bigger_folder = [
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 12',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 13',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 17',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 18\experiment 2023 12 18 - first',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 20',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 21\experiment 2023 12 21 first',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 02\2024 01 02 First 9.525 mm',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 22\P3',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 23\Round 1 - 10 mm - P2',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 24',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 26',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 27',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 28',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 08',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 21',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 22\P3',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 23\Round 1 - 10 mm - P2',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 24'
    ]



# experiment_number = 2
# the_bigger_folder = [
#     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 18\experiment 2023 12 18 - second',
#     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 21\experiment 2023 12 21 second',
#     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 02\2024 01 02 Second 9.7 mm 1300 kg m3',
#     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 22\P4',
#     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 23\Round 2 - 9.525 mm - P4 torlon'
#     ]


# experiment_number = 3
# the_bigger_folder = [
#     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 18\experiment 2023 12 18  - third',
#     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 23\Round 3 - 9.525 Nylon mm- P1',
#     ]



To_skip = [
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 20',
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 21',
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 22',
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 23',
    'Experiment Date 2023/12/20 Experiment Number 1 Record Number 1',
    'Experiment Date 2023/12/27 Experiment Number 1 Record Number 6',
    'Experiment Date 2024/1/8 Experiment Number 1 Record Number 15',
    'Experiment Date 2024/1/8 Experiment Number 1 Record Number 12',
    'Experiment Date 2024/1/24 Experiment Number 1 Record Number 10',
    'Experiment Date 2024/1/8 Experiment Number 1 Record Number 1',
    'Experiment Date 2024/1/8 Experiment Number 1 Record Number 2',
    'Experiment Date 2024/1/8 Experiment Number 1 Record Number 16',
    'Experiment Date 2024/1/8 Experiment Number 1 Record Number 18',
    'Experiment Date 2024/1/8 Experiment Number 1 Record Number 21'
    ]




save_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Python Thesis\Analysis\output'

list_of_folders = []
for fefe in the_bigger_folder:
    the_list = find_folders(fefe)
    for elele in the_list:
        list_of_folders.append(elele)
    

print(f'started at {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'
the_pickle = Mf.load_pickle(pickle_location)

###
# row number
###
experiment_number = 1
###
# row number
###


for ind,el in enumerate(list_of_folders):
    print(el)
   

    pic_folder = el
    experiment_date = Mf.read_data_from_cihx(pic_folder, '<date>')
    run_number = int(os.path.splitext(Mf.files_in_folder(pic_folder, 'cihx')[0])[0][-2:])


    read_name = f'Experiment Date {experiment_date} Experiment Number {experiment_number} Record Number {run_number}'
    if read_name in To_skip:
        if False:
            print(f'skipped {read_name}')
    
    if not read_name in To_skip:
      
  
        current = the_pickle[read_name]
        locationss = current['Sphere Locations']
        frames,x,y = locationss[:,0], locationss[:,1], locationss[:,2]
        vel_loc = current['Vertical Location For Velocity [m] list']
        vel_time = current['Velocity Times [sec] list']
        vel = current['Velocity [m/s] list']
        
        # # plot vel
        # def on_click(event):
        #     if event.inaxes is not None:
        #         print(read_name)
        #         # plt.text(event.xdata, event.ydata, 'ok', fontsize=12, color='green', ha='center')
        
        def on_key(event):
            if event.key == 'r':
                print(read_name)
                plt.close()
            elif event.key == 'escape':
                plt.close()
            
        
        y_times_cutted = vel_loc
        y_velocity_meter_cutted = vel
        if True:
            fig, ax = plt.subplots(figsize=(7, 7))
            plt.title(read_name)
            plt.scatter(y_times_cutted, y_velocity_meter_cutted, label='Skipped Data')
            plt.scatter(y_times_cutted[:5], y_velocity_meter_cutted[:5], color='red', label='Skipped Data start')


            ax.set_xlabel('time [sec]')
            ax.set_ylabel('y velocity [m/s]')
            plt.legend()
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            # fig.canvas.mpl_connect('button_press_event', on_click)
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()



        x_axis = x
        y_axis = y

        # Mf.display_images_with_points_napari(pic_folder, sphere_location_pixel, title = 'Verify sphere location')


        locccc = 0
        for inder, err in enumerate(y):
            locccc = inder
            if err > 0.20:
                break
        

        locccc2 = 0
        for inder2 in range(len(y)):
            locccc2 = inder2
            if y[::-1][inder2]<0.5:
                break

        
        if not False:            
            locccc = 0 #upper
            locccc2 = 0 #lower


        if locccc>0 or locccc2>0:
            pyperclip.copy('\n' + read_name)

            if False:
                fig, ax = plt.subplots(figsize=(7,7))
                plt.scatter(x_axis, y_axis)
                plt.scatter(x_axis[:locccc], y_axis[:locccc], color='red')
                # ax.set_xlim(0, 0.2)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                plt.title(read_name)
                plt.show()

            # print(read_name + f' number of pic {locccc}')
            print(read_name)
        #instead of print copy yo clipborad