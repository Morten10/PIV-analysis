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

   

the_bigger_folder = [r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 12']




To_skip = [
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 20',
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 21',
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 22',
    'Experiment Date 2023/12/13 Experiment Number 1 Record Number 23',
    'Experiment Date 2023/12/20 Experiment Number 1 Record Number 1',
    'Experiment Date 2023/12/27 Experiment Number 1 Record Number 6'
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
experiment_number = 1
for ind,el in enumerate(list_of_folders):
    # print(el)
   

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
        if False:
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

        locccc = 0
        locccc2 = 0
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
        
        sphere_location_meter = locationss[:, 1:]
        # print(current.keys())
        t_0_frame = current['First Frame']
        frames_per_second = Mf.read_data_from_cihx(pic_folder, return_number = True, return_int=True) # fps
        dt = 1/frames_per_second
        unit = Mf.read_data_from_cihx(pic_folder, '<distanceUnit>')
        scaling_factor = 1/Mf.read_data_from_cihx(pic_folder, '<sizeOfPixel>', return_number = True) #[pixel/mm]
        if 'mm' in unit:
            scaling_factor *= 1000
        scaling_factor #[pixel/m]
        origin = [0, 1024]

        # print(current.keys())
        sphere_location_pixel = current['Sphere Locations Pixel'][:, 1:]


        wanted_velocity_uncertainty = 0.001/2 #[m/s]
        location_uncertainty_pixel = 2 #[pixel]
        neighboring_points = Mf.neighboring_points_from_velocity_acceptable_error(velocity_acceptable_error = wanted_velocity_uncertainty, window_size = 0.3, dt = dt, location_uncertainty_pixel = location_uncertainty_pixel)
        y_location_meter_cutted, y_times_cutted, y_velocity_meter_cutted = Mf.calc_velocity_least_squares_lines_with_times(sphere_location_meter[:,1], dt=dt, neighboring_points=neighboring_points)
        postivie_vel = -y_velocity_meter_cutted
        row_number = experiment_number - 1
        experiment_date = Mf.read_data_from_cihx(pic_folder, '<date>')
        upper_viscosity, lower_viscosity, upper_density, lower_density, Interface_thickness, sphere_diameter, sphere_rho, sphere_type, Interface_center \
            = Mf.load_data_for_demintionless_number(experiment_date = experiment_date, row_number=row_number)

        from scipy.constants import g
        upper_velocity = -min(y_velocity_meter_cutted)
        lower_velocity = -y_velocity_meter_cutted[-1]
        min_velocity = -max(y_velocity_meter_cutted)

        upper_Re = upper_velocity*sphere_diameter/upper_viscosity
        lower_Re = lower_velocity*sphere_diameter/lower_viscosity
        Brunt_number = np.sqrt(2*g/Interface_thickness*(lower_density-upper_density)/(lower_density+upper_density))
        upper_Fr = upper_velocity/(Brunt_number*sphere_diameter)
        lower_Fr = lower_velocity/(Brunt_number*sphere_diameter)

        postivie_vel = -y_velocity_meter_cutted
        if min(postivie_vel) < 0:
            state = 'bouncing'
        else:
            if min(postivie_vel)*1.1 <= postivie_vel[-1]:
                state = 'minimum'
            else:
                state = 'no - minimum'
        print(f'does "{state}" make sense?')

        run_number = int(os.path.splitext(Mf.files_in_folder(pic_folder, 'cihx')[0])[0][-2:])
        calculated_sphere_rho = Mf.find_sphere_rho(sphere_diameter, upper_viscosity, upper_density, upper_velocity)
        print(f'''experiment_date = {experiment_date}
        run_number = {run_number}
        sphere_type = {sphere_type}
        sphere_diameter = {sphere_diameter}
        sphere_rho = {sphere_rho}
        upper_density = {upper_density}
        lower_density = {lower_density}
        upper_viscosity = {upper_viscosity}
        lower_viscosity = {lower_viscosity}
        calculated sphere density = {calculated_sphere_rho}
        upper_Re = {upper_Re}
        lower_Re = {lower_Re}
        upper_Fr = {upper_Fr}
        lower_Fr = {lower_Fr}
        Brunt_number = {Brunt_number}
        state = {state}''')


        print(f'In {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
        file_path = os.path.join(save_folder, 'data_from_PTV.pkl')
        # Mf.save_pickle(file_path, [sphere_location_meter, velocitys_meter])
        Mf.save_pickle(file_path, [y_location_meter_cutted, y_times_cutted, y_velocity_meter_cutted])


        dimensionless_numbers_path = os.path.join(save_folder, 'dimensionless_numbers_PTV.pkl')
        Mf.save_pickle(dimensionless_numbers_path, [upper_Re, lower_Re, upper_Fr, lower_Fr, Brunt_number, state])

        run_number = int(os.path.splitext(Mf.files_in_folder(pic_folder, 'cihx')[0])[0][-2:])
        dimensionless_numbers_path_excel = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\Dimensionless Numbers PTV.xlsx'


        xls = pd.ExcelFile(dimensionless_numbers_path_excel)
        selected_sheet_name = xls.sheet_names[0]
        df = pd.read_excel(dimensionless_numbers_path_excel, sheet_name=selected_sheet_name)

        new_data = {'Data Processing Date': datetime.date.today().strftime("%Y/%m/%d"), 'Data Processing Hour': datetime.datetime.now().strftime("%H:%M:%S"),
                    'Experiment Date': experiment_date, 'Record number': run_number, 'Sphere Type': sphere_type, 'Experiment number': experiment_number,
                    'Sphere Diameter [m]': sphere_diameter, 'Sphere density [kg/m^3]': sphere_rho, 'Upper density [kg/m^3]': upper_density,
                    'Lower density [kg/m^3]': lower_density, 'Upper viscosity [m^2/sec]': upper_viscosity, 'Lower viscosity [m^2/sec]': lower_viscosity,
                    'Upper velocity [m/s]': upper_velocity, 'Lower velocity [m/s]': lower_velocity,'Minimum velocity [m/s]': min_velocity,
                    'Upper Re': upper_Re, 'Lower Re': lower_Re, 'Upper Fr': upper_Fr, 'Lower Fr': lower_Fr, 'Brunt Number': Brunt_number, 'State': state,
                    'Calculated sphere density [kg/m^3]': calculated_sphere_rho, 'Interface width [m]': Interface_thickness,
                    'Full Trajectory Time [sec]': len(Mf.find_pictures(pic_folder))/frames_per_second, 'Index': len(df),
                    'Vertical Location For Velocity [m] list':np.array(y_location_meter_cutted).tolist(),
                    'Velocity Times [sec] list' : np.array(y_times_cutted).tolist(),
                    'Velocity [m/s] list': np.array(y_velocity_meter_cutted).tolist(),
                    'First Frame': t_0_frame, 'Frames Per Second': frames_per_second, 'Scaling Factor': scaling_factor}

        df.loc[len(df)] = new_data
        df.to_excel(dimensionless_numbers_path_excel, sheet_name=selected_sheet_name, index=False)
        xls.close()

        print(f'The file {os.path.basename(dimensionless_numbers_path_excel)} has been updated at: {os.path.dirname(dimensionless_numbers_path_excel)}')


        new_pickle_data = new_data.copy()
        new_column = np.arange(t_0_frame, t_0_frame + len(sphere_location_meter))
        sphere_location_meter_with_frame = np.hstack((new_column.reshape(-1, 1), sphere_location_meter))
        sphere_location_pixel_with_frame = np.hstack((new_column.reshape(-1, 1), sphere_location_pixel))

        new_pickle_data['Sphere Locations'] = sphere_location_meter_with_frame
        new_pickle_data['Sphere Locations Pixel'] = sphere_location_pixel_with_frame



        pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'
        The_big_pickle = Mf.load_pickle(pickle_location)

        The_big_pickle[f'Experiment Date {experiment_date} Experiment Number {experiment_number} Record Number {run_number}'] = new_pickle_data
        Mf.save_pickle(pickle_location, The_big_pickle)



        #instead of print copy yo clipborad