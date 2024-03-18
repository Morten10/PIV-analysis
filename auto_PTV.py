import My_functions as Mf
import pyperclip
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


def find_folders(path):
    # Get the list of all files and folders in the specified path
    items = os.listdir(path)

    # Filter out only the folders
    folders = [os.path.join(path, item) for item in items if os.path.isdir(os.path.join(path, item))]

    return folders



# orig_the_bigger_folder = [r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 17',
#           r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 18',
#           r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 20',
#           r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 24',
#           r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 26',
#           r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 27',
#           r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 28',
#             r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 08',
#             r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 18',
#             r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 21']







# to_fix_clipboard = pyperclip.paste()
# TO_FIX=to_fix_clipboard.replace('\n','').split('\r')
# print(TO_FIX)

the_bigger_folder = [r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 22\P3',
                     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 23\Round 1 - 10 mm - P2',
                     r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 24']



the_bigger_folder = [
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 18\experiment 2023 12 18  - third',
    # r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2023 12 21\experiment 2023 12 21 second',
    # r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 02\2024 01 02 Second 9.7 mm 1300 kg m3',
    # r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 22\P4',
    r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Lab Pictures\experiment 2024 01 23\Round 3 - 9.525 Nylon mm- P1',
    ]




TO_FIX = ['Experiment Date 2024/1/8 Experiment Number 1 Record Number 7', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 10', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 11', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 13', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 14', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 15', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 18', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 22', 'Experiment Date 2024/1/8 Experiment Number 1 Record Number 26', 'Experiment Date 2024/1/21 Experiment Number 1 Record Number 1', 'Experiment Date 2024/1/21 Experiment Number 1 Record Number 3']
TO_FIX = []


list_of_folders = []
for fefe in the_bigger_folder:
    the_list = find_folders(fefe)
    for elele in the_list:
        list_of_folders.append(elele)

print(f'started at {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
save_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Python Thesis\Analysis\output'


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
    
    print(f'{os.path.basename(el)} started at {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

    experiment_date = Mf.read_data_from_cihx(pic_folder, '<date>')
    run_number = int(os.path.splitext(Mf.files_in_folder(pic_folder, 'cihx')[0])[0][-2:])


    frames_per_second = Mf.read_data_from_cihx(pic_folder, return_number = True, return_int=True) # fps
    dt = 1/frames_per_second
    print('frame per second =', frames_per_second)
    print('date of expirement : ', Mf.read_data_from_cihx(pic_folder, '<date>'))
    round_number = int(os.path.splitext(Mf.files_in_folder(pic_folder, 'cihx')[0])[0][-2:])
    print(f'Record number = {round_number} started at  {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

    Mf.create_directory(save_folder)
    Mf.copy_file(Mf.files_in_folder(pic_folder, 'cihx', True)[0], save_folder)


    picture_list = Mf.find_pictures(pic_folder)
    print(f't=0 is defined for frame number {int(os.path.splitext(os.path.basename(picture_list[0]))[0][-5:])}')
    t_0_frame = int(os.path.splitext(os.path.basename(picture_list[0]))[0][-5:])

    t_0_file_path = os.path.join(save_folder, 't_0.pkl')
    Mf.save_pickle(t_0_file_path, [t_0_frame, frames_per_second])
    unit = Mf.read_data_from_cihx(pic_folder, '<distanceUnit>')
    scaling_factor = 1/Mf.read_data_from_cihx(pic_folder, '<sizeOfPixel>', return_number = True) #[pixel/mm]
    if 'mm' in unit:
        scaling_factor *= 1000
    scaling_factor #[pixel/m]
    origin = [0, 1024]

    flagyyyy = True
    if len(TO_FIX)>0:
        flagyyyy = False
        the_name = f'Experiment Date {experiment_date} Experiment Number {experiment_number} Record Number {run_number}'
        if the_name in TO_FIX:
            flagyyyy = True
    if flagyyyy:
        

        sphere_location_pixel = Mf.track_object_series(pic_folder, fft=True, read_from_pickle_if_possible=True,
                                                        image_number = 57,
                                                        sphere_location = [501, 149],
                                                        Object_data = [[455, 102], [455, 197], [547, 197], [547, 102]],
                                                        PTV_data = [[384, 0], [384, 1024], [589, 1024], [589, 0]])
        sphere_location_meter = Mf.transform_coordinates_physical(sphere_location_pixel, pic_folder, scaling_factor, origin_point=origin)

        wanted_velocity_uncertainty = 0.001/2 #[m/s]
        location_uncertainty_pixel = 2 #[pixel]
        neighboring_points = Mf.neighboring_points_from_velocity_acceptable_error(velocity_acceptable_error = wanted_velocity_uncertainty, window_size = 0.3, dt = dt, location_uncertainty_pixel = location_uncertainty_pixel)
        # y_location_meter_cutted, y_velocity_meter_cutted = Mf.calc_velocity_least_squares_lines(sphere_location_meter[:,1], dt=dt, velocity_acceptable_error = wanted_velocity_uncertainty, location_uncertainty_pixel = location_uncertainty_pixel)
        # y_location_meter_cutted = Mf.location_with_respect_to_velocity(sphere_location_meter[:,1], dt=dt, velocity_acceptable_error=wanted_velocity_uncertainty, location_uncertainty_pixel = location_uncertainty_pixel)
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





print(f'Done at {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
