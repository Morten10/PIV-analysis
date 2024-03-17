# import My_functions as Mf
# import matplotlib.pyplot as plt
# import numpy as np

# pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'
# The_big_pickle = Mf.load_pickle(pickle_location)
# keys=The_big_pickle.keys()
# keys = [key for key in keys if key != 'Object To Follow'] #remove object to follow

# has_been_changed_to_minimum = ['Experiment Date 2023/12/13 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 4', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 5', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 8', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 1', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 2', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 4', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 5', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 6', 'Experiment Date 2023/12/21 Experiment Number 1 Record Number 1', 'Experiment Date 2023/12/21 Experiment Number 1 Record Number 2', 'Experiment Date 2023/12/24 Experiment Number 1 Record Number 7', 'Experiment Date 2023/12/24 Experiment Number 1 Record Number 11', 'Experiment Date 2023/12/24 Experiment Number 1 Record Number 13', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 1', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 5', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 8', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 9', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 10', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 20', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 2', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 8', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 9', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 10', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 17', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 18', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 20', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 2', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 3', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 5', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 6', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 1', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 3', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 5', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 7', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 8', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 9', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 10', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 1', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 2', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 3', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 5', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 6', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 7', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 1', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 2', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 3', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 4', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 5', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 6', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 7', 'Experiment Date 2024/1/22 Experiment Number 2 Record Number 14', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 14', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 15', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 16', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 17', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 18', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 19']
# the_lst = []

# for key in keys:
#     current = The_big_pickle[key]
   
#     locations = np.array(current['Vertical Location For Velocity [m] list'])
#     time = current['Velocity Times [sec] list']
#     y_values = np.array(current['Velocity [m/s] list'])
#     # fps = current['Frames Per Second']
#     # dt = 1/fps
#     interface_width = current['Interface width [m]']




#     pos_vel = -y_values
#     # x_values = time/interface_width
#     x_values = time


#     fig, ax = plt.subplots()
#     def on_key(event):
#         if event.key == 'a':
#             print(f'{state} approved {key} has been changed')
#             current['State'] = state
#             the_lst.append(f'{key} {state}')
#             plt.close()
#         elif event.key == 'escape':
#             plt.close()

#     error = 0.001/2
#     errors = np.ones(len(y_values))*error
#     ax.scatter(x_values, pos_vel, color='red')
#     # ax.errorbar(x_values, pos_vel, yerr=errors, fmt='', label='Data with Error Bars')
#     ax.axhline(pos_vel[-1]-error, color='g', linestyle='--')
#     ax.axhline(min(pos_vel)+error, color='r', linestyle='--')
#     # plt.gca().invert_xaxis()

#     if pos_vel[-1]-error - min(pos_vel)+error <= error:
#         print (f'Danger Zone for {key}')
#     if min(pos_vel)<0:
#         print('bouncing')
#         state = 'bouncing'
#     elif pos_vel[-1]-error > min(pos_vel)+error:
#         print('sure - minimum')
#         state = 'minimum'
#     else:
#         print('no-minimum')
#         state = 'no-minimum'


#     fig.canvas.mpl_connect('key_press_event', on_key)
#     plt.show()


import os
import My_functions as Mf
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import plotly.express as px


plots_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Plots'
excel_path = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\Dimensionless Numbers PTV.xlsx'
pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'
the_pickle = Mf.load_pickle(pickle_location)
the_keys = the_pickle.keys()
the_keys = [key for key in the_keys if key != 'Object To Follow'] #remove object to follow



color_map = {'bouncing' : 'green', 'minimum' : 'red', 'no - minimum':'blue'}
to_show = ['Experiment Date 2024/1/24 Experiment Number 1']
to_show = []


fig, ax = plt.subplots()
for ind, key in enumerate(the_keys):
    current_date = key[:45]
    if current_date in to_show or len(to_show) == 0:
    
        current = the_pickle[key]
        time = np.array(current['Velocity Times [sec] list'])
        locc = np.array(current['Vertical Location For Velocity [m] list'])
        velocii = np.array(current['Velocity [m/s] list'])
        state = current['State']
        # print(state)


        value_to_find_1 = 0.05
        value_to_find_2 = 0.2
        index_1 = np.argmin(np.abs(locc - value_to_find_1))
        index_2 = np.argmin(np.abs(locc - value_to_find_2))
        time_scale = np.abs(time[index_1] - time[index_2])


        #from min to max:
        index_1 = np.argmax(-velocii)
        index_2 = np.argmin(-velocii)
        time_scale = np.abs(time[index_1] - time[index_2])
        # ax.scatter(ind, time_scale, color=color_map[state])
        # print(time_scale)

        #from min to v_2:
        index_11 = np.argmin(-velocii)
        index_22 = len(velocii)-1
        time_scale2 = np.abs(time[index_11] - time[index_22])



        distance = np.abs(value_to_find_2-value_to_find_1)

        # max_index = np.argmax(-velocii)
        # min_index = np.argmin(-velocii)
        # time_scale = time[min_index] - time[max_index]
        
        # Normalize velocity
        normalized_velocii = (-velocii) / max(-velocii)


        x_values = locc
        y_values = 0.1 / -velocii

        # print(len(x_values), len(normalized_velocii))
        labell = f"{current['Experiment Date']} {current['Experiment number']} {current['Record number']}    min_velocity = {min(-velocii):.4f}"

       
        if state == 'bouncing':
            continue
        else:
            ax.plot(x_values, y_values, label = labell)
            ########ax.plot(x_values, y_values, color=color_map[state], label = labell)


        if ind == 7:
            break


ax.legend()
ax.set_xlabel('vertical location [m]')
ax.set_ylabel('time to pass 0.1 meter [sec]')
plt.show()




# import os
# import numpy as np
# import My_functions as Mf
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D


# excel_path = r"C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\Dimensionless Numbers PTV.xlsx"



# state_column = 'State'
# solution_column = 'Solution'
# date_column = 'Experiment Date'

# brunt1 = 'Brunt Number'
# Re_u1 = 'Upper Re'
# Re_l1 = 'Lower Re'
# Fr_u1 = 'Upper Fr'
# Fr_l1 = 'Lower Fr'
# nu_u1 = 'Upper viscosity [m^2/sec]'
# nu_l1 = 'Lower viscosity [m^2/sec]'
# rho_u1 = 'Upper density [kg/m^3]'
# rho_l1 = 'Lower density [kg/m^3]'
# U_min1 = 'Minimum velocity [m/s]'
# sphere_diameter1 = 'Sphere Diameter [m]'
# u_upper1 = 'Upper velocity [m/s]'
# rho_s1 = 'Calculated sphere density [kg/m^3]'
# interfacey1 = 'Interface width [m]'
# traj_time1 = 'Full Trajectory Time [sec]'



# df = pd.read_excel(excel_path)
# df = df[df[Re_u1] < 600]
# df = df[df['Index'] != 160]
# # df.dropna(subset=[U_min1], inplace=True)


# color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
# df['color'] = df[state_column].map(color_mapping)
# solution_mapping = {'Water - Salt' : 'P', 'Water - Glycerol' : 'o'}
# df['marker type'] = df[solution_column].map(solution_mapping)


# x_col = Re_u1
# y_col = Re_l1

# y_label = r'$\frac {mg \cdot U_u} {\nu_u \cdot \rho_u \cdot a^2}$'
# y_label = r'$\frac {\rho_{sphere}} {\rho_{lower}}$'
# x_label = r'${Re}_{u}$'
# y_label = r'${Re}_{l}$'


# fig, ax = plt.subplots(figsize=(7, 7))

# groups = df.groupby(['color', 'marker type'])
# for (color, marker), group in groups:
#     ax.scatter(group[x_col], group[y_col], marker=marker, color=color)

# # ax.set_xscale('log')  # Set x-axis to logarithmic scale
# # ax.set_yscale('log')  # Set x-axis to logarithmic scale
# # ax.set_xlim(left=4.5)
# # ax.set_xlim(right=0.1e160)



# legend_elements = [
#     Line2D([0], [0], color='w', label='minimum'),
#     Line2D([0], [0], color='w', label='no - minimum'),
#     Line2D([0], [0], color='w', label='bouncing'),
#     Line2D([0], [0], marker=solution_mapping.get('Water - Salt'), color='w', label='Water - Salt', markerfacecolor='black', markersize=8),
#     Line2D([0], [0], marker=solution_mapping.get('Water - Glycerol'), color='w', label='Water - Glycerol', markerfacecolor='black', markersize=8),
# ]

# legend = plt.legend(handles=legend_elements, loc='best')

# # Color the legend text
# for ind, text in enumerate(legend.get_texts()):
#     if ind <=2:
#         text.set_color(color_mapping.get(text.get_text()))


# plt.xlabel(x_label)
# ylabel = plt.ylabel(y_label, rotation=0)
# ylabel.set_verticalalignment('bottom')  # Align at the bottom of the label
# ylabel.set_y(ylabel.get_position()[1] - 0.05)
# # plots_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Plots'
# # plt.savefig(os.path.join(plots_folder, 'Re map.pgf'), format='pgf')
# plt.show()




# #     current = the_pickle[el]
# #     time = np.array(current['Velocity Times [sec] list'])
# #     locc = np.array(current['Vertical Location For Velocity [m] list'])
# #     velocii = np.array(current['Velocity [m/s] list'])

# #     max_index = np.argmax(-velocii)
# #     min_index = np.argmin(-velocii)
# #     time_scale = time[min_index] - time[max_index]
# #     # Normalize velocity
# #     normalized_velocii = -velocii / max(-velocii)
# #     tmp.append(pd.DataFrame({'time/ret time': time/time_scale, 'Normalized Velocity': normalized_velocii, 'Source':el}))

# # tmp = pd.concat(tmp)

# # # and plot it:
# # fig = px.line(tmp, x='time/ret time', y='Normalized Velocity', color='Source')

# # # Customize the axes labels
# # fig.update_xaxes(title_text='Vertical Location [m]', showline=True)
# # fig.update_yaxes(title_text=r'$\frac {u} {u_{max}}$', showline=True)

# # # Invert x-axis
# # # fig.update_xaxes(autorange="reversed")
# # fig.show()