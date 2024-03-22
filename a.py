import os
import My_functions as Mf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.constants import g, pi


plots_folder = '..\\Plots'
excel_path = '.\\Analysis\\Excel Files\\Dimensionless Numbers PTV.xlsx'





state_column = 'State'
solution_column = 'Solution'
date_column = 'Experiment Date'

brunt1 = 'Brunt Number'
Re_u1 = 'Upper Re'
Re_l1 = 'Lower Re'
Fr_u1 = 'Upper Fr'
Fr_l1 = 'Lower Fr'
nu_u1 = 'Upper viscosity [m^2/sec]'
nu_l1 = 'Lower viscosity [m^2/sec]'
rho_u1 = 'Upper density [kg/m^3]'
rho_l1 = 'Lower density [kg/m^3]'
U_min1 = 'Minimum velocity [m/s]'
sphere_diameter1 = 'Sphere Diameter [m]'
u_upper1 = 'Upper velocity [m/s]'
rho_s1 = 'Calculated sphere density [kg/m^3]'
interfacey1 = 'Interface width [m]'
traj_time1 = 'Full Trajectory Time [sec]'



df = pd.read_excel(excel_path)
df = df[df[state_column] != 'no - minimum']


# df.dropna(subset=[U_min1], inplace=True)


color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
df['color'] = df[state_column].map(color_mapping)
solution_mapping = {'Water - Salt' : 'P', 'Water - Glycerol' : 'o'}
df['marker type'] = df[solution_column].map(solution_mapping)


x_col = rho_u1
y_col = rho_l1

y_label = r'$\frac {mg \cdot U_u} {\nu_u \cdot \rho_u \cdot a^2}$'
y_label = r'$\frac {\rho_{sphere}} {\rho_{lower}}$'
x_label = r'${\rho}_{u}$'
y_label = r'${\rho}_{l}$'



# fig, ax = plt.subplots(figsize=(7, 7))



# Create a figure and axis with specified size
fig, ax = plt.subplots(figsize=(7, 7))

# Create a 3D scatter plot
ax = fig.add_subplot(111, projection='3d')

groups = df.groupby(['color', 'marker type'])
for (color, marker), group in groups:
    ax.scatter(group[x_col], group[y_col],group[rho_s1], marker=marker, color=color)
    # ax.scatter(group[x_col], group[y_col], marker=marker, color=color)


legend_elements = [
    Line2D([0], [0], color='w', label='minimum'),
    Line2D([0], [0], color='w', label='no - minimum'),
    Line2D([0], [0], color='w', label='bouncing'),
    Line2D([0], [0], marker=solution_mapping.get('Water - Salt'), color='w', label='Water - Salt', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker=solution_mapping.get('Water - Glycerol'), color='w', label='Water - Glycerol', markerfacecolor='black', markersize=8),
]

legend = plt.legend(handles=legend_elements, loc='best', framealpha = 0)

# Color the legend text
for ind, text in enumerate(legend.get_texts()):
    if ind <=2:
        text.set_color(color_mapping.get(text.get_text()))


plt.xlabel(x_label)
ylabel = plt.ylabel(y_label, rotation=0)
ylabel.set_verticalalignment('bottom')  # Align at the bottom of the label
ylabel.set_y(ylabel.get_position()[1] - 0.05)

plt.show()
stop 

import My_functions as Mf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Your code for loading data and calculations...
pickle_location = '.\\Analysis\\\\Excel Files\\The Big Pickle.pkl'
the_pickle = Mf.load_pickle(pickle_location)
current = the_pickle['Experiment Date 2023/12/13 Experiment Number 1 Record Number 1']
# print(current.keys())
times = current['Velocity Times [sec] list']
v_b = np.array(current['Velocity [m/s] list'])
dt = 1/current['Frames Per Second']
a = current['Sphere Diameter [m]']
z_0 = current['Vertical Location For Velocity [m] list'][0]



r1 = np.linspace(-0.15, 0.15, 100)
z1 = np.linspace(0, 0.35, 100)
r,z = np.meshgrid(r1, z1)


# fig, ax = plt.subplots(figsize=(8,8))
fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.1)  # Adjust the bottom to make room for the slider

time_index = 0
z_bar = z-(z_0 + np.trapz(v_b[:time_index], times[:time_index]))
vb = v_b[time_index] #The velocity at a specific time. if we dont do that we have 3 dimentions, Z R T
dzdt = vb*a**3*(-2*z_bar**2+r**2)/(2*(z_bar**2+r**2)**(5/2))
drdt = -3*vb*a**3*z_bar*r/(2*(z_bar**2+r**2)**(5/2))

# Create initial quiver plot
quiv = ax.quiver(r, z, drdt, dzdt)

plt.xlabel('r [m]')
plt.ylabel('z [m]')
plt.title('Velocity Field in Cylindrical Coordinates')

# Add slider
ax_times = plt.axes([0.2, 0.01, 0.6, 0.03])  # [left, bottom, width, height]
slider_time = Slider(ax_times, 'Time', times[0], times[-1], valinit=times[0])

def update(val):
    new_time = slider_time.val  # Get the new time from the slider
    time_index = np.abs(times - np.float64(new_time)).argmin()  # Find the index corresponding to the new time
    z_bar = z-(z_0 + np.trapz(v_b[:time_index], times[:time_index]))
    vb = v_b[time_index] 
    dzdt = vb*a**3*(-2*z_bar**2+r**2)/(2*(z_bar**2+r**2)**(5/2))
    drdt = -3*vb*a**3*z_bar*r/(2*(z_bar**2+r**2)**(5/2))
    quiv.set_UVC(drdt, dzdt)  # Update velocity components
    fig.canvas.draw_idle()  # Redraw the plot

slider_time.on_changed(update)  # Call update function when slider value changes

plt.show()

stop

import os
import My_functions as Mf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.constants import g, pi


excel_path = '.\\Analysis\\Excel Files\\Dimensionless Numbers PTV.xlsx'
# print(os.path.abspath(excel_path))
# s

state_column = 'State'
solution_column = 'Solution'
date_column = 'Experiment Date'

brunt1 = 'Brunt Number'
Re_u1 = 'Upper Re'
Re_l1 = 'Lower Re'
Fr_u1 = 'Upper Fr'
Fr_l1 = 'Lower Fr'
nu_u1 = 'Upper viscosity [m^2/sec]'
nu_l1 = 'Lower viscosity [m^2/sec]'
rho_u1 = 'Upper density [kg/m^3]'
rho_l1 = 'Lower density [kg/m^3]'
U_min1 = 'Minimum velocity [m/s]'
sphere_diameter1 = 'Sphere Diameter [m]'
u_upper1 = 'Upper velocity [m/s]'
rho_s1 = 'Calculated sphere density [kg/m^3]'
interfacey1 = 'Interface width [m]'
traj_time1 = 'Full Trajectory Time [sec]'



df = pd.read_excel(excel_path)
df = df[df[rho_s1] < 1160]
# df = df[df[solution_column] != 'Water - Glycerol']
# df = df[df[state_column] != 'no - minimum']
# df = df[df[Re_u1] > 100]

# df = df[df[Re_l1] < 70]

# df.dropna(subset=[U_min1], inplace=True)


color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
df['color'] = df[state_column].map(color_mapping)
solution_mapping = {'Water - Salt' : 'P', 'Water - Glycerol' : 'o'}
df['marker type'] = df[solution_column].map(solution_mapping)


x_col = rho_l1
y_col = rho_s1


x_label = r'$\rho_{l}$'
y_label = r'$\rho_{s}$'


fig, ax = plt.subplots(figsize=(7, 7))

groups = df.groupby(['color', 'marker type'])
for (color, marker), group in groups:
    ax.scatter(group[x_col], group[y_col], marker=marker, color=color)
    # ax.scatter(group[x_col]/group[rho_u1], group[y_col]/group[rho_u1], marker=marker, color=color)


legend_elements = [
    Line2D([0], [0], color='w', label='minimum'),
    Line2D([0], [0], color='w', label='no - minimum'),
    Line2D([0], [0], color='w', label='bouncing'),
    Line2D([0], [0], marker=solution_mapping.get('Water - Salt'), color='w', label='Water - Salt', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker=solution_mapping.get('Water - Glycerol'), color='w', label='Water - Glycerol', markerfacecolor='black', markersize=8),
]

legend = plt.legend(handles=legend_elements, loc='best', framealpha = 0)

# Color the legend text
for ind, text in enumerate(legend.get_texts()):
    if ind <=2:
        text.set_color(color_mapping.get(text.get_text()))

# y=1.03x-0.0295
        
x1_value = np.linspace(min(df[x_col]), max(df[x_col]), 50)

line_1_label = r'$\rho^{*}_{s} = 1.03 \rho_{l} - 0.0295$'
color_1 = 'black'
y1_value = 1.03*x1_value-30
y1_value = 0.9*x1_value+112
y1_value = 1.03*x1_value-0.0295
ax.plot(x1_value, y1_value, linestyle='--',color=color_1, label=line_1_label)
ax.annotate(line_1_label, xy=(x1_value[20], y1_value[20]), color=color_1,xytext=(-50, 50), textcoords='offset points', arrowprops=dict(arrowstyle='->', color=color_1))


# y = 1.0815ρ2 – 0.0815
line_2_label = r'$\overline{\rho}_{s} = 1.0815 \rho_{l} - 0.0815$'
color_2 = 'red'
y2_value = 1.0815*x1_value-80
y2_value = 1.0815*x1_value-0.0815
ax.plot(x1_value, y2_value,color=color_2,label=line_2_label)
ax.annotate(line_2_label, xy=(x1_value[20], y2_value[20]), xytext=(-50, 50),color=color_2, textcoords='offset points', arrowprops=dict(arrowstyle='->',color=color_2))


#my line
y3_value = 1.0815*x1_value-85
color_3='gray'
line_3_label = 'my line'
ax.plot(x1_value, y3_value, linestyle='-.',color=color_3, label=line_3_label)
ax.annotate(line_3_label, xy=(x1_value[20], y3_value[20]), xytext=(-25, 50),color=color_3, textcoords='offset points', arrowprops=dict(arrowstyle='->',color=color_3))





plt.xlabel(x_label)
ylabel = plt.ylabel(y_label, rotation=0)
ylabel.set_verticalalignment('bottom')  # Align at the bottom of the label
ylabel.set_y(ylabel.get_position()[1] - 0.05)
plots_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Plots'
# plt.savefig(os.path.join(plots_folder, 'Zoomed Re map.pgf'), format='pgf', transparent=True)

plt.show()
###############################################################################################################################################
import os
import My_functions as Mf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.constants import g, pi


excel_path = '.\\Analysis\\Excel Files\\Dimensionless Numbers PTV.xlsx'



state_column = 'State'
solution_column = 'Solution'
date_column = 'Experiment Date'

brunt1 = 'Brunt Number'
Re_u1 = 'Upper Re'
Re_l1 = 'Lower Re'
Fr_u1 = 'Upper Fr'
Fr_l1 = 'Lower Fr'
nu_u1 = 'Upper viscosity [m^2/sec]'
nu_l1 = 'Lower viscosity [m^2/sec]'
rho_u1 = 'Upper density [kg/m^3]'
rho_l1 = 'Lower density [kg/m^3]'
U_min1 = 'Minimum velocity [m/s]'
sphere_diameter1 = 'Sphere Diameter [m]'
u_upper1 = 'Upper velocity [m/s]'
rho_s1 = 'Calculated sphere density [kg/m^3]'
interfacey1 = 'Interface width [m]'
traj_time1 = 'Full Trajectory Time [sec]'
u_u1 = 'Upper velocity [m/s]'


df = pd.read_excel(excel_path)
df = df[df[state_column] != 'no - minimum']
# df = df[df[date_column] == '2024/1/24']
# df = df[(df[date_column] == '2024/1/24') | (df[date_column] == '2024/1/23')| (df[date_column] == '2024/1/22')| (df[date_column] == '2024/1/21')]
df = df[df[solution_column] == 'Water - Glycerol']



# df = df[df[rho_s1] < 1160]
# df = df[df[Re_u1] > 100]

# df = df[df[Re_l1] < 70]

# df.dropna(subset=[U_min1], inplace=True)


color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
df['color'] = df[state_column].map(color_mapping)
solution_mapping = {'Water - Salt' : 'P', 'Water - Glycerol' : 'o'}
df['marker type'] = df[solution_column].map(solution_mapping)


df['sphere_volume'] = 4/3*pi*(df[sphere_diameter1]/2)**3
df['F_b'] = g*df['sphere_volume']*df[rho_l1]
df['mg'] = g*df['sphere_volume']*df[rho_s1]
df['momentum'] = df[u_u1]*df[rho_s1]*df['sphere_volume']

x_col = rho_l1
y_col = rho_s1


x_label = r'$\rho_{l}$'
y_label = r'$\rho_{s}$'

x_label = ''
y_label = ''



fig, ax = plt.subplots(figsize=(7, 7))

groups = df.groupby(['color', 'marker type'])
for (color, marker), group in groups:
    # ax.scatter(group[x_col], group[y_col], marker=marker, color=color)
    ax.scatter(group['mg']-group['F_b'], group['momentum'], marker=marker, color=color)
    # ax.scatter(group[x_col]/group[rho_u1], group[y_col]/group[rho_u1], marker=marker, color=color)


legend_elements = [
    Line2D([0], [0], color='w', label='minimum'),
    Line2D([0], [0], color='w', label='no - minimum'),
    Line2D([0], [0], color='w', label='bouncing'),
    Line2D([0], [0], marker=solution_mapping.get('Water - Salt'), color='w', label='Water - Salt', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker=solution_mapping.get('Water - Glycerol'), color='w', label='Water - Glycerol', markerfacecolor='black', markersize=8),
]

legend = plt.legend(handles=legend_elements, loc='best', framealpha = 0)

# Color the legend text
for ind, text in enumerate(legend.get_texts()):
    if ind <=2:
        text.set_color(color_mapping.get(text.get_text()))

# # y=1.03x-0.0295
        
# x1_value = np.linspace(min(df[x_col]), max(df[x_col]), 50)

# line_1_label = r'$\rho^{*}_{s} = 1.03 \rho_{l} - 0.0295$'
# color_1 = 'black'
# y1_value = 1.03*x1_value-30
# y1_value = 0.9*x1_value+112
# y1_value = 1.03*x1_value-0.0295
# ax.plot(x1_value, y1_value, linestyle='--',color=color_1, label=line_1_label)
# ax.annotate(line_1_label, xy=(x1_value[20], y1_value[20]), color=color_1,xytext=(-50, 50), textcoords='offset points', arrowprops=dict(arrowstyle='->', color=color_1))


# # y = 1.0815ρ2 – 0.0815
# line_2_label = r'$\overline{\rho}_{s} = 1.0815 \rho_{l} - 0.0815$'
# color_2 = 'red'
# y2_value = 1.0815*x1_value-80
# y2_value = 1.0815*x1_value-0.0815
# ax.plot(x1_value, y2_value,color=color_2,label=line_2_label)
# ax.annotate(line_2_label, xy=(x1_value[20], y2_value[20]), xytext=(-50, 50),color=color_2, textcoords='offset points', arrowprops=dict(arrowstyle='->',color=color_2))


# #my line
# y3_value = 1.0815*x1_value-85
# color_3='gray'
# line_3_label = 'my line'
# ax.plot(x1_value, y3_value, linestyle='-.',color=color_3, label=line_3_label)
# ax.annotate(line_3_label, xy=(x1_value[20], y3_value[20]), xytext=(-25, 50),color=color_3, textcoords='offset points', arrowprops=dict(arrowstyle='->',color=color_3))





plt.xlabel(x_label)
ylabel = plt.ylabel(y_label, rotation=0)
ylabel.set_verticalalignment('bottom')  # Align at the bottom of the label
ylabel.set_y(ylabel.get_position()[1] - 0.05)
plots_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Plots'
# plt.savefig(os.path.join(plots_folder, 'Zoomed Re map.pgf'), format='pgf', transparent=True)

plt.show()
###############################################################################################################################################
# import My_functions as Mf
# # to_pop=[
# #     'Experiment Date 2024/1/8 Experiment Number 1 Record Number 1',
# #     'Experiment Date 2024/1/8 Experiment Number 1 Record Number 2',
# #     'Experiment Date 2024/1/8 Experiment Number 1 Record Number 16',
# #     'Experiment Date 2024/1/8 Experiment Number 1 Record Number 18',
# #     'Experiment Date 2024/1/8 Experiment Number 1 Record Number 21'
# # ]
# to_pop=[
#     'Experiment Date 2024/1/2 Experiment Number 2 Record Number 1',
#     'Experiment Date 2024/1/2 Experiment Number 2 Record Number 2',
#     'Experiment Date 2024/1/2 Experiment Number 2 Record Number 3',
# ]

# pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'
# The_big_pickle = Mf.load_pickle(pickle_location)
# for el in to_pop:
#     The_big_pickle.pop(el)
# Mf.save_pickle(pickle_location, The_big_pickle)
# print('successfully')
# stop










# import os
# import My_functions as Mf
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import numpy as np
# from scipy.constants import g, pi

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
#     # ax.scatter(np.log(np.log(group[x_col])), group[y_col], marker=marker, color=color)
#     ax.scatter(np.exp(group[x_col]), group[y_col], marker=marker, color=color)
#     # ax.scatter(group[x_col], group[y_col], marker=marker, color=color)

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
# plots_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Plots'
# # plt.savefig(os.path.join(plots_folder, 'Re map.pgf'), format='pgf')
# plt.show()

# atop















import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.constants import g, pi

excel_path = '.\\Analysis\\Excel Files\\Dimensionless Numbers PTV.xlsx'

state_column = 'State'
date_column = 'Experiment Date'

brunt1 = 'Brunt Number'
Re_u1 = 'Upper Re'
Re_l1 = 'Lower Re'
Fr_u1 = 'Upper Fr'
Fr_l1 = 'Lower Fr'
nu_u1 = 'Upper viscosity [m^2/sec]'
nu_l1 = 'Lower viscosity [m^2/sec]'
rho_u1 = 'Upper density [kg/m^3]'
rho_l1 = 'Lower density [kg/m^3]'
U_min1 = 'Minimum velocity [m/s]'
sphere_diameter1 = 'Sphere Diameter [m]'
u_upper1 = 'Upper velocity [m/s]'
rho_s1 = 'Calculated sphere density [kg/m^3]'
interfacey1 = 'Interface width [m]'
traj_time1 = 'Full Trajectory Time [sec]'



df = pd.read_excel(excel_path)
df =df[df[Re_u1]<600]
# df.dropna(subset=[U_min1], inplace=True)


color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
df['color'] = df[state_column].map(color_mapping)

brunt = df[brunt1]
Re_u = df[Re_u1]
Re_l = df[Re_l1]
Fr_u = df[Fr_u1]
Fr_l = df[Fr_l1]
nu_u = df[nu_u1]
nu_l = df[nu_l1]
rho_u = df[rho_u1]
rho_l = df[rho_l1]
sphere_diameter = df[sphere_diameter1]

U_min = df[U_min1]
u_upper = df[u_upper1]
rho_s = df[rho_s1]
interfacey = df[interfacey1]
traj_time = df[traj_time1]



Re = Re_u
Ab1 = pi*sphere_diameter**2/4 #the projected surface area
Cd = 0.4 + 24/Re + 6/(1+np.sqrt(Re))
Fd = Cd*rho_u*u_upper**2*Ab1/2

volume = 4/3*pi*(sphere_diameter/2)**3
# wake_volume = f(Re) 


y_col = Re_l
x_col = Re_u

y_label = r'$\frac {mg \cdot U_u} {\nu_u \cdot \rho_u \cdot a^2}$'
y_label = r'$\frac {\rho_{sphere}} {\rho_{lower}}$'
x_label = r'${Re}_{Lower}$'


# Add the following lines to create a unique marker for each "Sphere Type"
sphere_type_mapping = {'P1': 'o', 'P2': '*', 'P3': 'D', 'P4': 's'}
df['sphere marker'] = df['Sphere Type'].map(sphere_type_mapping)




fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(x_col, y_col, c=df['color'], alpha=0.7)



# for marker, color in zip(df['sphere marker'].unique(), df['color'].unique()):
#     subset = df[df['sphere marker'] == marker]
#     plt.scatter(subset[Re_l1], subset[Re_u1], c=color, marker=marker, label=marker, alpha=0.7)


legend_elements=[Line2D([0], [0], marker='o', color='w', label='minimum', markerfacecolor=color_mapping.get('minimum'), markersize=10)]
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='no - minimum', markerfacecolor=color_mapping.get('no - minimum'), markersize=10))
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='bouncing', markerfacecolor=color_mapping.get('bouncing'), markersize=10))

ax.legend(handles=legend_elements)

plt.xlabel(x_label)
plt.ylabel(y_label, rotation=0, fontsize = 16)


show_index = True
if show_index:
    # Annotation loop
    for i, txt in enumerate(df['Index']):
        ax.annotate(txt, (x_col.iloc[i], y_col.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')


plt.show()
stop
#Check if good manually
# import My_functions as Mf
# import matplotlib.pyplot as plt

# pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'
# the_pickle = Mf.load_pickle(pickle_location)
# all_the_keys = the_pickle.keys()
# for key in all_the_keys:
#     current = the_pickle[key]
#     locationss = current['Sphere Locations']
#     frames,x,y = locationss[:,0], locationss[:,1], locationss[:,2]
#     vel_loc = current['Vertical Location For Velocity [m] list']
#     vel_time = current['Velocity Times [sec] list']
#     vel = current['Velocity [m/s] list']

#     def on_key(event):
#         if event.key == 'r':
#             print(key)
#             plt.close()
#         elif event.key == 'escape':
#             plt.close()
#     y_times_cutted = vel_loc
#     y_velocity_meter_cutted = vel
  
#     fig, ax = plt.subplots(figsize=(7, 7))
    
    
#     plt.title(key)
#     plt.scatter(y_times_cutted, y_velocity_meter_cutted, label='Skipped Data')
#     plt.scatter(y_times_cutted[:5], y_velocity_meter_cutted[:5], color='red', label='Skipped Data start')


#     ax.set_xlabel('time [sec]')
#     ax.set_ylabel('y velocity [m/s]')
#     plt.legend()
#     plt.gca().invert_xaxis()
#     plt.gca().invert_yaxis()
#     fig.canvas.mpl_connect('key_press_event', on_key)
#     plt.show()


import os
import My_functions as Mf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.constants import g, pi

excel_path = r"C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\Dimensionless Numbers PTV.xlsx"

r'''
'Experiment Date',
 'Experiment number',
 'Record number',
 'Sphere Type',
 'Sphere Diameter [m]',
 'Sphere density [kg/m^3]',
 'Upper density [kg/m^3]',
 'Lower density [kg/m^3]',
 'Upper viscosity [m^2/sec]',
 'Lower viscosity [m^2/sec]',
 'Upper velocity [m/s]',
 'Lower velocity [m/s]',
 'Minimum velocity [m/s]',
 'Upper Re',
 'Lower Re',
 'Upper Fr',
 'Lower Fr',
 'Brunt Number',
 'State',
 'Notes',
 'Data Processing Date',
 'Data Processing Hour',
 'Upper Ar',
 'Lower Ar'
 'Calculated sphere density [kg/m^3]']
 mg [N]
 Lower bouyancy [N]
'''

state_column = 'State'
date_column = 'Experiment Date'

brunt1 = 'Brunt Number'
Re_u1 = 'Upper Re'
Re_l1 = 'Lower Re'
Fr_u1 = 'Upper Fr'
Fr_l1 = 'Lower Fr'
nu_u1 = 'Upper viscosity [m^2/sec]'
nu_l1 = 'Lower viscosity [m^2/sec]'
rho_u1 = 'Upper density [kg/m^3]'
rho_l1 = 'Lower density [kg/m^3]'
U_min1 = 'Minimum velocity [m/s]'
sphere_diameter1 = 'Sphere Diameter [m]'
u_upper1 = 'Upper velocity [m/s]'
rho_s1 = 'Calculated sphere density [kg/m^3]'
interfacey1 = 'Interface width [m]'
traj_time1 = 'Full Trajectory Time [sec]'
lower_bouyancy1 = 'Lower bouyancy [N]'
mmg1 = 'mg [N]'
upper_ar1 = 'Upper Ar'
lower_ar1 = 'Lower Ar'



df = pd.read_excel(excel_path)

df.dropna(subset=[U_min1], inplace=True)


color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
df['color'] = df[state_column].map(color_mapping)

brunt = df[brunt1]
Re_u = df[Re_u1]
Re_l = df[Re_l1]
Fr_u = df[Fr_u1]
Fr_l = df[Fr_l1]
nu_u = df[nu_u1]
nu_l = df[nu_l1]
rho_u = df[rho_u1]
rho_l = df[rho_l1]
sphere_diameter = df[sphere_diameter1]

U_min = df[U_min1]
u_upper = df[u_upper1]
rho_s = df[rho_s1]
interfacey = df[interfacey1]
traj_time = df[traj_time1]
mmg = df[mmg1]
lower_bouyancy = df[lower_bouyancy1]
lower_ar = df[lower_ar1]
upper_ar = df[upper_ar1]



volume = 3*pi/4*(sphere_diameter/2)**3
mass = rho_s*volume
Fb = volume*rho_l

y_col = (rho_s-rho_l)/4
y_col = Re_l
y_col = (rho_s/rho_l)
x_col = mass*g-Fb

x_col = Re_u
y_col = Re_l


y_label = r'$\frac {mg \cdot U_u} {\nu_u \cdot \rho_u \cdot a^2}$'
y_label = r'$\frac {\rho_{sphere}} {\rho_{lower}}$'
x_label = r'${Re}_{u}$'
y_label = r'${Re}_{l}$'


v_f = brunt * (interfacey + sphere_diameter)
Re_f = v_f * sphere_diameter/nu_l 
C_d = 24/Re_f + 6/(1+np.sqrt(Re_f)) + 0.4
A = sphere_diameter**2 *pi/4
F_new = C_d*(v_f**2)*A*rho_l 
Force_ratio = F_new/mmg 
Density_ratio = rho_s/rho_l - 1

x_col = Force_ratio
y_col = Density_ratio



print(len(x_col), len(y_col))
print(list(x_col)[2], list(y_col)[2],list(x_col)[5], list(y_col)[5])
# , x_col[3], y_col[3])



fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(x_col, y_col, c=df['color'], alpha=0.7)

legend_elements=[Line2D([0], [0], marker='o', color='w', label='minimum', markerfacecolor=color_mapping.get('minimum'), markersize=10)]
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='no - minimum', markerfacecolor=color_mapping.get('no - minimum'), markersize=10))
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='bouncing', markerfacecolor=color_mapping.get('bouncing'), markersize=10))

ax.legend(handles=legend_elements)

plt.xlabel(x_label)
# plt.ylabel(y_label, rotation=0, fontsize = 16)
ylabel = plt.ylabel(y_label, rotation=0)
ylabel.set_verticalalignment('bottom')  # Align at the bottom of the label
ylabel.set_y(ylabel.get_position()[1] - 0.05)
# plt.savefig(os.path.join(Mf.select_directory(), 'Re_map.pgf'), format='pgf')
plt.show()

stop





import My_functions as Mf
import os

import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [5,6,5,8]

plt.scatter(x, y, marker='x')
plt.savefig(os.path.join(Mf.select_directory(), 'norm.pgf'), format='pgf')



stop
import My_functions as Mf
import os
import pandas as pd

# Read Excel file
excel_file_path = Mf.select_file()
df = pd.read_excel(excel_file_path)

# Convert DataFrame to LaTeX table
latex_table = df.to_latex(index=False)


# Save LaTeX table to a file
latex_file_path = os.path.join(Mf.select_directory(),'output_table.tex')
with open(latex_file_path, 'w') as f:
    f.write(latex_table)

stop
import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import os


k = 4

# Define the numerator and denominator coefficients of the transfer function
numerator = [0.25]
denominator1 = [1, 0.25*10*np.sqrt(k), 0.25*k]
denominator2 = [1, 0.25*1*np.sqrt(k), 0.25*k]
denominator3 = [1, 0.25*4*np.sqrt(k), 0.25*k]


# Create a transfer function using tf()
sys_tf1 = ctrl.TransferFunction(numerator, denominator1)
sys_tf2 = ctrl.TransferFunction(numerator, denominator2)
sys_tf3 = ctrl.TransferFunction(numerator, denominator3)



# Step response plot
time = np.linspace(0, 25, 1000)
response1 = ctrl.step_response(sys_tf1, time)[1]
response2 = ctrl.step_response(sys_tf2, time)[1]
response3 = ctrl.step_response(sys_tf3, time)[1]

plt.plot(time, response1, label = r'Overdamped $c>4\sqrt{k}$')
plt.plot(time, response2, label = r'Underdamped $c<4\sqrt{k}$')
plt.plot(time, response3, label = r'Critical Damped $c=4\sqrt{k}$')
plt.legend()
plt.title(f'Step Response k={k}')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
locy = r'C:\Users\Morten\Desktop\DEL ME'
plt.savefig(os.path.join(locy, 'output.svg'), format='svg', bbox_inches='tight')
plt.show()
plt.close()

plt.figure()
# Extract poles and zeros
poles1 = ctrl.pole(sys_tf1)
zeros1 = ctrl.zero(sys_tf1)
poles2 = ctrl.pole(sys_tf2)
zeros2 = ctrl.zero(sys_tf2)
poles3 = ctrl.pole(sys_tf3)
zeros3 = ctrl.zero(sys_tf3)



# Plot poles and zeros
plt.scatter(np.real(poles1), np.imag(poles1), marker='x', label=r'Overdamped $c>4\sqrt{k}$', color='red')
plt.scatter(np.real(zeros1), np.imag(zeros1), marker='o', label='', color='red')
plt.scatter(np.real(poles2), np.imag(poles2), marker='x', label=r'Underdamped $c<4\sqrt{k}$', color='blue')
plt.scatter(np.real(zeros2), np.imag(zeros2), marker='o', label='', color='blue')
plt.scatter(np.real(poles3), np.imag(poles3), marker='x', label=r'Critical Damped $c=4\sqrt{k}$', color='green')
plt.scatter(np.real(zeros3), np.imag(zeros3), marker='o', label='', color='green')

plt.legend()
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
plt.title('Pole-Zero Map')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

# ctrl.pzmap()
# ctrl.pzmap_mort([sys_tf1,sys_tf2,sys_tf3], labels = [r'Overdamped $c>4\sqrt{k}$', r'Underdamped $c<4\sqrt{k}$', r'Critical Damped $c=4\sqrt{k}$'])



stop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.constants import g, pi

excel_path = r"C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\Dimensionless Numbers PTV.xlsx"

r'''
'Experiment Date',
 'Experiment number',
 'Record number',
 'Sphere Type',
 'Sphere Diameter [m]',
 'Sphere density [kg/m^3]',
 'Upper density [kg/m^3]',
 'Lower density [kg/m^3]',
 'Upper viscosity [m^2/sec]',
 'Lower viscosity [m^2/sec]',
 'Upper velocity [m/s]',
 'Lower velocity [m/s]',
 'Minimum velocity [m/s]',
 'Upper Re',
 'Lower Re',
 'Upper Fr',
 'Lower Fr',
 'Brunt Number',
 'State',
 'Notes',
 'Data Processing Date',
 'Data Processing Hour',
 'Upper Ar',
 'Lower Ar'
 'Calculated sphere density [kg/m^3]']
 mg [N]
 Lower bouyancy [N]
'''

state_column = 'State'
date_column = 'Experiment Date'

brunt1 = 'Brunt Number'
Re_u1 = 'Upper Re'
Re_l1 = 'Lower Re'
Fr_u1 = 'Upper Fr'
Fr_l1 = 'Lower Fr'
nu_u1 = 'Upper viscosity [m^2/sec]'
nu_l1 = 'Lower viscosity [m^2/sec]'
rho_u1 = 'Upper density [kg/m^3]'
rho_l1 = 'Lower density [kg/m^3]'
U_min1 = 'Minimum velocity [m/s]'
sphere_diameter1 = 'Sphere Diameter [m]'
u_upper1 = 'Upper velocity [m/s]'
rho_s1 = 'Calculated sphere density [kg/m^3]'
interfacey1 = 'Interface width [m]'
traj_time1 = 'Full Trajectory Time [sec]'
lower_bouyancy1 = 'Lower bouyancy [N]'
mmg1 = 'mg [N]'
upper_ar1 = 'Upper Ar'
lower_ar1 = 'Lower Ar'



df = pd.read_excel(excel_path)

df.dropna(subset=[U_min1], inplace=True)


color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
df['color'] = df[state_column].map(color_mapping)

brunt = df[brunt1]
Re_u = df[Re_u1]
Re_l = df[Re_l1]
Fr_u = df[Fr_u1]
Fr_l = df[Fr_l1]
nu_u = df[nu_u1]
nu_l = df[nu_l1]
rho_u = df[rho_u1]
rho_l = df[rho_l1]
sphere_diameter = df[sphere_diameter1]

U_min = df[U_min1]
u_upper = df[u_upper1]
rho_s = df[rho_s1]
interfacey = df[interfacey1]
traj_time = df[traj_time1]
mmg = df[mmg1]
lower_bouyancy = df[lower_bouyancy1]
lower_ar = df[lower_ar1]
upper_ar = df[upper_ar1]


Re = Re_u
Ab1 = pi*sphere_diameter**2/4 #the projected surface area
Cd = 0.4 + 24/Re + 6/(1+np.sqrt(Re))
Fd = Cd*rho_u*u_upper**2*Ab1/2

volume = 4/3*pi*(sphere_diameter/2)**3
# wake_volume = f(Re) 

x_col = Fd/(g*volume*(rho_s-rho_l))
y_col = rho_s/rho_l



volume = 3*pi/4*(sphere_diameter/2)**3
mass = rho_s*volume
Fb = volume*rho_l

y_col = (rho_s-rho_l)/4
y_col = (rho_s/rho_l)

y_col = Re_u
x_col = (mass*g-Fb)/0.0005

y_col = U_min/u_upper
x_col = Re_l**(1/2)




y_label = r'$\frac {mg \cdot U_u} {\nu_u \cdot \rho_u \cdot a^2}$'
y_label = r'$\frac {\rho_{sphere}} {\rho_{lower}}$'
x_label = r'${Re}_{Lower}$'


# Add the following lines to create a unique marker for each "Sphere Type"
sphere_type_mapping = {'P1': 'o', 'P2': '*', 'P3': 'D', 'P4': 's'}
df['sphere marker'] = df['Sphere Type'].map(sphere_type_mapping)




fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(x_col, y_col, c=df['color'], alpha=0.7)



# for marker, color in zip(df['sphere marker'].unique(), df['color'].unique()):
#     subset = df[df['sphere marker'] == marker]
#     plt.scatter(subset[Re_l1], subset[Re_u1], c=color, marker=marker, label=marker, alpha=0.7)


legend_elements=[Line2D([0], [0], marker='o', color='w', label='minimum', markerfacecolor=color_mapping.get('minimum'), markersize=10)]
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='no - minimum', markerfacecolor=color_mapping.get('no - minimum'), markersize=10))
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='bouncing', markerfacecolor=color_mapping.get('bouncing'), markersize=10))

ax.legend(handles=legend_elements)

plt.xlabel(x_label)
plt.ylabel(y_label, rotation=0, fontsize = 16)


show_index = False
if show_index:
    # Annotation loop
    for i, txt in enumerate(df['Index']):
        ax.annotate(txt, (x_col.iloc[i], y_col.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')


plt.show()





stop

















from datetime import datetime
#used in lab
import pyautogui
import time
import keyboard
import win32api, win32con

def click(x, y, pause = 0.1):
    '''
    recive x,y coordinates and clicks at them
    '''
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(pause) #This pauses the script
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
    time.sleep(pause) #This pauses the script




# #show coordinaes and rgb all the time
# print(pyautogui.displayMousePosition())



# # when click show coordinates
# while(1):
#     if win32api.GetKeyState(win32con.VK_LBUTTON) < 0:
#         current_x, current_y = pyautogui.position()
#         print(f'Mouse position: ({current_x}, {current_y})')
#         time.sleep(5) #5 sec sleep

# דאםפ



def veri(x = 103, y = 39):
    click(x, y)
def ini(x = 480, y = 981):
    click(x, y)
def ball_prog(x =  465, y = 1056):
    click(x, y)
def release_ball(x = 910, y = 503):
    click(x, y)
def record(x = 494, y = 982):
    click(x, y)
 
buttum = 'ctrl+R'
buttum2 = 'ctrl+r'


print('run')
while True:
    if keyboard.is_pressed(buttum) or keyboard.is_pressed(buttum2):
        veri()
        ini()
        ball_prog()
        time.sleep(0.5)
        release_ball()
        time.sleep(0.2)
        record()
        # click(1569, 105)
   

        print('Realse ball at:', datetime.now().strftime('%Y-%m-%d   %H:%M:%S'))


    if keyboard.is_pressed('ctrl+t'):
        print('úser block')
        break
# (910, 503) relaese ball
# (494, 982)qinaaaaaaaaaaraaaaaaarrrr







import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 5, 8, 3, 6]

fig, ax = plt.subplots()
line, = ax.plot(x, y, 'o-', label='Data')

# Function to update text when hovering over a point
def update_text(event):
    if event.inaxes == ax:
        x_data, y_data = event.xdata, event.ydata
        index = min(range(len(x)), key=lambda i: (x[i]-x_data)**2 + (y[i]-y_data)**2)

        text.set_text(f'Point: ({x[index]:.2f}, {y[index]:.2f})')
        text.set_position((x[index], y[index]))
        fig.canvas.draw()

# Connect the update_text function to the motion_notify_event
fig.canvas.mpl_connect('motion_notify_event', update_text)

# Create a text annotation
text = ax.text(0, 0, '', color='red')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Hover over points to display data')

plt.legend()
plt.grid(True)
plt.show()




from sympy import symbols, fourier_series, pi

# Define the variable and the function
x = symbols('x')
f = x  # This is the function for which we want to compute the Fourier series

# Define the period of the function
p = 10*pi  # You can change this to the period of your function

# Compute the Fourier series
fourier = fourier_series(f, (x, -p/2, p/2))

print(fourier)
n=20


import matplotlib.pyplot as plt
import numpy as np

# Define the x values for the plot
x_values = np.linspace(-10, 10, 400)

# Compute the y values for y = x and y = F
y_x = x_values
y_F = [fourier.truncate(n=n).subs(x, val).evalf() for val in x_values]
# y_F = fourier.subs(x, x_values).truncate(n=n).evalf()


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_x, label='y = x')
plt.plot(x_values, y_F, label='y = F')
plt.legend()
plt.grid(True)
plt.show()















import My_functions as Mf
file_list = Mf.files_in_folder(r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Python Thesis\Analysis\output\Multy', full_path=True)


import napari
import numpy as np
viewer = napari.Viewer(title = 'cat', axis_labels=['y', 'x'])
# image_paths = find_pictures(directory, True)
# all_images = [read_image(path) for path in image_paths]
# Images = np.stack(all_images)
# viewer.add_image(Images, name='Images')



def read_data(filename):
    a = np.loadtxt(filename)
    x, y, u, v, flags, mask = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    return x, y, u, v, flags, mask


veccc = []
for filename in file_list:
    x, y, u, v, flags, mask = read_data(filename)
    vectors = np.dstack((x, y, u, v))
    vectors = vectors.reshape((len(x), 2, 2))
    veccc.append(vectors)



vecccy = np.stack(veccc)
viewer.add_vectors(vecccy, name='Vectors')

napari.run()



#!/usr/bin/env python3

"""Glycerol_calculator.py"""

## This python code is based on the MatLab code orginaly provided by Chris Westbrook
## http://www.met.reading.ac.uk/~sws04cdw/viscosity_calc.html

__author__  = "Matthew Partridge"
__license__ = "GPL"
__version__ = "1.0"
__credits__ = "Chris Westbrook"


#Required packages ----------------

import numpy
import math


#Variables ----------------

T = 20 				#temperature (degrees Celcius)
waterVol = 3 		#volume of water required (ml)
glycerolVol = 0.75	#volume of Glycerol used (ml)


#Densities ----------------

glycerolDen = (1273.3-0.6121*T)/1000 			#Density of Glycerol (g/cm3)
waterDen = (1-math.pow(((abs(T-4))/622),1.7)) 	#Density of water (g/cm3)


#Fraction cacluator ----------------

glycerolMass=glycerolDen*glycerolVol
waterMass=waterDen*waterVol
totalMass=glycerolMass+waterMass
mass_fraction=glycerolMass/totalMass
vol_fraction= glycerolVol/(glycerolVol+waterVol)

print ("Mass fraction of mixture =", round(mass_fraction,5))
print ("Volume fraction of mixture =", round(vol_fraction,5))


#Density calculator ----------------

##Andreas Volk polynomial method
contraction_av = 1-math.pow(3.520E-8*((mass_fraction*100)),3)+math.pow(1.027E-6*((mass_fraction*100)),2)+2.5E-4*(mass_fraction*100)-1.691E-4
contraction = 1+contraction_av/100

## Distorted sine approximation method
#contraction_pc = 1.1*math.pow(math.sin(numpy.radians(math.pow(mass_fraction,1.3)*180)),0.85)
#contraction = 1 + contraction_pc/100

density_mix=(glycerolDen*vol_fraction+waterDen*(1-vol_fraction))*contraction

print ("Density of mixture =", round(density_mix,5),"g/cm3")


#Viscosity calcualtor ----------------

glycerolVisc=0.001*12100*numpy.exp((-1233+T)*T/(9900+70*T))
waterVisc=0.001*1.790*numpy.exp((-1230-T)*T/(36100+360*T))

a=0.705-0.0017*T
b=(4.9+0.036*T)*numpy.power(a,2.5)
alpha=1-mass_fraction+(a*b*mass_fraction*(1-mass_fraction))/(a*mass_fraction+b*(1-mass_fraction))
A=numpy.log(waterVisc/glycerolVisc)

viscosity_mix=glycerolVisc*numpy.exp(A*alpha)

print ("Viscosity of mxiture =",round(viscosity_mix,5), "Ns/m2")

































# import My_functions as Mf

# Mf.plot_states_with_Perceptron(use_perceptron=False)



# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# excel_path = r"C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Python Thesis\Analysis\Dimensionless Numbers PTV.xlsx"
# the_x_column = 'Upper Re'
# the_y_column = 'Lower Re'
# state_column = 'State'
# date_column = 'Date'  # Replace 'Date' with the actual column name containing dates
# date_column = 'Experiment Date'


# df = pd.read_excel(excel_path)
# color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing': 'violet'}

# # Convert the entire 'Date' column to datetime
# df[date_column] = pd.to_datetime(df[date_column])

# # Original code without perceptron
# df['color'] = df[state_column].map(color_mapping)

# unique_dates = df['Experiment Date'].dt.date.unique()
# # Create a dictionary for mapping states to markers
# date_marker_mapping = {date: marker for date, marker in zip(unique_dates, ['o', 's', 'D', 'v', '^', '<', '>', 'P', 'X', '*'])}


# # Group by date and state, then plot with different markers and colors
# fig, ax = plt.subplots(figsize = (10,10))
# for (date, state), group in df.groupby([date_column, state_column]):
#     date_str = str(date.date())  # Convert timestamp to string
#     marker = date_marker_mapping.get(date.date(), 'o')  # Provide a default marker if not found in the mapping
#     color = color_mapping[state]
#     ax.scatter(group[the_x_column], group[the_y_column], c=color, alpha=0.7, marker=marker, label=f"{date_str}")



# # legend_labels = [date for date in df[date_column].dt.date.unique()]
# # legend_handles = [Line2D([0], [0], marker=date_marker_mapping.get(date), label=str(date)) for date in unique_dates]
# legend_handles = [Line2D([0], [0], marker=date_marker_mapping.get(date, 'o'), color='w', markerfacecolor='gray', markersize=10, label=str(date)) for date in unique_dates]
# legend_labels2 = ['minimum', 'no - minimum', 'bouncing']
# legend_handles2 = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[label], markersize=10, label=label) for label in legend_labels2]

# legend_handles = legend_handles+legend_handles2
# ax.legend(handles=legend_handles, title=f"{date_column}")

# # plt.legend(handles=legend_handles)


# plt.xlabel(the_x_column)
# plt.ylabel(the_y_column)
# plt.show()

