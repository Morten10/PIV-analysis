""" Plot Chen/Aviv data using interactive plotly """
import My_functions as Mf
import pandas as pd
import numpy as np
from scipy.constants import pi
pd.options.plotting.backend = "plotly"



import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# locations
plots_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Plots'
excel_path = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\Dimensionless Numbers PTV.xlsx'
pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'


# Reading the data:
df = pd.read_excel(excel_path)
the_pickle = Mf.load_pickle(pickle_location)
the_keys = the_pickle.keys()

# manual selection
to_skip = [
    'Object To Follow',
    'Experiment Date 2023/12/20 Experiment Number 1 Record Number 1',
    'Experiment Date 2023/12/18 Experiment Number 2 Record Number 5',
    'Experiment Date 2024/1/8 Experiment Number 2 Record Number 16',
    'Experiment Date 2024/1/23 Experiment Number 2 Record Number 12',
    'Experiment Date 2024/1/23 Experiment Number 2 Record Number 13',
    'Experiment Date 2024/1/2 Experiment Number 2 Record Number 2',
    'Experiment Date 2024/1/2 Experiment Number 2 Record Number 3',
    'Experiment Date 2024/1/22 Experiment Number 2 Record Number 13',
    'Experiment Date 2024/1/22 Experiment Number 2 Record Number 15',
    'Experiment Date 2024/1/22 Experiment Number 2 Record Number 18',
    ]
to_skip = [
    'Object To Follow',
    ]

# rename stuff
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



# select stuf
df = df[df[Re_u1] < 600]
# df = df[df['Index'] != 160]


color_mapping = {'minimum': 'red', 'no - minimum': 'blue', 'bouncing':  'green'}
df['color'] = df[state_column].map(color_mapping)
solution_mapping = {'Water - Salt' : 'o', 'Water - Glycerol' : 's'}
df['marker type'] = df[solution_column].map(solution_mapping)

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


volume = 3*pi/4*(sphere_diameter/2)**3
mass = rho_s*volume
Fb = volume*rho_l
fig = df.plot.scatter(x=Re_u1, y=Re_l1, color=state_column)
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
fig.show()






# # Velocity
 
# One way is to create a dedicated dataframe: 
tmp = []

for el in the_keys:
    if el not in to_skip:
        current = the_pickle[el]

        locc = np.array(current['Vertical Location For Velocity [m] list'])
        velocii = np.array(current['Velocity [m/s] list'])

        # Normalize velocity
        normalized_velocii = -velocii / max(-velocii)
        tmp.append(pd.DataFrame({'Vertical Location [m]': locc, 'Normalized Velocity': normalized_velocii, 'Source':el}))

tmp = pd.concat(tmp)

# and plot it:
fig = px.line(tmp, x='Vertical Location [m]', y='Normalized Velocity',color='Source')

# Customize the axes labels
fig.update_xaxes(title_text='Vertical Location [m]', showline=True)
fig.update_yaxes(title_text=r'$\frac {u} {u_{max}}$', showline=True)

# Invert x-axis
fig.update_xaxes(autorange="reversed")
fig.show()


# Another plot
to_do = ['Experiment Date 2024/1/23 Experiment Number 1']

# fig, ax = plt.subplots(figsize=(7, 7))
tmp = []
for el in the_keys:
    if to_do[0] in el:
        if not el in to_skip:
       
            current = the_pickle[el]

            locc = current['Vertical Location For Velocity [m] list']
            # timee = current['Velocity Times [sec] list']
            velocii = np.array(current['Velocity [m/s] list'])
            date = current['Experiment Date']
            # ax.plot(locc, -velocii/max(-velocii), label = date + " " + str(int(el[-2:])))
            # ax.plot(locc, -velocii/max(-velocii))
            tmp.append(pd.DataFrame(dict(locc = locc, vel = -velocii/max(-velocii), source = el)))

tmp = pd.concat(tmp)

#  plotting
fig = tmp.plot.line(x='locc', y='vel', color='source', labels={'locc': 'Vertical Location [m]', 'vel': r'$\frac {u} {u_{max}}$'}, title='Normalized Velocity')
fig.update_layout(
    xaxis = dict(autorange="reversed")
)
fig.show()
import os
plots_folder = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Plots'

fig.write_html(os.path.join(plots_folder, '2024_1_23.html'))









to_do = ['Experiment Date 2024/1/24 Experiment Number 1']

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

for el in the_keys:
    holo = el[:45]
    if holo in to_do and el not in to_skip:
        current = the_pickle[el]
        locc = current['Vertical Location For Velocity [m] list']
        velocii = np.array(current['Velocity [m/s] list'])
        date = current['Experiment Date']
        Interface_thickness = current['Interface width [m]']
        
        fig.add_trace( go.Scatter(x=locc, y=-velocii/max(-velocii), name = el), secondary_y = False )

fig.update_layout(xaxis=dict(autorange="reversed"))
# fig.show()

# Load density profile data
y_loccccc, densityccccc = Mf.load_density_profile_from_excel(date=date)
y_loccccc2 = np.array(y_loccccc)
densityccccc2 = np.array(densityccccc)

# Calculate interface height
Interface_height = Mf.find_average_density_location(y_loccccc2, densityccccc2)
Interface_thickness = current['Interface width [m]']
interface_lower_location = Interface_height - Interface_thickness/2
interface_width = Interface_thickness

fig.add_vrect(x0=interface_lower_location, x1=interface_lower_location+interface_width, line_width=0, fillcolor="yellow", opacity=0.3) 
fig.add_trace( go.Scatter(x=y_loccccc2, y=densityccccc2, name='Density Profile', fillcolor="black"), secondary_y=True)
fig.update_yaxes(title_text=r'Density $\left[\frac{kg}{m^3}\right]$', secondary_y=True)
fig.show()


