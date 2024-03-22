import My_functions as Mf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

# Your code for loading data and calculations...

current_path = Path.cwd()
pickle_location = current_path / "Excel Files" / "The Big Pickle.pkl"
the_pickle = Mf.load_pickle(pickle_location)

current = the_pickle['Experiment Date 2023/12/13 Experiment Number 1 Record Number 1']
# print(current.keys())
times = current['Velocity Times [sec] list']
v_b = np.array(current['Velocity [m/s] list'])
dt = 1/current['Frames Per Second']
a = current['Sphere Diameter [m]']
z_0 = current['Vertical Location For Velocity [m] list'][0]


the_keys = the_pickle.keys()
# options = [key for key in the_keys if key != 'Object To Follow'] #remove object to follow
options = [key + f'-   {the_pickle[key]["State"]}' for key in the_keys if key != 'Object To Follow'] #remove object to follow




import tkinter as tk
selected_experiment = []
def show_selected_option():
    selected_index = listbox.curselection()
    if selected_index:
        selected_option = options[selected_index[0]]
        selected_experiment.append(selected_option.split('-', 1)[0])
        print("You selected:", selected_option)
        root.quit()  # Close the Tkinter window
    else:
        print("Please select an option.")


root = tk.Tk()
root.title("Select an Experiment")


listbox = tk.Listbox(root, width=100, height=28) 
for option in options:
    listbox.insert(tk.END, option)
listbox.pack(padx=20, pady=20)

# Create a button to get the selected option
select_button = tk.Button(root, text="Select", command=show_selected_option)
select_button.pack()

# Start the GUI event loop
root.geometry("600x600")  # Set the width and height of the window

root.mainloop()

root.quit()  # Close the Tkinter window









current = the_pickle[selected_experiment[0]]
# print(current.keys())
times = current['Velocity Times [sec] list']
v_b = np.array(current['Velocity [m/s] list'])
dt = 1/current['Frames Per Second']
a = current['Sphere Diameter [m]']
z_0 = current['Vertical Location For Velocity [m] list'][0]












r1 = np.linspace(-0.15, 0.15, 100)
z1 = np.linspace(0, 0.35, 100)
r,z = np.meshgrid(r1, z1)


fig, ax = plt.subplots(figsize=(8,8))
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.1)  # Adjust the bottom to make room for the slider

t_0 = 0
z_bar = z-(z_0 + np.trapz(v_b[:t_0], times[:t_0]))
vb = v_b[t_0] #The velocity at a specific time. if we dont do that we have 3 dimentions, Z R T
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