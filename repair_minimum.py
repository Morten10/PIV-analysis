import My_functions as Mf
import matplotlib.pyplot as plt
import numpy as np

pickle_location = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Shared Folder\Excel Files\The Big Pickle.pkl'
The_big_pickle = Mf.load_pickle(pickle_location)
keys=The_big_pickle.keys()
keys = [key for key in keys if key != 'Object To Follow'] #remove object to follow

the_lst = []
has_been_changed_to_minimum = ['Experiment Date 2023/12/13 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 4', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 5', 'Experiment Date 2023/12/17 Experiment Number 1 Record Number 8', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 1', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 2', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 4', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 5', 'Experiment Date 2023/12/18 Experiment Number 1 Record Number 6', 'Experiment Date 2023/12/21 Experiment Number 1 Record Number 1', 'Experiment Date 2023/12/21 Experiment Number 1 Record Number 2', 'Experiment Date 2023/12/24 Experiment Number 1 Record Number 7', 'Experiment Date 2023/12/24 Experiment Number 1 Record Number 11', 'Experiment Date 2023/12/24 Experiment Number 1 Record Number 13', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 1', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 3', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 5', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 8', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 9', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 10', 'Experiment Date 2023/12/26 Experiment Number 1 Record Number 20', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 2', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 8', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 9', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 10', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 17', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 18', 'Experiment Date 2023/12/28 Experiment Number 1 Record Number 20', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 2', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 3', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 5', 'Experiment Date 2024/1/2 Experiment Number 1 Record Number 6', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 1', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 3', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 5', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 7', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 8', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 9', 'Experiment Date 2024/1/22 Experiment Number 1 Record Number 10', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 1', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 2', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 3', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 5', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 6', 'Experiment Date 2023/12/18 Experiment Number 2 Record Number 7', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 1', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 2', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 3', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 4', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 5', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 6', 'Experiment Date 2023/12/21 Experiment Number 2 Record Number 7', 'Experiment Date 2024/1/22 Experiment Number 2 Record Number 14', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 14', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 15', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 16', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 17', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 18', 'Experiment Date 2024/1/23 Experiment Number 3 Record Number 19']



for key in keys:
    current = The_big_pickle[key]

   
    if current['State'] == 'no - minimum':
    # if current['Index'] == 156:
    # if '2023/12/26' == current['Experiment Date'] and 1 == current['Experiment number'] and  4 == current['Record number']:
        print(current['Experiment Date'] , current['Experiment number'] ,current['Record number'],current['Index'])

        x_values = np.array(current['Vertical Location For Velocity [m] list'])
        time = current['Velocity Times [sec] list']
        y_values = np.array(current['Velocity [m/s] list'])
        fps = current['Frames Per Second']
        dt = 1/fps
        pos_vel = -y_values



        fig, ax = plt.subplots()
        def on_key(event):
            if event.key == 'a':
                print(f'{state} approved {key} has been changed')
                current['State'] = state
                the_lst.append(f'{key} {state}')
                plt.close()
            elif event.key == 'escape':
                plt.close()

        error = 0.001/2
        errors = np.ones(len(y_values))*error
        ax.set_title(f"{current['Experiment Date']} - {current['Experiment number']} - {current['Record number']}")
        ax.scatter(x_values, pos_vel, color='red')
        ax.errorbar(x_values, pos_vel, yerr=errors, fmt='', label='velocity with Error Bars')
        ax.axhline(pos_vel[-1]-error, color='g', linestyle='--')
        ax.axhline(min(pos_vel)+error, color='r', linestyle='--')
        ax.legend(framealpha = 0)
        plt.gca().invert_xaxis()

        if pos_vel[-1]-error - min(pos_vel)+error <= error:
            print (f'Uncertainty Zone for {key}')
        
        if min(pos_vel)<0:
            state = 'bouncing'
        elif pos_vel[-1]-error > min(pos_vel)+error:
            state = 'minimum'
        else:
            state = 'no - minimum'
        print(state)

    
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        print(f"RE_U={current['Upper Re']} RE_L = {current['Lower Re']}")
        
        # print(pos_vel[-1]-min(pos_vel))
        # print(f'was {current["State"]} but now:')

        # if min(pos_vel) == pos_vel[-1]:
        #     print('no-minimum')
        # elif min(pos_vel)<0:
        #     print('bouncing')
        # elif pos_vel[-1]-min(pos_vel)>2*error:
        #     print('minimum')
        # elif 0<pos_vel[-1]-min(pos_vel)<=2*error:
        #     print('transient')
        # else:
        #     print('no-minimum')

        

        # lolol = Mf.derivative_of_2d_location(np.column_stack((x_values, y_values)))
        # xxxx,yyyy = lolol[:,0],lolol[:,1]
        # fig, ax = plt.subplots()
        # # error = 0.001/2
        # # errors = np.ones(len(y_values))*error
        # ax.scatter(xxxx, yyyy, color='red')
        # plt.show()


# Mf.save_pickle(pickle_location, The_big_pickle)
# print('successfully')
        
# def write_list_to_txt(lst, file_name):
#     with open(file_name, 'w') as file:
#         for item in lst:
#             file.write(str(item) + '\n')



# # File name to write to
# file_name = r"C:\Users\Morten\Desktop\nice.txt"

# # Writing the list to the text file
# write_list_to_txt(the_lst, file_name)

# print(f"List has been written to {file_name}")