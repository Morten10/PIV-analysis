import napari, cv2, os, scipy, shutil, pickle, scipy
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from tkinter import filedialog, Tk, filedialog
from typing import Optional


########
# General functions
########
def create_directory(location : str):
    r'''
    creating folder if not exists. otherwise does nothing.

    Parameters
    ----------
    location : str
        Input str path of the folder.

    Returns
    -------
    None
    '''
    if not os.path.isdir(location):
        os.makedirs(location)
        print('the folder ' + location + ' has been created')

def select_file():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename()

def eliminate_folder(folder_path):
    from send2trash import send2trash
    try:
        send2trash(folder_path)
        print(f"Folder '{folder_path}' has been moved to the recycle bin.")
    except Exception as e:
        print(f"An error occurred while moving '{folder_path}' to the recycle bin: {str(e)}")
    
    # try:
    #     shutil.rmtree(folder_path)
    #     print(f"Folder '{folder_path}' has been deleted successfully.")
    # except Exception as e:
    #     print(f"An error occurred while deleting '{folder_path}': {str(e)}")

def read_image(path : str, gray : bool = True):
    if gray:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(path)

def save_image(path : str, pic : list):
    cv2.imwrite(path, pic)

def save_pickle(path : str, files):
    r'''
    saves files in the path using pickle module


    Parameters
    ----------
    path : str
        Input str path of the folder.
    files
        the object to save, must of the time its a list.
    '''
    with open(path, 'wb') as file:
        pickle.dump(files, file)

def load_pickle(path : str):
    r'''
    saves files in the path using pickle module


    Parameters
    ----------
    path : str
        Input str of the folder.
    files
        the object to save, must of the time its a list.
    
        
    Returns
    -------
    files
        the pickle file
    '''

    with open(path, 'rb') as file:
        files = pickle.load(file)
    return files

def t_student_value(confidence : float = 0.95, N : int = 3) -> float:
    r'''
    t student distirbution

    Parameters
    ----------
    confidence : float = 0.95
        the % confidence of uncertainty. a number between 0 and 1 - 1 = 100% certainty, 0 = 0 % certainty
    N : int = 3
        Number of elements or number of measerments
        

    Returns
    -------
    float
        the value of t-student distrubution, also known as coverage factor

    
    Examples
    --------
    use the t_student_value to find the coverage factor of N measurements with "confidence" precentage of confidence
    >>> t_student_value(confidence = 0.99, N : int = 61)
    2.66
    '''
    alpha = 1 - confidence
    df = N - 1  # degrees of freedom
    return scipy.stats.t.ppf(1 - alpha/2, df)
    

def draw_filled_triangle(image_path, vertices):
    image = read_image(image_path, False)
    color = (0, 0, 0)
    pts = np.array(vertices, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], color)
    path, ext = os.path.splitext(image_path)
    output_path = path + "_triangle" + ext
    cv2.imwrite(output_path, image)

def files_in_folder(path : str, extension : str | list = 'txt', full_path : bool = False):
    r'''
    uses os mudule to find all files ending with 'extension'


    Parameters
    ----------
    path : str
        Input str of the folder.
    extension : str | list = 'txt'
        an str of the file extension.
    full_path : bool = False
        if true adds the path to the file name so each cell contains the full path.
        

    Returns
    -------
    files : list
        each cell correspond to a file

    Examples
    --------
    use the files_in_folder to find all the files ending with 'extension'
    >>> files_in_folder(path)
    ['a.txt', 'b.txt', 'c.txt']
    >>> files_in_folder(path, full_path = True)
    ['C:\a.txt', 'C:\b.txt', 'C:\c.txt']
    >>> files_in_folder(path, extension = ['txt','png'])
    ['a.txt', 'b.png', 'c.png']
    '''

    if type(extension) is not list:
        extension = [extension]
    files = []
    for file in os.listdir(path):
        if any(file.endswith('.' + ext) for ext in extension):
            if full_path:
                files.append(os.path.join(path, file))
            else:
                files.append(file)

    # old but working very good
    # if type(extension) is list:
    #     files=[]
    #     for el in extension:
    #         if full_path:
    #             files = files + [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + el)]
                
    #         else:
    #             files = files + [f for f in os.listdir(path) if f.endswith('.' + el)]
    # else:
    #     if full_path:
    #         files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + extension)]
    #     else:
    #         files = [f for f in os.listdir(path) if f.endswith('.' + extension)]
    return files

def find_pictures(directory : str, full_path : bool = True, extension : list = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'ico', 'jfif', 'webp']):
    r'''
    return all the pictures in the directory, 


    Parameters
    ----------
    directory : str
        Input str of the folder.
    extension : list = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'ico', 'jfif', 'webp']
        a list of the picture extensions.
    full_path : bool = False
        if true adds the path to the file name so each cell contains the full path.
        

    Returns
    -------
    list each cell correspond to picture

    Examples
    --------
    use the find_pictures to find all the files ending with 'extension'
    >>> find_pictures(path)
    ['a.tiff', 'b.jpg', 'c.png']
    >>> find_pictures(path, full_path = True)
    ['C:\a.png', 'C:\b.png', 'C:\c.png']
    '''
    return files_in_folder(directory, extension, full_path)

def find_pictures_subfolder(directory, extension=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'ico', 'jfif', 'webp']):
    '''
    return a list of pictures in the folder and the subfolders
    '''
    pictures = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith('.' + ext) for ext in extension):
                pictures.append(os.path.basename(file))
    return sorted(pictures)

def convert_2d_list_to_ints(input_list : list, decimals : int = 0) -> list :
    r'''
    Parameters
    ----------
    input_list : list
        2d list of numbers.
    
    decimals : int, optional = 0
        Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.
    Examples
    --------
    >>> convert_2d_list_to_ints([[395.0, 477.0], [395.0, 550.0], [465.0, 550.0], [465.0, 477.0]])
    [[395, 477], [395, 550], [465, 550], [465, 477]]
    '''
    if decimals > 0:
        return np.round(np.array(input_list), decimals=decimals).tolist()
    return np.round(np.array(input_list)).astype(int).tolist()

def density_function(amplitude : float = 5, interface_width : float = 0.05,
                    center_of_interface : float = 0.3, mean_density_value : float = 1125,
                    normalizer : float = 2*np.pi, plot : bool = False):
    r'''
    callable dunction
    adjust amplitude, interface_width [m], center_of_interface [m], mean_density_value [kg/m^3] to manipulate the erf to match the density function

    mean_density_value - is the place where the middle of tranaction occurs.
    if we set mean_density_value = 1125 it meant the the average value will be 1125 and amplitude = -5 it means to add and substrace 5 from mean_density_value,
    thus the function will be bounded from 1125+5 - 1125-5
    

    Examples
    --------
    >>> rho = density_function()
    rho(0.6)
    returns 1130
    '''
    if plot:
        # Generate x values
        x = np.linspace(0, 0.6, 1000)
        y = -amplitude*scipy.special.erf(normalizer/interface_width*(x - center_of_interface)) + mean_density_value
        # Plot the results
        plt.plot(x, y, label='erf(x)')
        plt.xlabel('x')
        plt.ylabel('erf(x)')
        plt.title('Error Function (erf) Plot')
        plt.legend()
        plt.show()
    return lambda x: - amplitude * scipy.special.erf(normalizer/interface_width*(x - center_of_interface)) + mean_density_value

def creat_callable_function_of_data(x_values = [0.5863636363636364, 0.5668831168831169, 0.5490259740259741, 0.5311688311688312, 0.5198051948051948, 0.4938311688311689, 0.4840909090909092, 0.4662337662337663, 0.45, 0.4353896103896105, 0.41266233766233773, 0.40616883116883123, 0.3883116883116884, 0.38344155844155847, 0.36396103896103904, 0.3477272727272728, 0.3347402597402598, 0.3266233766233767, 0.31850649350649357, 0.31688311688311693, 0.3120129870129871, 0.3071428571428572, 0.3006493506493507, 0.2990259740259741, 0.2990259740259741, 0.29740259740259745, 0.2957792207792208, 0.2892857142857143, 0.28766233766233773, 0.28441558441558445, 0.28116883116883123, 0.2795454545454546, 0.27142857142857146, 0.26818181818181824, 0.2633116883116884, 0.2535714285714286, 0.24707792207792212, 0.23409090909090913, 0.2243506493506494, 0.21785714285714292, 0.21136363636363642, 0.20487012987012992, 0.19512987012987015, 0.1837662337662338, 0.17402597402597408, 0.16915584415584423, 0.14480519480519483, 0.1204545454545455, 0.09772727272727276, 0.08636363636363638, 0.07012987012987014, 0.05551948051948054, 0.04090909090909094, 0.027922077922077945, 0.014935064935064954],
                            f_x_values = [1120, 1120, 1120, 1120, 1120, 1120, 1120, 1120, 1120, 1120, 1120, 1120, 1120,1120,1120, 1120.2217741935485, 1120.342741935484, 1120.5846774193549, 1121.0685483870968, 1121.4314516129034, 1122.0967741935485, 1122.8225806451615, 1123.4274193548388, 1123.9112903225807, 1124.2137096774195, 1124.758064516129, 1125.1814516129034, 1125.7862903225807, 1126.2701612903227, 1126.6935483870968, 1126.9959677419356, 1127.2983870967744, 1127.782258064516, 1127.9032258064517, 1128.4475806451615, 1128.75, 1128.9314516129034, 1129.2943548387098, 1129.3548387096776, 1129.3548387096776, 1129.475806451613, 1129.5362903225807, 1129.7, 1129.9, 1130, 1130, 1130, 1130, 1130, 1130, 1130, 1130, 1130, 1130, 1130]):
    return lambda x : np.interp(x, x_values, f_x_values, period=360)



def save_stream_line_on_pic(
    filename: str,
    save_path: str,
    stream_line = True,
    on_img: Optional[bool]=False,
    image_name: Optional[str]=None,
    window_size: Optional[int]=32,
    scaling_factor: Optional[float]=1,
    ax = None,
    width: Optional[float] = 0.003,
    show_invalid: Optional[bool]=True,
    show = False,
    **kw
):
    """ Displays quiver plot of the data stored in the file 
    
    
    Parameters
    ----------
    filename :  string
        the absolute path of the text file

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by 
        image_name

    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interrogation window size to fit the 
        background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background
        image to the vector field
    
    show_invalid: bool, show or not the invalid vectors, default is True

        
    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]
    
    
    See also:
    ---------
    matplotlib.pyplot.quiver
    
        
    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100, 
                                           width=0.0025) 

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field(Path('./exp1_0000.txt'), on_img=True, 
                                          image_name=Path('exp1_001_a.bmp'), 
                                          window_size=32, scaling_factor=70, 
                                          scale=100, width=0.0025)
    
    """

    # print(f' Loading {filename} which exists {filename.exists()}')
    a = np.loadtxt(filename)
    # parse
    x, y, u, v, flags, mask = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if on_img is True:  # plot a background image
                                        # import PIL
                                        # loaded_image = PIL.Image.open(image_name)  # Replace with the path to your image
                                        # dpi = loaded_image.info['dpi']
        im = read_image(image_name, False)
        # im = negative(im)  # plot negative of the image for more clarity
        xmax = np.amax(x) + window_size / (2 * scaling_factor)
        ymax = np.amax(y) + window_size / (2 * scaling_factor)
        ax.imshow(im, extent=[0.0, xmax, 0.0, ymax])    


    # first mask whatever has to be masked
    u[mask.astype(bool)] = 0.
    v[mask.astype(bool)] = 0.
    
    # now mark the valid/invalid vectors
    invalid = flags > 0 # mask.astype("bool")  
    valid = ~invalid

    # visual conversion for the data on image
    # to be consistent with the image coordinate system

    # if on_img:
    #     y = y.max() - y
    #     v *= -1


    if stream_line:
        X,Y = np.meshgrid(x,y)
        X,Y,U,V = txt_to_list(filename)
        ax.streamplot(X[::-1],Y[::-1],U[::-1],V[::-1])
    else:
        ax.quiver(x[valid], y[valid], u[valid], v[valid], color="b", width=width, **kw)

        if show_invalid and len(invalid) > 0:
            ax.quiver(x[invalid],y[invalid],u[invalid],v[invalid],color="r",width=width,**kw)
    
    
    # if on_img is False:
    #     ax.invert_yaxis()
    
    ax.set_aspect(1.)
    # fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')
    
    # plt.savefig(save_path)
    ax.set_axis_off()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi='figure')

    if show:
        plt.show()
    

    return fig, ax


def Change_contrast(pic_folder: str, output_folder: str = 'contrast', title: str = 'Choose contrast', file_extension: str = 'png'):
    viewer = napari.Viewer(title=title, axis_labels=['y', 'x'])
    image_paths = find_pictures(pic_folder, True)
    all_images = [read_image(path, False) for path in image_paths]
    Images = np.stack(all_images)
    viewer.add_image(Images, name='Images')
    napari.run()

    contrast_limits = viewer.layers[0].contrast_limits
    print("Contrast Limits:", contrast_limits)

    save_folder = os.path.join(pic_folder, output_folder)
    create_directory(save_folder)

    for el in find_pictures(pic_folder):
        img = read_image(el, False)  # Read the current image
        lower_limit, upper_limit = contrast_limits

        # Apply contrast adjustment
        adjusted_image = np.clip(img, lower_limit, upper_limit)
        normalized_image = (adjusted_image - lower_limit) / (upper_limit - lower_limit)
        normalized_image = normalized_image * 255

        # Save the processed image
        save_image(os.path.join(save_folder, os.path.basename(el)[:-4] + '.' + file_extension), normalized_image)
        
            # # for tif
            # normalized_image = normalized_image.astype(np.uint8)
            # # save_image(os.path.join(save_folder, os.path.basename(el)), normalized_image)




def select_directory():
    # Create the root window and hide it
    root = Tk()
    root.withdraw()

    # messagebox.showinfo("Information", "Choose the picture folder")

    # Open the directory chooser and get the selected directory
    directory = filedialog.askdirectory()

    # Destroy the root window
    root.destroy()

    return directory



########
# orginize files
########

def skip_cells(a, place : int = 2, always_include_last: bool = False):
    r'''
    this code always includes the first and last cell

     Examples
    --------
    use the skip_cells to creat a new list with relevant cells
    >>> skip_cells([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 5, always_include_last = True)
    [1, 6, 11, 16, 21, 23]
    >>> skip_cells([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 5, always_include_last = False)
    [1, 6, 11, 16, 21]
    '''
    if place == 0:
        place = 1
    if always_include_last:
        return [a[i] for i in range(len(a)) if i % place == 0 or i == len(a) - 1]
    else:
        return [a[i] for i in range(len(a)) if i % place == 0]

def save_cihx_file(folder_path, content : str, file_name : str):
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the content you want to save in the .cihx file
    content = '<cih> \n' + content + '\n </cih>'

    # Construct the file path for the .cihx file
    file_path = os.path.join(folder_path, file_name)

    try:
        # Open the file and write the content to it
        with open(file_path, "w") as file:
            file.write(content)
        print(f".cihx file saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the .cihx file: {str(e)}")

def skip_photos(path, skip_every : int = 2, copy_cihx : bool = True):
    picture_list = skip_cells(find_pictures(path,True), skip_every)
    new_path = path + " skipped"
    create_directory(new_path)
    if copy_cihx:
        fps = '<recordRate>' + str(round(round(find_fps(path))/skip_every)) + '</recordRate>'
        date = '<date>' + find_fps(path, '<date>') + '</date>'
        file_name = files_in_folder(path, 'cihx')[0]
        save_cihx_file(new_path, fps + " " + date,  file_name)
    for file in picture_list:
        shutil.copy(file, os.path.join(new_path, os.path.basename(file)))
    

def series_pattern(pic_folder : str, new_folder_name : str = 'Orginazed Patterns',
                       file_extension : str = 'tif', template : str = 'Pattern', A_pattern : str = '_A', B_pattern : str = '_B'):
    r'''
    this function is made for openPIV module, it creats a copy of all files that end with 'file_extension' in "pic_folder" and renames it as series in variable 'template' 'A_pattern' 'B_pattern' are mainly for making the relevant pattern


    Parameters
    ----------
    pic_folder : str
        Input str of the folder path.
    new_folder_name : str = 'Orginazed Patterns'
        Input str of the new folder name.
    file_extension : str = 'tif'
        the file extensions.
    template : str = 'Pattern'
        the template of the pattern.
    A_pattern : str = '_A'
        the template of A pattern.
    B_pattern : str = '_B'
        the template of B pattern.
        

    Returns
    -------
    None

    Examples
    --------
    use the series_pattern to creat a new directory with sorted pattern
    assume 'pic_folder' containes the following pictures: ['a1.tif', 'a2.tif', 'a3.tif']
    >>> series_pattern(path)
    ['Pattern_1_A.tiff', 'Pattern_1_B.tiff', 'Pattern_2_A.tiff', 'Pattern_2_B.tiff']
    where 'Pattern_1_A.tiff' is 'a1.tif', 'Pattern_1_B.tiff' is 'Pattern_2_A.tiff' and also is 'a2.tif', 'Pattern_2_B.tiff' is 'a3.tif'
    '''
    
    create_directory(os.path.join(pic_folder, new_folder_name))
    all_files = files_in_folder(pic_folder, file_extension, True)

    for ind, file in enumerate(all_files):
        if not (ind==0 or ind==len(all_files)-1):
            fily = template + '_%s' % (ind) + B_pattern + '.' + file_extension
            shutil.copy(file, os.path.join(pic_folder, new_folder_name, fily))
            fily2 = template + '_%s' % (ind+1) + A_pattern + '.' + file_extension
            shutil.copy(file, os.path.join(pic_folder, new_folder_name, fily2))
        else:
            if ind==len(all_files)-1:
                fily = template + '_%s' % (1) + A_pattern + '.' + file_extension
                shutil.copy(all_files[0], os.path.join(pic_folder, new_folder_name, fily))
                fily = template + '_%s' % (ind) + B_pattern + '.' + file_extension
                shutil.copy(file, os.path.join(pic_folder, new_folder_name, fily))



def find_fps(pic_folder : str, pattern : str = '<recordRate>'):
    r'''
    this function extracts the frame rate (fps) from the 'txt' camera file 

    Parameters
    ----------
    pic_folder : str
        Input str of the folder path.
    pattern : str = '<recordRate>'
        the pattern to extract the frame rate

    Returns
    -------
    fps : int

    Examples
    --------
    >>> find_fps(path)
    500
    '''
    end_pattern = '</' + pattern[1:]
    the_cihx = files_in_folder(pic_folder, 'cihx', True)[0]
    with open(the_cihx, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            first_last_line = [ i for i in range(len(lines)) if '<cih>' in lines[i] or '</cih>' in lines[i] ]
            xml = ''.join(lines[first_last_line[0]:first_last_line[-1]+1])
   
    # Initialize variables
    matches = []
    start = 0
   
    # Search for the pattern without regular expressions
    while True:
        start = xml.find(pattern, start)
        if start == -1:
            break

        end = xml.find(end_pattern, start)
        if end == -1:
            break

        match = xml[start + len (pattern) :end]
        matches.append(match)
        start = end + len(end_pattern)
    if pattern != '<recordRate>':
        return matches[0]
    return int(matches[0]) #transfer str to int

########
# PIV functions
########
def transform_coordinates_origin(x, y, u, v):
    r''' Converts coordinate systems from/to the image based / physical based 
    
    Input/Output: x,y,u,v

        image based is 0,0 top left, x = columns to the right, y = rows downwards
        and so u,v 

        physical or right hand one is that leads to the positive vorticity with 
        the 0,0 origin at bottom left to be counterclockwise
    
    '''
    y = y[::-1, :]
    v *= -1
    return x, y, u, v

def pixel_to_meter_scaling(atrribute, scaling_factor):
    r"""
    Apply an uniform scaling. from pixel to meter

    Parameters
    ----------
    atrribute : 2d np.ndarray
    scaling_factor : float
        the image scaling factor in pixels per meter

    Return
    ----------
    atrributes : 2d np.ndarray
    """
    atrributes = np.array(atrribute)
    return atrributes / scaling_factor


def transform_coordinates_pysical_back_up(sphere_location, pic_folder, scaling_factor, choose_origin = None):
    r'''
    trasfer coordinate system origin to bottom left corner, and change pixel to meter
    choose_origin = the location of the origin. in pixels
    '''
    #chen
    # origin_point = choose_origin_napari(read_image(find_pictures(pic_folder)[0])) / scaling_factor
    y_max_in_meter = read_image(find_pictures(pic_folder)[0]).shape[0] / scaling_factor
    scaled_location = np.array(sphere_location) / scaling_factor
    scaled_location[:, 1] = y_max_in_meter - scaled_location[:, 1]
    if choose_origin is not None:
        # origin_point = choose_origin_napari(read_image(find_pictures(pic_folder)[0]))
        scaled_location = scaled_location - np.array(choose_origin) / scaling_factor
    return scaled_location


def transform_coordinates_pysical(sphere_location, pic_folder, scaling_factor, origin_point = None):
    r'''
    trasfer coordinate system origin to bottom left corner, and change pixel to meter
    choose_origin = the location of the origin. in pixels
    '''
    sphere_location = np.array(sphere_location)
    y_max = read_image(find_pictures(pic_folder)[0]).shape[0]
    if origin_point is None:
        origin_point = choose_origin_napari(find_pictures(pic_folder)[0])
    sphere_location[:, 1] = y_max - sphere_location[:, 1]
    sphere_location = sphere_location - np.array([origin_point[0], y_max-origin_point[1]])
    return sphere_location / scaling_factor

def velocity_from_location(sphere_location : np.ndarray[np.float64] | list, dt : int | float | list):
    r'''
    Calculte velocity from locations and dt

    Parameters
    ----------
    sphere_location : 2d np.ndarray | list
        the x,y coordinates of the location. [[10, 10], [11, 18]]
    dt : int | float | list
        the time passed between each location. may be constant or a list of floats.
    
    Examples
    --------
    >>> velocity_from_location(sphere_location = [[10, 10], [11, 11], [12, 12], [15, 15]], dt = [1,2,3])
    array([[1. , 1. ], [0.5, 0.5], [1. , 1. ]])
    >>> velocity_from_location(sphere_location = [[10, 10], [11, 11], [12, 12], [15, 15]], dt = 5)
    array([[0.2, 0.2], [0.2, 0.2], [0.6, 0.6]])
    '''

    if type(dt) is int or type(dt) is float:
        dt = [dt]
    return np.diff(np.array(sphere_location), axis=0)/np.array(dt)[:, None]

def calc_velocity_from_acceptable_error(locations_meter : np.ndarray[np.float64] | list, scaling_factor : float, dt : int | float, velocity_acceptable_error : float = 0.002):
    r'''
    velocity_acceptable_error units depends of the locations_meter units.
    if locations_meter [m] then velocity_acceptable_error [m/s].
    another posibility to input locations_meter as [pixels] and scaling factor as 1 . in that case velocity_acceptable_error is in [pixel/s].
    
    Parameters
    ----------
    sphere_location : 2d np.ndarray | list
        the x,y coordinates of the location im meter. [[.18, 0.21], [0.1876, 0.23225]]
    dt : int | float
        the time passed between each location.
    '''
    skipped_dt = 1/scaling_factor/velocity_acceptable_error #[s]
    if round(skipped_dt/dt) < 1:
        factor = 1
    else: factor = round(skipped_dt/dt)
    locations_skipped = skip_cells(locations_meter, factor) #[m]
    return velocity_from_location(locations_skipped, factor*dt) # [m/s]



def interpolate_between_points(x, y, full_list, factor):
    r'''
    fills the misssing data points of y_interpolated_points
    '''
    y_interpolated = []
    for i in range(len(x) - 1):
        x_pair = [x[i], x[i + 1]]
        y_pair = [y[i], y[i + 1]]
        y_extended_pair = np.interp(full_list[i*factor:i*factor+factor+1], x_pair, y_pair, period=360)
        if i ==len(x)-1-1:
            y_interpolated.extend(y_extended_pair)
        else:
            y_interpolated.extend(y_extended_pair[:-1])
    return y_interpolated

def interpolate_between_points_2d(x, y, full_list, factor):
    r'''
    fills the misssing data points of y_interpolated_points
    '''
    x1, y1, full_list_1 = np.array(x), np.array(y), np.array(full_list)
    new_x = interpolate_between_points(x1[:,0], y1[:,0], full_list_1[:,0], factor)
    new_y = interpolate_between_points(x1[:,1], y1[:,1], full_list_1[:,1], factor)
    return np.column_stack((new_x, new_y))

def match_closest_points_to_shorter_list(Full_list, shorted_list):
    r'''
    FIX EXPLANTION


    len(shorted_list) = len(closest_indices)

    Examples
    --------
    >>> match_closest_points_to_shorter_list([(890, 450), (898, 4), (0,0), (1, 1), (2, 2), (20, 70), (10, 12), (300, 450), (890, 450)], [(1, 1), (10, 12), (300, 450)])
    [[2, 3, 4], [5, 6], [0, 1, 7, 8]]
    '''

    shorted_list = np.array(shorted_list)
    closest_indices = [[] for ELE in range(len(shorted_list))]
    for ind, target_point in enumerate(Full_list):
        closest_base_index = np.argmin([np.linalg.norm(np.array(target_point) - base_point) for base_point in shorted_list])
        closest_indices[closest_base_index].append(ind)
    return closest_indices

def append_the_same_value(locations_skipped, ranges):
    r'''
    helping function of match_closest_points_to_shorter_list
    '''
    maxy=-1
    for el in ranges:
        for ely in el:
            maxy = max(maxy, ely)
  
    new_hit = [None] * (maxy+1)
    for ind, el in enumerate(ranges):
        for elo in el:
            if ind == len(locations_skipped): new_hit[elo]=locations_skipped[ind-1]
            else: new_hit[elo]=locations_skipped[ind]
    return np.array(new_hit)



def rotate_image(img, deg : int = 90, return_grayscale : bool = False):
    # Ensure both images are in grayscale
    if not len(img.shape) == 2:
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape
        return_grayscale = False

    # Rotate the image by 90 degrees clockwise
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    if return_grayscale:
        return cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    return rotated_img



def txt_to_column(path : str, precision : int = 18):
    r'''
    openpiv saves data in txt. this function extract x,y,u,v
    recive Path of txt file
    return paramers as list
    '''
    np.set_printoptions(precision = precision)
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


    # OLD
    # with open(path, 'r') as f:
    #     data = f.read().split('\n')

    # columns = [[] for _ in range(len(data[0].split('\t')))]

    # for row in data:
    #     if row:
    #         for i, col in enumerate(row.split('\t')):
    #             columns[i].append(col)

    # daty = []
    # for i in range(4):
    #     holder = columns[i][1:]
    #     daty.append([])
    #     for el in holder:
    #         if el[0] == ' ':
    #             daty[i].append(float(el[1:]))
    #         else:
    #             daty[i].append(float(el))
    
    # return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

def txt_to_list(path : str):
    r'''
    THIS IS DIFFERENT FROM txt_to_column
    openpiv saves data in txt. this function extract x,y,u,v
    recive Path of txt file
    return paramers as a 2d lists (meshgrid)

    Parameters
    ----------
    path : str
        Input str of the txt path.



    Returns
    -------
    touple of np.arrays
        each cell correspond to the txt frame, the first array is x the second is y and so on.

    Examples
    --------
    use the txt_to_list to extract data from txt into orginaize lists
    >>> txt_to_list(path)
    [[1,2,3], [1,2,3], [1,2,3]],  [[1,1,1], [2,2,2], [3,3,3]],  [[1,1,1], [2,2,2], [3,3,3]],  [[1,1,1], [2,2,2], [3,3,3]]
    '''
    # a = np.loadtxt(filename)

    with open(path, 'r') as f:
        data = f.read().split('\n')

    columns = [[] for _ in range(len(data[0].split('\t')))]

    for row in data:
        if row:
            for i, col in enumerate(row.split('\t')):
                columns[i].append(col)
    
    # checking the y culum
    holder = columns[1][1:]
    colums = 0
    while holder[0] == holder[colums]:
        colums = colums + 1

    daty = []
    for i in range(4):
        holder = columns[i][1:]
        daty.append([])
        counter = 0
        for el in holder:
            if counter == 0:
                daty[i].append([])
            if el[0] == ' ':
                daty[i][-1].append(float(el[1:]))
                counter = counter + 1
            else:
                daty[i][-1].append(float(el))
                counter = counter + 1
            if counter == colums:
                counter = 0
    return np.array(daty[0]), np.array(daty[1]), np.array(daty[2]), np.array(daty[3])




def track_object_series(image_dir : str, fft : bool = False, small_displacement : bool = False,
                        win_size : int = 50, show : bool = False, dual_object : Optional[int] = None,
                        image_number : int = 0,
                        sphere_location : list = [500, 517],
                        Object_data : list = [[610, 504], [610, 574], [680, 574], [680, 504]],
                        PTV_data : list = [[600, 470], [600, 730], [690, 730], [690, 470]]):
    r'''
    say a few words
    recive folder directory
    return coordinates of the object, in cutted is known changing the coordinates acoordinaly,
    cutted should be in the form of cutted= [220, 800, 180, 380] #[y1, y2, x1, x2], just like slicing image in numpy
    show is to plot if wanted
    adjust_coordinate if True, changes the coordinates accordint to the sphere selection

    Parameters
    ----------
    image_dir : str
        folder path containing all the images
    fft : bool = False
        whether to use fft or cv.2 built in function. the fft is slower but more robust
    dual_object : Optional[int] = None
        int : the sencond image number for better PTV. WARNING: increase run time by factor of 2

    *******fix return***********
    Returns
    -------
    touple of np.arrays
        each cell correspond to the txt frame, the first array is x the second is y and so on.
    
    **********fix example***********
    Examples
    --------
    use the track_object_series to 
    >>> track_object_series(path)

    '''
    the_second_object = dual_object
    if small_displacement:
        fft = True
    
    # # Read the first image
    image_files = find_pictures(image_dir, True)

                #OLD
                    # first_image_path = image_files[img_number]
                    # first_image = cv2.imread(first_image_path)
                    # x_max_in_pixel, y_max_in_pixel = first_image.shape[1], first_image.shape[0]

                    # #select the prior knowledge
                    # if small_displacement:
                    #     x111, y111, w111, h111 = select_area_napari_notebook(image_dir, 'Small Displacement! select the area containing the object', stack_photos=True)
                    # else:
                    #     x111, y111, w111, h111 = select_area_napari_notebook(image_dir, 'Select the area of trajectory', stack_photos=True)
                        
                    # # do not delete, important
                    # if w111%2 == 1:
                    #     w111 = w111 - 1
                    # if h111%2 == 1:
                    #     h111 = h111 - 1
                    # prior_knowledge = (slice(y111, y111 + h111), slice(x111, x111 + w111))

                    # # Select the object to follow
                    # x111, y111, w111, h111 = select_area_napari_notebook(first_image_path, 'Select the object to follow')


                    # # center the coordinates
                    # real_cor = select_circle_napari_advance(first_image_path)
                    # bias = [x111 + w111/2 - real_cor[0], y111 + h111/2 - real_cor[1]]
                #OLD    
    img_number, real_cor, object_range_cut, prior_knowledge = select_circle_napari_advance(image_dir,image_number=image_number, sphere_location=sphere_location, Object_data=Object_data, PTV_data=PTV_data, to_print=True)
  
    # prior_knowledge = slice(PTV_slicing[1], PTV_slicing[1] + PTV_slicing[3]), slice(PTV_slicing[0], PTV_slicing[0] + PTV_slicing[2])
    # object_range_cut = slice(obj_slicing[1], obj_slicing[1] + obj_slicing[3]), slice(obj_slicing[0], obj_slicing[0] + obj_slicing[2])

    bias = [object_range_cut[1].start + (object_range_cut[1].stop - object_range_cut[1].start)/2 - real_cor[0], object_range_cut[0].start + (object_range_cut[0].stop - object_range_cut[0].start)/2 - real_cor[1]]
    im_a_path = image_files[img_number]
    the_selected_image = read_image(im_a_path)
    object_pattern = the_selected_image[object_range_cut]

    x_max_in_pixel, y_max_in_pixel = the_selected_image.shape[1], the_selected_image.shape[0]

    # first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    # object_pattern = np.copy(first_image[y111 : y111 + h111, x111 : x111 + w111])

    if dual_object is not None:
        object_range_cut_2 = select_area_napari_notebook(image_files[dual_object], 'select the second object')
        second_image = read_image(image_files[dual_object])
        real_cor_2 = select_circle_napari(image_files[dual_object], title = 'seond bias')
        the_second_object_here = np.copy(second_image[object_range_cut_2])
        second_bias = [object_range_cut[1].start + (object_range_cut[1].stop - object_range_cut[1].start)/2 - real_cor_2[0], object_range_cut[0].start + (object_range_cut[0].stop - object_range_cut[0].start)/2 - real_cor_2[1]]
        the_second_object = [the_second_object_here, second_bias]
    

    tracked_positions = []
    # Loop through the rest of the images
    for image_path in image_files:
        if small_displacement:
            if len(tracked_positions)>0:
                x2, y2 = tracked_positions[-1]
                prior_knowledge = (slice(max(0, y2 - win_size), min(y2 + win_size, y_max_in_pixel)), slice(max(0, x2 - win_size), min(x2 + win_size, x_max_in_pixel)))

        x1, y1 = track_object(object_pattern, image_path, bias = bias, prior_knowledge = prior_knowledge, fft = fft, dual_object = the_second_object)
        tracked_positions.append((x1, y1))

        # delete me
        # if True:
        #     plt.title('wow')
        #     plt.scatter(tracked_positions[-1][0],tracked_positions[-1][1], c = 'red')
        #     plt.imshow(cv2.imread(image_path)[prior_knowledge])
        #     plt.show()
        # delete me
     
        if show:
            print(os.path.basename(image_path))
            print(x1, y1)
            plt.title('Nice')
            plt.scatter(tracked_positions[-1][0],tracked_positions[-1][1], c = 'red')
            plt.imshow(read_image(image_path))
            plt.show()
        
    return tracked_positions

def track_object(object_pattern : np.ndarray | list | str, image_path : str, bias : Optional[list] = None,
                  prior_knowledge : Optional[tuple] = None, fft : bool = False, cutted : list | None = None, show : bool = False, dual_object : Optional[list] | np.ndarray = None):
    r'''
    finds the object in a picture and returns the center coordinates

    Parameters
    ----------
    object_pattern : np.ndarray | list | str
        a Gray scale image of the object stored in a numpy arrray or a list, better to use numpy array.
        may also be the picture path as str.
    image_path : str
        the image path where to search the object at
    bias : list | None = None
        the pixels change in x and y for the real center of circle, if known.
    prior_knowledge : tuple | None = None
        makes the search area smaller, decrease runtime significantly,
        should be in the form of prior_knowledge = (slice(y1, y2), slice(x1, x2)), just like slicing image in numpy.
    cutted : list | None = None
        should be in the form of cutted = [220, 800, 180, 380] [y1, y2, x1, x2], just like slicing image in numpy
    show : bool = False
        if True plots the picture
    fft : bool = False
        uses fft - Fase fourier transform to find the coordinates. should be much faster than cross_correlation
 
    ###
    CUTTED HASEN'T BEEN TESTED YET
    ####
    
    Returns
    -------
    (x, y) : tuple 
        the coordinates of the center of the object, if cutted is known changing the coordinates acoordinaly
    '''
    # making sure image_path is str
    # if isinstance(image_path, os.PathLike):
    image_path = str(image_path)

    #If object_pattern is a PATH
    if type(object_pattern) is str:
        first_image = read_image(object_pattern)
        how_to_slice = select_area_napari_notebook(first_image, 'Selecte object to find')
        if bias is None:
            real_cor = select_circle_napari(object_pattern)
            bias = [how_to_slice[1].start + (how_to_slice[1].stop - how_to_slice[1].start)/2 - real_cor[0], how_to_slice[0].start + (how_to_slice[0].stop - how_to_slice[0].start)/2 - real_cor[1]]

        object_pattern = np.copy(first_image[how_to_slice])

    # checks if the picture is grayscale
    if not len(object_pattern.shape) == 2:
        object_pattern = cv2.cvtColor(object_pattern, cv2.COLOR_BGR2GRAY)
    
    imagey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if object_pattern.shape[0] > imagey.shape[0] or object_pattern.shape[1] > imagey.shape[1]:
        raise ValueError('template shape is biiger that the image shape')

    # cuting the object_pattern lengh to make sure its lengh is even. center of the image must be a whole number
    if np.array(object_pattern).shape[1] % 2 == 1:
        object_pattern = np.copy(object_pattern)[:, :-1]
    if np.array(object_pattern).shape[0] % 2 == 1:
        object_pattern = np.copy(object_pattern)[:-1, :]
    
    # List to store the tracked object positions
    tracked_positions = []

    if not prior_knowledge is None:
        imagey = np.copy(imagey[prior_knowledge])

    if dual_object is not None:
        fft = True
    
    if fft:
        if dual_object is not None:
            x1, y1, max_value1 = fft_correlation(imagey, object_pattern, return_max_value = True)
        else:
            x1, y1 = fft_correlation(imagey, object_pattern)

    else:
        x1, y1 = cv2_matchTemplate(imagey, object_pattern)

    if dual_object is not None:
        dual_object, second_bais = dual_object
        xx1, yy1, max_value11 = fft_correlation(imagey, dual_object, return_max_value = True)
        
        if max_value11 > max_value1:
            x1, y1 = xx1, yy1
            bias = second_bais

    if not prior_knowledge is None:
        x1, y1 = x1 + prior_knowledge[1].start, y1 + prior_knowledge[0].start
        
    tracked_positions.append((x1, y1))
    
    if not cutted is None:
        a, b, c, d = cutted
        tracked_positions = [(x+c, y+a) for (x, y) in tracked_positions]
        print('This is the coordinates of the ORIGINAL photos.')

    if not bias is None:
        # tracked_positions = [(round(x - bias[0]), round(y - bias[1])) for (x, y) in tracked_positions]
        tracked_positions = [(x - bias[0], y - bias[1]) for (x, y) in tracked_positions]
    
    if show:
        print(os.path.basename(image_path))
        print(tracked_positions[0])

        plt.title('selected object to find in picture')
        plt.imshow(object_pattern)
        plt.show()
        
        if bias is not None:
            plt.title('taken into account the center of the sphere bias')
        else:
            plt.title('Did not take into account the center of the sphere bias')
        plt.scatter(tracked_positions[0][0], tracked_positions[0][1], s = 60, c = 'red')
        plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        plt.show()
    return tracked_positions[0]


def find_the_slice(image_path : str, object_image: np.ndarray | str , path : bool = False, prior_knowledge : Optional[tuple] = None, fft : bool = False):
    r'''
    recive:
      image_path - the image we search the object
      object_image - the object we are looking for 2d numpy array grayscale.
      prior_knowledge - slice, an approximated sphere location slicing the image to improve runtime.
      prior_knowledge = slice(400,700), slice(150,400)
    return:
     parameters for slice. example [100,200,300,500] #[y1, y2, x1, x2]
    '''
    if path:
        object_image = cv2.imread(object_image, cv2.IMREAD_GRAYSCALE)
    
    first_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if not prior_knowledge is None:
        new_image = first_image[prior_knowledge]
        if object_image.shape[0] > new_image.shape[0] or object_image.shape[1] > new_image.shape[1]:
            print('object_pattern is bigger than the searh image, using cv2.TM_SQDIFF_NORMED instead of fft')
            fft = False
        if fft:
            x1, y1 = fft_correlation(new_image, object_image)
        else:
            x1, y1 = cv2_matchTemplate(new_image, object_image)
        x1, y1 = x1 + prior_knowledge[1].start, y1 + prior_knowledge[0].start

    else:
        x1, y1 = cv2_matchTemplate(first_image, object_image)
    
    shapey = object_image.shape
    x1, y1 = x1 - shapey[1]/2, y1 - shapey[0]/2
    return y1, y1 + shapey[0], x1, x1 + shapey[1]



def fft_correlation(img : np.ndarray | list, template : np.ndarray | list, show : bool = False, return_max_value : bool = False):
    r'''
    finds the object in a picture and returns it coordinates Using FFT algoritm for 2d arrays (Gray scale pictures)

    Parameters
    ----------
    img : np.ndarray | list
        the grayscale image
    template : np.ndarray | list
        the grayscale template we are looking for
    show : bool = False
        plot the corrdinates of the object
 
    Returns
    -------
    (x, y) : tuple 
        the coordinates of the center of the object
    '''

    # Ensure both images are numpy array
    img = np.array(img)
    template = np.array(template)

    # Ensure both images are in grayscale
    if not len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not len(template.shape) == 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)   
 
    img = img - img.mean()
    template = template - template.mean()

    # Compute 2D FFT of the image and template
    img_fft = np.fft.fft2(img)
    template_fft = np.fft.fft2(template, s = img.shape)

    # Compute cross-correlation in the frequency domain
    cross_correlation_fft = img_fft * np.conjugate(template_fft)

    # Perform inverse FFT to obtain the cross-correlation in the spatial domain
    cross_correlation = np.fft.ifft2(cross_correlation_fft)

    # Find the location of the maximum value in the cross-correlation
    y, x = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    
    x, y = x + template.shape[1]/2, y + template.shape[0]/2
  
    if show:
        print('corr and img shape', cross_correlation.shape, img.shape)
        plt.scatter(x - template.shape[1]/2, y - template.shape[0]/2, s = 100, c = 'black')
        plt.imshow(np.abs(cross_correlation), cmap='hot')
        plt.show()
        plt.scatter(x, y)
        plt.imshow(img)
        plt.show()
    
    if return_max_value:
        return x, y, np.argmax(np.abs(cross_correlation))
    return x, y


def cv2_matchTemplate(img : np.ndarray | list, template : np.ndarray | list, show : bool = False):
    r'''
    finds the object in a picture and returns it coordinates Using Opencv functions for 2d arrays (Gray scale pictures)

    Parameters
    ----------
    img : np.ndarray | list
        the grayscale image
    template : np.ndarray | list
        the grayscale template we are looking for
    show : bool = False
        plot the corrdinates of the object
 
    Returns
    -------
    (x, y) : tuple 
        the coordinates of the center of the object
    '''
    img2 = img.copy()
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #choose the method
    meth = methods[-1]
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if show:
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
    return top_left[0] + w/2, top_left[1] + h/2

def correlate2d_for_me(img : np.ndarray | list, template : np.ndarray | list, show : bool = False):
    r'''
    finds the object in a picture and returns it coordinates

    Parameters
    ----------
    img : np.ndarray | list
        the grayscale image
    template : np.ndarray | list
        the grayscale template we are looking for
    show : bool = False
        plot the corrdinates of the object
 
    Returns
    -------
    (x, y) : tuple 
        the coordinates of the center of the object
    '''
    # Ensure both images are numpy array
    img = np.array(img)
    template = np.array(template)

    # Ensure both images are in grayscale
    if not len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not len(template.shape) == 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  

    img = img - img.mean()
    template = template - template.mean()
    
    corr = scipy.signal.correlate2d(img, template)

    max_coords = np.argmax(corr)
    y, x = np.unravel_index(max_coords, corr.shape)

    if not (template.shape[0] % 2 == 0 and template.shape[1] % 2 == 0):
        print(' the cordinates are shifted becuase template.shape is not an even number')
    
    x, y = x - template.shape[1] / 2, y - template.shape[0] / 2

    if show:
        print('corr and img shape', corr.shape, img.shape)
        print(x, y)
        plt.scatter(x + template.shape[1] / 2, y + template.shape[0] / 2, s = 100, c='black')
        plt.imshow(corr, cmap='hot')
        plt.show()
        plt.scatter(x, y, s = 100, c='black')
        plt.imshow(img)
        plt.show()
    
    print('in correlate2d there is an unknown bias, most of the time 1 pixel error in both x and y, thats why I added 1 pixel to x and y and now it should be fine\nin fft no bias has been reported.')
    #bais fix
    x, y = x + 1, y + 1
    return x, y


def enhance_coordinates(path : str, corr : list, threshold1 : int = 200, threshold2 : int = 600, bias : int = 36, show : bool = False, printy : bool = False):
    r'''
    finds diameter of a sphere automaticly, working good on matplotlib circles,
    need to check on real sphere images from the lab
    '''
    raw_image = cv2.imread(path)
    cut_seq = slice(corr[1]-bias, corr[1]+bias), slice(corr[0]-bias, corr[0]+bias)
    if show:
        cv2.imshow('Original Image', raw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #mort
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image[cut_seq]
    bright_filtered = bright_objects(gray_image)


    # bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
    bilateral_filtered_image = cv2.bilateralFilter(bright_filtered, 5, 175, 175)

    if show:
        cv2.imshow('Bilateral', bilateral_filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    edge_detected_image = cv2.Canny(bilateral_filtered_image, threshold1=threshold1, threshold2=threshold2)

    if show:
        cv2.imshow('Edge', edge_detected_image)
        cv2.circle(edge_detected_image, (corr[0], corr[1]), 5, (225, 255, 100), 2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (len(approx) < 23) & (area > 30)):
            contour_list.append(contour)
    
    # if show:
    #     cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
    #     cv2.imshow('Objects Detected',raw_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        
    xis = []
    yis = []
    for contour in contour_list:
        for el in contour:
            col = el[:, 0][0]
            xis.append(col)
            yis.append(el[:, 1][0])
    
    # right and left
    max_index = xis.index(max(xis))
    min_index = xis.index(min(xis))
    most_right = (xis[max_index], yis[max_index])
    most_left = (xis[min_index], yis[min_index])

    # up and down
    max_index_y = yis.index(max(yis))
    min_index_y = yis.index(min(yis))
    most_down = (xis[max_index_y], yis[max_index_y])
    most_up = (xis[min_index_y], yis[min_index_y])


    if printy:
        print(cut_seq, cut_seq[1].start)
        print(os.path.basename(path), np.linalg.norm(np.array(most_right)-np.array(most_left)))
    new_corr = [(most_right[0]+most_left[0])/2+cut_seq[1].start, (most_down[1]+most_up[1])/2+cut_seq[0].start]

    if show:
        cv2.drawContours(gray_image, contour_list,  -1, (255,0,0), 2)
        cv2.imshow('center is',gray_image)
        print('mort')
        cv2.circle(gray_image, (round((most_right[0]+most_left[0])/2), round((most_down[1]+most_up[1])/2)), 1, (225, 255, 255), 2)
        print('mort')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return new_corr



# #######################
# # napari #####################
# #######################


def add_index_to_points(points : list):
    r'''
    A helping function for plot in napari.
    adds index to each point.

    Parameters
    ----------
    points : list | 2d np.array
        Input points in a list, for example [[10,10], [20,53]]
        
    Returns
    -------
    NumPy arrays of points with the index added to each point

    Examples
    --------
    >>> add_index_to_points(points = [[10,10], [20,53]])
    [[0, 10, 10], [1, 20, 53]]
    '''
    return np.hstack((np.arange(len(points)).reshape(-1, 1), np.array(points)))

def replace_elements_3(arr):
    r'''
    Swap elements at positions 1 and 2 in each sublist
    nupari uses [y,x] insstead of [x,y].

    input [[1, 20, 30], [2, 30, 55]]
    output [[1, 30, 20], [2, 55, 30]]
    '''
    np_arr = np.array(arr)
    np_arr[:, [1, 2]] = np_arr[:, [2, 1]]
    return np_arr

def replace_elements_2(arr):
    r'''
    Swap elements at positions 0 and 1 in each sublist.

    input [[20, 30], [30, 55]]
    output [[30, 20], [55, 30]]
    '''
    np_arr = np.array(arr)
    np_arr[:, [0, 1]] = np_arr[:, [1, 0]]
    return np_arr

def rectangle_coordinates_to_slice(coordinates_from_napari, return_int : bool = False):
    r'''
    coordinates_from_napari of a rectaingle is in the form [[x,y], [x,y], [x,y], [x,y]]
    top left corner, bottom left corner, bottom right corner, top right corner

    note: originaly napari returns [[y,x], [y,x], [y,x], [y,x]], very strange behaviour. we use the regular form [[x,y], [x,y], [x,y], [x,y]]

    '''
    x111, y111, w111, h111 = coordinates_from_napari[0][0], coordinates_from_napari[0][1], coordinates_from_napari[2][0] - coordinates_from_napari[0][0], coordinates_from_napari[2][1] - coordinates_from_napari[0][1]

    if return_int:
        x111, y111, w111, h111 = round(coordinates_from_napari[0][0]), round(coordinates_from_napari[0][1]), round(coordinates_from_napari[2][0] - coordinates_from_napari[0][0]), round(coordinates_from_napari[2][1] - coordinates_from_napari[0][1])
        if w111%2 == 1:
            w111 = w111 - 1
        if h111%2 == 1:
            h111 = h111 - 1
    
    return slice(y111, y111 + h111), slice(x111, x111 + w111)

def display_images_with_points_napari(directory : str, points : list = None, scaling_factor : float = 1,
                                       pysical : bool = False, title : str = 'Show images', symbol ='cross') -> int:
    r'''
    pysical is to convert pixel to meter.
    '''
    viewer = napari.Viewer(title = title, axis_labels=['y', 'x'])
    image_paths = find_pictures(directory, True)
    all_images = [read_image(path) for path in image_paths]
    Images = np.stack(all_images)
    if pysical:
        viewer.add_image(Images, name='Images', scale=[1/scaling_factor, 1/scaling_factor])
        if points is not None:
            points = np.array(points)/scaling_factor
            pointss = add_index_to_points(points)
            Points = replace_elements_3(pointss)
            points_layer = viewer.add_points(Points, name='Points', size=10/scaling_factor, face_color='red', edge_color='red')
        viewer.scale_bar.unit = "m"

    else:
        viewer.add_image(Images, name='Images')
        if points is not None:
            pointss = add_index_to_points(points)
            Points = replace_elements_3(pointss)
            points_layer = viewer.add_points(Points, name='Points', symbol=symbol, size=10, face_color='red', edge_color='red')
        viewer.scale_bar.unit = "pixel"


    viewer.axes.visible = True
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True
    # viewer.scale_bar.color = 'violet'

    # set the first image to show in napari slider
    viewer.dims.current_step = (0,) + viewer.dims.current_step[1:]
    
    # napari.run()
    settings = napari.settings.get_settings()
    settings.application.playback_fps = round(len(image_paths)/5)
    settings.application.ipy_interactive = True
    # settings.application.playback_mode = 'once'
    settings.application.save_window_geometry = True
    
    viewer.show(block=True)

    to_return_1 = viewer.dims.current_step[0]

    #trying to clear RAM
    viewer.close()
    del viewer
    del Images
    del all_images
    import gc
    gc.collect()

    
    return to_return_1


def choose_origin_napari(image_path : str, origin : list = [0, 1024], point_size : int = 30, title : str = 'choose origin') -> list:
    r'''
    return the choosen location of origin in [x,y] # [pixel]
    return [x,y]
    '''
    viewer = napari.Viewer(title = title, axis_labels=['y', 'x'])
    the_image = read_image(image_path)
    viewer.add_image(the_image, name='Image')

    origin_layer = viewer.add_points(replace_elements_2([origin]), name='Coordinates origin', size=point_size, face_color='yellow', edge_color = 'yellow')

    viewer.axes.visible = True
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True
    viewer.scale_bar.unit = "pixel"

    viewer.show(block=True)
    to_return = replace_elements_2(origin_layer.data)[0]

    #trying to clear RAM
    viewer.close()
    del viewer
    import gc
    gc.collect()

    return to_return




def calculate_distance_between_points_napari_notebook_line(image_path : str,
                                                           title = 'Select diameter with a line, hold shift to make a horizontal line',
                                                           initial_guess = [[630, 545], [663, 545]], line_name = 'diameter',
                                                           to_print : bool = True):
    viewer = napari.Viewer(title = title, axis_labels=['y', 'x'])
    viewer.add_image(read_image(image_path), name='Image')
    line_layer = viewer.add_shapes(data=replace_elements_2(initial_guess), shape_type='line', edge_color='red', edge_width=2, name=line_name)


    viewer.axes.visible = True
    viewer.scale_bar.unit = "pixel"
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True

    viewer.show(block=True)
    point1, point2 = line_layer.data[0][0], line_layer.data[0][1]
    diameter = np.linalg.norm(point1 - point2)
    if to_print:
        print(f'initial_guess = {convert_2d_list_to_ints(replace_elements_2(line_layer.data[0]), decimals=2)}')

    #trying to clear RAM
    viewer.close()
    del viewer
    import gc
    gc.collect()
    return diameter

def calculate_minimun_and_maximun_diameter_napari_notebook(image_path : str,
                                                               title = 'Select minimun amd maximun diameter with the lines, hold shift to make a horizontal line',
                                                               initial_guess_1 = [[207, 197], [258, 197]],
                                                               initial_guess_2 = [[207, 197], [267, 197]]):
    viewer = napari.Viewer(title = title, axis_labels=['y', 'x'])
    viewer.add_image(read_image(image_path), name='Image')
    # points_layer = viewer.add_points(name='Points', size=1, face_color='red', edge_color='red')
    lower_line_layer = viewer.add_shapes(data=replace_elements_2(initial_guess_1), shape_type='line', edge_color='red', edge_width=2, name='minimum')
    upper_line_layer = viewer.add_shapes(data=replace_elements_2(initial_guess_2), shape_type='line', edge_color='blue', edge_width=2, name='maximun')


    viewer.axes.visible = True
    viewer.scale_bar.unit = "pixel"
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True
   
    viewer.show(block=True)
  
    point1, point2 = lower_line_layer.data[0][0], lower_line_layer.data[0][1]
    minim = np.linalg.norm(point1 - point2)

    point11, point22 = upper_line_layer.data[0][0], upper_line_layer.data[0][1]
    maxi = np.linalg.norm(point11 - point22)

    #trying to clear RAM
    viewer.close()
    del viewer
    import gc
    gc.collect()

    return minim, maxi


def select_area_napari_notebook(image_path : str | list, title = 'Select area',
                                rectangle_data = [[500, 500], [500, 600], [600, 600], [600, 500]],
                                stack_photos = False, return_slice : bool = True, return_int : bool = True, to_print : bool = True):
    # napari uses [y,x] thats why we use replace_elements_2 function so we input and output [x,y]
    viewer = napari.Viewer(title = title, axis_labels=['y', 'x'])
    if type(image_path) is str:
        if stack_photos:
            image_paths = find_pictures(image_path, True)
            all_images = [read_image(path) for path in image_paths]
            Images = np.stack(all_images)
            viewer.add_image(Images, name='Images')
        else:
            viewer.add_image(read_image(image_path), name='Image')
    else:
        viewer.add_image(image_path, name='Image')


    # Coordinates of the square [top-left, bottom-left, bottom-right, top-right]
    square_layer = viewer.add_shapes(replace_elements_2(rectangle_data), shape_type='rectangle', edge_width=2, edge_color='red', face_color='transparent', name='Area for PIV')
   
    viewer.axes.visible = True
    viewer.scale_bar.unit = "pixel"
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True
    if stack_photos:
        viewer.dims.current_step = (0,) + viewer.dims.current_step[1:]

    viewer.show(block=True)


    data = square_layer.data[0]
    adata = replace_elements_2(data)

    if to_print:
        print(f'rectangle_data = {convert_2d_list_to_ints(adata)}')


    if return_slice:
        to_return = rectangle_coordinates_to_slice(adata, return_int)
        #trying to clear RAM
        viewer.close()
        del viewer
        import gc
        gc.collect()
        return to_return
  
    else:
        to_return = adata
        #trying to clear RAM
        viewer.close()
        del viewer
        import gc
        gc.collect()
        return to_return

def select_circle_napari_advance(image_folder_path : str,
                                title = "Select the sphere, the area for PTV and the image number",
                                image_number : int  = 0,
                                sphere_location : list = [647, 545], Object_data : list = [[500, 500], [500, 574], [574, 574], [574, 500]],
                                PTV_data : list = [[200, 200], [200, 800], [500, 800], [500, 200]],
                                origin : list = [0, 1024], point_size = 30,
                                return_int : bool  = True, return_slice : bool = True, to_print : bool = False):
    r'''
    select a circle using the mouse, for fine tuning please use the "a s d w + -" keys.
    return center coordinates and diameter of the circle
    '''
    print('use arrow keys for fine adjustments, toggle Points visability on and off to see changes \nSelect the sphere, the area for PTV and the image number \nimportant!!! make sure the slider is on the relevant frame')
    viewer = napari.Viewer(title = title, axis_labels=['y', 'x'])
    # stack_photos
    image_paths = find_pictures(image_folder_path, True)
    all_images = [read_image(path) for path in image_paths]
    Images = np.stack(all_images)
    viewer.add_image(Images, name='Images')
        # single image:
        #     viewer.add_image(read_image(image_folder_path), name='Image')
    points_layer = viewer.add_points(replace_elements_2([sphere_location])[0], name='Center of Sphere', size=point_size, face_color='red', edge_color = 'red')

    # Coordinates of the square [top-left, bottom-left, bottom-right, top-right]
    object_layer = viewer.add_shapes(replace_elements_2(Object_data), shape_type='rectangle', edge_width=2, edge_color='red', face_color='transparent', name='Object to follow')
    PTV_shape = viewer.add_shapes(replace_elements_2(PTV_data), shape_type='rectangle', edge_width=2, edge_color='blue', face_color='transparent', name='Area for PTV')
    
    # origin_layer = viewer.add_points(replace_elements_2(origin), name='Coordinates origin', size=point_size, face_color='yellow', edge_color = 'yellow')

    viewer.axes.visible = True
    viewer.scale_bar.unit = "pixel"
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True
    viewer.dims.current_step = (image_number,) + viewer.dims.current_step[1:]


    # @viewer.bind_key('Enter')
    def up_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][0] -= 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Up', lambda v: up_arrow(v))

    def down_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][0] += 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Down', lambda v: down_arrow(v))

    def left_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][1] -= 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Left', lambda v: left_arrow(v))

    def right_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][1] += 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Right', lambda v: right_arrow(v))

    viewer.show(block=True)
    
    data_object = replace_elements_2(object_layer.data[0])
    data_PTV = replace_elements_2(PTV_shape.data[0])

    if to_print:
        print(f'image_number = {viewer.dims.current_step[0]},\nsphere_location = {replace_elements_2([points_layer.data[0]])[0].tolist()},\nObject_data = {convert_2d_list_to_ints(data_object)},\nPTV_data = {convert_2d_list_to_ints(data_PTV)}')

    if return_slice:
        object_slice = rectangle_coordinates_to_slice(data_object, return_int)
        PTV_slice = rectangle_coordinates_to_slice(data_PTV, return_int)
        to_return = viewer.dims.current_step[0], replace_elements_2([points_layer.data[0]])[0], object_slice, PTV_slice
        #trying to clear RAM
        viewer.close()
        del viewer
        import gc
        gc.collect()
        return to_return
    else:
        to_return = viewer.dims.current_step[0], replace_elements_2([points_layer.data[0]])[0], data_object, data_PTV
        #trying to clear RAM
        viewer.close()
        del viewer
        import gc
        gc.collect()
        return to_return


def select_circle_napari(image_path : str, title = "select the sphere's center", initial_guess = [647, 545], size = 34):
    r'''
    select a circle using the mouse, for fine tuning please use the "a s d w + -" keys.
    return center coordinates and diameter of the circle
    '''
    print('use arrow keys for fine adjustments, toggle Points visability on and off to see changes')
    viewer = napari.Viewer(title = title, axis_labels=['y', 'x'])
    viewer.add_image(read_image(image_path), name='Image')
    points_layer = viewer.add_points(replace_elements_2([initial_guess])[0], name='Points', size=size, face_color='red', edge_color = 'red')

    viewer.axes.visible = True
    viewer.scale_bar.unit = "pixel"
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True

    # @viewer.bind_key('Enter')
    def up_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][0] -= 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Up', lambda v: up_arrow(v))

    def down_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][0] += 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Down', lambda v: down_arrow(v))

    def left_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][1] -= 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Left', lambda v: left_arrow(v))

    def right_arrow(viewer):
        nonlocal points_layer
        points_layer.data[0][1] += 1
        # points_layer = viewer.add_points(replace_elements_2([points_layer.data[0][0]+1, points_layer.data[0][1]])[0], name='Points', size=size, face_color='red', edge_color = 'red')
    viewer.bind_key('Right', lambda v: right_arrow(v))

    viewer.show(block=True)
    to_return = replace_elements_2([points_layer.data[0]])[0]

    #trying to clear RAM
    viewer.close()
    del viewer
    import gc
    gc.collect()
    return to_return


def bright_objects(image, filter_size=7):
    '''
    recive:
    gray image - an image of (x,y,1) dimentions.
    return the bright ocbect, masked.
    Ideal for finding diameter.
    '''
    imcopy = np.copy(image)
    background = scipy.ndimage.gaussian_filter(scipy.ndimage.median_filter(image, filter_size), filter_size)
    mask = background > filters.threshold_otsu(background)
    imcopy[mask] = 254
    return imcopy



# #######################
# # Sphere_diameter_calculator #####################
# #######################
def detect_contour_of_big_object(path, countour_lenght = 200, show = False, plot = False, rotate_deg=0):
    r'''
    Function of Sphere_diameter_calculator.ipynb
    '''
    import imutils

    image = rotate_image(read_image(path, False), deg = rotate_deg)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    if show:
        # show the original image and the edge detected image
        print("STEP 1: Edge Detection")
        cv2.imshow("Image", image)
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    fixed_cnts = []

    # Iterate through contours
    for contour in cnts:
        # Calculate the length of the contour using arc length
        length = cv2.arcLength(contour, closed=True)
        
        # Check if the length is greater than 200 pixels
        if length >= countour_lenght:
            # Add the contour to the fixed_cnts list
            fixed_cnts.append(contour)
    

    c = np.vstack(fixed_cnts)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    if show:
        cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(image, extRight, 8, (0, 255, 0), -1)
        cv2.circle(image, extTop, 8, (255, 0, 0), -1)
        cv2.circle(image, extBot, 8, (255, 255, 0), -1)

        cv2.drawContours(image, fixed_cnts, -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if plot:
        # plt.figure()
        plt.imshow(image)
        for cnt in fixed_cnts:
            cnt = cnt.squeeze()  # Remove unnecessary dimensions
            plt.plot(cnt[:, 0], cnt[:, 1], color='green', linewidth=2)
            
        plt.scatter(extLeft[0], extLeft[1], color = 'red')
        plt.scatter(extRight[0], extRight[1], color = 'red')
        plt.scatter(extTop[0], extTop[1], color = 'blue')
        plt.scatter(extBot[0], extBot[1], color = 'blue')
        # plt.show()
        # plt.close()
    return ((extLeft, extRight), (extTop, extBot))


# #######################
# # Control volume #####################
# #######################

def find_closest_range_to_values_in_array(arry: list, target: float):
    r'''
    input:
    arr - array
    target - value
    output:
    2 indexes (integres) "start" and "end" representing the range where the closest value is

    example:
    arr = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
    target = 2.8

    the function rertuns: 6, 8
    '''
    arr = np.array(arry)
    diff = np.abs(arr - target)
    closest_index = np.argmin(diff)
    target = arr[closest_index]
 
    start = end = closest_index
    while start > 0 and arr[start - 1] == target:
        start -= 1
    while end < len(arr) - 1 and arr[end + 1] == target:
        end += 1
      
    return start, end

def find_closest_values_in_array(arry: list, target: float):
    r'''
    input:
    arr - array
    target - value
    output:
    index of the closest value
    '''
    arr = np.array(arry)
    diff = np.abs(arr - target)
    closest_index = np.argmin(diff)
    target = arr[closest_index]
    return closest_index


##
# FOR Drag Force
##

# def cartesian_to_polar(x1, y1, u1, v1, center = (0,0)):
#     '''
#     input x, y, u, v arrays
#     return coorisponding array in poolar coordinates system
#     '''
#     x, y, u, v = np.array(x1) - center[0], np.array(y1) - center[1], np.array(u1), np.array(v1)

#     r = np.sqrt(x ** 2 + y ** 2)
#     theta = np.arctan2(y, x)

#     v_r = (x*u + y*v)/r
#     v_theta = (x*v - y*u)/r

#     return r, theta, v_r, v_theta

def cartesian_to_polar(x1 : list, y1: list, u1: list, v1: list, center: list | tuple = (0,0)):
    r'''
    x1 and y1 must be right-hand coordinate system left bottom corner
    input x, y, u, v arrays
    return coorisponding array in poolar coordinates system
    '''
    x, y, u, v = np.array(x1) - center[0], np.array(y1) - center[1], np.array(u1), np.array(v1)

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    v_r = np.divide(x*u + y*v, r, out = np.zeros_like(r), where = r!= 0)
    v_theta = np.divide(x*v - y*u, r, out = np.zeros_like(r), where = r!= 0)

    return r, theta, v_r, v_theta

def find_minimum_elemt_location(arry: list):
    arr = np.array(arry)
    if len(arr.shape) == 1:
        min_indices = np.argmin(arr)
    else:
        min_indices = np.unravel_index(np.argmin(arr), arr.shape)
    return min_indices


def find_elements_in_circular_region(arr, R):
    r"""
    Find elements in a 2D array representing distances from the origin
    that are within a circular region with radius R and have values greater
    than or equal to R.

    Parameters:
        arr (numpy.ndarray): The 2D array of distances.
        R (float): The radius of the circular region.

    Returns:
        List of tuples: A list of tuples containing the indices of elements
        in the circular region meeting the criteria.
    """
    # Create a boolean mask for the circular region with radius R
    circular_mask = (arr >= R) & (arr < (R + 1))  # Adjust the range as needed

    # Find the indices of elements within the circular region
    indices_in_circular_region = np.argwhere(circular_mask)

    # Collect all elements in the circular region that meet the criteria
    elements_in_circular_region = []

    for idx in indices_in_circular_region:
        i, j = idx
        distance = arr[i, j]
        if distance >= R:
            elements_in_circular_region.append((i, j))

    return elements_in_circular_region


# #######################
# # Other
# #######################

def edge_detection(imgy, method = 'gaussian', filter_size = 3, threshold1 = 120, threshold2 = 500, is_path = True):
    r'''
    taking an images and detect the edges, very nice.
    '''
    if method == 'gaussian':
        if is_path:
            img1 = plt.imread(imgy)
        else:
            img1 = imgy
        imcopy = np.copy(img1)
        background = scipy.ndimage.gaussian_filter(scipy.ndimage.median_filter(img1, filter_size), filter_size)
        mask = background > filters.threshold_otsu(background)
        imcopy[mask] = 0
        return imcopy
    else:
        if is_path:
            img = cv2.imread(imgy, cv2.IMREAD_GRAYSCALE)
        else:
            img = imgy
        
        edges = cv2.Canny(img, threshold1, threshold2)
        return edges
    

def calc_parameters(sphere_diameter = 9.525/1000, sphere_density = 1134, fluid_density = (1120, 1130), nu : tuple = (1.385e-6, 1.452e-6), to_print = False):
    from scipy.constants import pi, g
    import sympy as sy
    from scipy.optimize import fsolve


    if to_print:
        print('rho_p/rho_f_1 = ', sphere_density/fluid_density[0])
    nu1, nu2 = nu[0], nu[1] #[m^2/s]
    sphere_volume = 4/3*pi*(sphere_diameter/2)**3 #sphere volume [m^3]

    rho_f, rho_s, a, U, Ab, Vb, g1, nu= sy.symbols('rho_f, rho_s, a, U, A_b, V_b, g, nu')

    Ab1 = sy.pi*a**2/4 #the projected surface area
    Re = U*a/nu
    Cd = 0.4+24/Re+6/(1+sy.sqrt(Re))

    Fd = Cd*rho_f*U**2*Ab/2
    F_B = rho_f*Vb*g1
    MG = rho_s*g1*Vb

    the_eq = Fd+F_B-MG
    if to_print:
        print('the symbolic expr', sy.Eq(the_eq+MG, MG))

    the_eq_1 = the_eq.subs([(rho_f, fluid_density[0]), (rho_s, sphere_density),
                        (nu, nu1), (Vb, sphere_volume), (Ab, Ab1),(a, sphere_diameter), (g1, g)])
    equa1 = sy.lambdify(U, the_eq_1, "numpy")
    the_veloci_1 = fsolve(equa1, 1)
    the_veloci_1 = the_veloci_1[0]

    the_eq_2 = the_eq.subs([(rho_f, fluid_density[1]), (rho_s, sphere_density),
                        (nu, nu2), (Vb, sphere_volume), (Ab, Ab1),(a, sphere_diameter), (g1, g)])
    equa2 = sy.lambdify(U, the_eq_2, "numpy")
    the_veloci_2 = fsolve(equa2, 1)
    the_veloci_2 = the_veloci_2[0]

    if equa1(the_veloci_1) <= 1e-4:
        # print('verify that "the_veloci_1"=0 ', equa1(the_veloci_1))
        if to_print:
            print('All good')
    else:
        raise ValueError('the_veloci_1=%f' % the_veloci_1)

    Re1 = the_veloci_1*sphere_diameter/nu1
    Re2 = the_veloci_2*sphere_diameter/nu2

    # print('Re1 = ',Re1, 'Re calcuated with mu of bouncing = ',the_veloci_1*sphere_diameter*fluid_density[0]/bouncing_ball_mu_u, 'Re calcuated with nu = ', the_veloci_1*sphere_diameter/nu1)

    if to_print:
 
        print("U_1 = %f" % the_veloci_1, '[m/s]','\nU_2 = %f' % the_veloci_2,'[m/s]')
        print('Re1 = ', Re1)
        print('Re2 = ',Re2)
    return Re1, Re2



def add_black_strip(abb, position='right', strip_width=100):
    r"""
    FIX UP AND DOWN
    Add a black strip to the specified position of the image.

    Parameters:
        abb (str): The file path of the input image.
        position (str): The position to add the black strip. Options: 'right', 'left', 'bottom', 'up'.
                       Default is 'right'.
        strip_width (int): The width of the black strip in pixels. Default is 100.

    Returns:
        None
    """
    base_name, extension = os.path.splitext(abb)
    new_base_name = base_name + '_1'
    new_filename = os.path.join(os.path.dirname(abb), new_base_name + extension)

    # Load the image
    # imgy = Image.open(abb)
    imgy = cv2.imread(abb)
    img = np.array(imgy)

    # Create a new array with the desired size based on position
    if position == 'right':
        new_width = img.shape[1] + strip_width
        new_height = img.shape[0]
    elif position == 'left':
        new_width = img.shape[1] + strip_width
        new_height = img.shape[0]
    elif position == 'bottom':
        new_width = img.shape[1]
        new_height = img.shape[0] + strip_width
    elif position == 'up':
        new_width = img.shape[1]
        new_height = img.shape[0] + strip_width
    else:
        raise ValueError("Invalid position. Use 'right', 'left', 'bottom', or 'up'.")

    asada = (new_height, new_width, img.shape[2])
    new_img = np.zeros(asada, dtype=np.uint8)

    # Copy the original image data into the new array based on position
    if position == 'right':
        new_img[:img.shape[0], :img.shape[1], :] = img
    elif position == 'left':
        new_img[:img.shape[0], strip_width:, :] = img
    elif position == 'bottom':
        new_img[:img.shape[0], :img.shape[1], :] = img
    elif position == 'up':
        new_img[strip_width:, :img.shape[1], :] = img

    # Create a black strip with the desired size
    black_strip = np.zeros((new_height, strip_width, img.shape[2]), dtype=np.uint8)

    # Set the black strip to be all black
    black_strip[:, :, :] = [0] * img.shape[2]

    # Concatenate the black strip to the right, left, bottom, or top of the original array
    if position == 'right':
        new_img[:, img.shape[1]:, :] = black_strip
    elif position == 'left':
        new_img[:, :strip_width, :] = black_strip
    elif position == 'bottom':
        new_img[img.shape[0]:, :, :] = black_strip
    elif position == 'up':
        new_img[:strip_width, :, :] = black_strip

    # Save the new array as the new image
    save_image(new_filename, new_img)


def click_and_creat_graph():
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            points.append([event.xdata, event.ydata])
            scatter_point = plt.scatter(event.xdata, event.ydata, color='red')
            scatter_points.append(scatter_point)
            plt.draw()

    def on_key(event):
        if event.key == 'backspace' and points:
            # Remove the last point
            last_point = points.pop()
            # Remove the corresponding scatter point
            last_scatter_point = scatter_points.pop()
            last_scatter_point.remove()
            plt.draw()
            print(f"Deleted last point: ({last_point[0]}, {last_point[1]})")

    def print_clicked_points(points):
        print("Clicked Points:")
        for point in points:
            print(f"({point[0]}, {point[1]})")

    # Create an empty list to store clicked points
    points = []
    scatter_points = []

    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter([], [])
    ax.set_title('Click on points')

    # Set the x-axis and y-axis limits
    ax.set_ylim(0, 0.6)
    ax.set_xlim(1110, 1140)
    ax.set_xlabel('density [kg/m^3]')
    ax.set_ylabel('height [m]')

    # Connect the click event handler
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    # Connect the key event handler
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    # Show the plot
    plt.show()

    # Print clicked points when the plot is closed
    print(points)

    # Disconnect the event handlers
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    points = np.array(points)
    # Get the indices that would sort the array based on the second column
    sorted_indices = np.argsort(points[:, 1])

    # Use the sorted indices to rearrange the rows of the array
    sorted_array = points[sorted_indices]
    return sorted_array

def click_and_get_y(x_data, y_data, label : str = 'function'):
    r'''
    insert x and y data andrecive a clickable function
    '''
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            x_clicked = event.xdata
            y_clicked = np.interp(x_clicked, x_data, y_data)
            clicked_points.append([x_clicked, y_clicked])
            scatter_point = plt.scatter(x_clicked, y_clicked, color='red')
            scatter_points.append(scatter_point)
            plt.draw()
            print(f"Clicked on ({x_clicked}, {y_clicked})")

    def on_key(event):
        if event.key == 'backspace' and clicked_points:
            # Remove the last clicked point
            last_clicked_point = clicked_points.pop()
            # Remove the corresponding scatter point
            last_scatter_point = scatter_points.pop()
            last_scatter_point.remove()
            plt.draw()
            print(f"Deleted last clicked point: ({last_clicked_point[0]}, {last_clicked_point[1]})")


    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, label=label)
    ax.set_title('Click on the plot')

    # Create empty lists to store clicked points and scatter points
    clicked_points = []
    scatter_points = []

    # Connect the click event handler
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    # Connect the key event handler
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    # Show the plot
    plt.show()

    # Disconnect the event handlers
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    return clicked_points

# def save_cutted_picture(file_path, range_cut, where_to_save):
#     r'''
#     file_path = r'c:\a\template\a.png'
#     range cut should be -> [a,b,c,d] a, b for y, and c, d for x
#     range_cut 2 ranges first y then x -> slice(220, 800), slice(180, 380)
#     '''
#     create_directory(where_to_save)
#     a, b, c, d = range_cut
#     range_cut = slice(a,b), slice(c,d)
#     im_a = plt.imread(file_path)
#     file_name = os.path.basename(file_path)
#     save_path = os.path.join(where_to_save, 'cutted_'+file_name)
#     save_image(save_path, im_a[range_cut])


#old



    # #######################
    # # OLD cv2 functions. has been replaced with napari
    # #######################


# class pixy:
#     r'''
#     find the pixel distance between the line drawed in the cv2 picture.
#     good for measuring diameter
#     '''

#     def __init__(self, thickness : int = 1):
#         self.click_count = 0
#         self.img = None
#         self.location = []
#         self.curPth = None
#         self.keep_same = False
#         self.title = 'Select Diameter'
#         self.thickness = thickness
#         self.zoom_factor = 1.1
#         self.current_zoom = 1
#         self.mouse_wheel = False
#         self.is_dragging = False
#         self.start_x = -1
#         self.start_y = -1
#         self.img_state = None
#         self.absolut_dx_dy = [0, 0]
   

#     def mouse_callback(self, event, x, y, flags, param):
#         if event == cv2.EVENT_RBUTTONDOWN:
#             self.start_x, self.start_y = x, y
#             self.is_dragging = True

#         elif event == cv2.EVENT_RBUTTONUP:
#             self.is_dragging = False

#         elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
#             # Shift the displayed image based on mouse movement
#             dx, dy = x - self.start_x, y - self.start_y
#             self.start_x, self.start_y = x, y
#             self.absolut_dx_dy = [self.absolut_dx_dy[0]+dx, self.absolut_dx_dy[1]+dy]
#             self.img = np.roll(self.img, dx, axis=1)
#             self.img = np.roll(self.img, dy, axis=0)

    
#         if event == cv2.EVENT_MOUSEWHEEL:
#             self.mouse_wheel = True
#             if flags > 0:  # Zoom in when scrolling up
#                 self.current_zoom *= self.zoom_factor
#             else:  # Zoom out when scrolling down
#                 self.current_zoom /= self.zoom_factor
  
#         self.img_state = np.copy(self.img)
        
#         if event == cv2.EVENT_LBUTTONDOWN:
#             self.click_count += 1
            
#             if self.click_count == 1:
#                 self.location.append([x,y])
#                 height, width, channels = self.img.shape
#                 cv2.line(self.img, (0, y), (width,y), (0,255,0), thickness = self.thickness)
#                 cv2.line(self.img, (x, 0), (x, height), (0,255,0), thickness = self.thickness)
#                 cv2.imshow(self.title, self.img) 

#             if self.click_count == 2:
#                 height, width, channels = self.img.shape

#                 if (self.keep_same == 'x' or self.keep_same == 'y'):
#                     if self.keep_same =='y':
#                         self.location.append([x, self.location[0][1]])
#                         cv2.line(self.img, (0, self.location[0][1]), (width,self.location[0][1]), (0,255,0), thickness = self.thickness)
#                         cv2.line(self.img, (x, 0), (x, height), (0,255,0), thickness = self.thickness)
#                         cv2.imshow(self.title, self.img)
#                     if self.keep_same =='x':
#                         self.location.append([self.location[0][0], y])
#                         cv2.line(self.img, (0, y), (width,y), (0,255,0), thickness = self.thickness)
#                         cv2.line(self.img, (self.location[0][0], 0), (self.location[0][0], height), (0,255,0), thickness = self.thickness)
#                         cv2.imshow(self.title, self.img)
#                 else:
#                     cv2.line(self.img, (0, y), (width,y), (0,255,0), 2)
#                     cv2.line(self.img, (x, 0), (x, height), (0,255,0), 2)
#                     cv2.imshow(self.title, self.img) 
#                     self.location.append([x,y])

#     def update_line(self):
#         # self.img.fill(0)
#         self.img = cv2.imread(self.curPth)
#         self.img = cv2.resize(cv2.imread(self.curPth), None, fx=self.current_zoom, fy=self.current_zoom)
#         self.img = np.roll(self.img, self.absolut_dx_dy[0], axis=1)
#         self.img = np.roll(self.img, self.absolut_dx_dy[1], axis=0)
#         # print(self.dy, self.dx)

#         # self.img = np.copy(self.img_state)
#         height, width, channels = self.img.shape


#         if self.click_count == 1:
#             cv2.line(self.img, (0, self.location[self.click_count-1][1]), (width, self.location[self.click_count-1][1]), (0,255,0), thickness = self.thickness)
#             cv2.line(self.img, (self.location[self.click_count-1][0], 0), (self.location[self.click_count-1][0], height), (0,255,0), thickness = self.thickness)
#             cv2.imshow(self.title, self.img) 

#         if self.click_count == 2:
#             cv2.line(self.img, (0, self.location[self.click_count-2][1]), (width, self.location[self.click_count-2][1]), (0,255,0), thickness = self.thickness)
#             cv2.line(self.img, (self.location[self.click_count-2][0], 0), (self.location[self.click_count-2][0], height), (0,255,0), thickness = self.thickness)
#             cv2.line(self.img, (0, self.location[self.click_count-1][1]), (width, self.location[self.click_count-1][1]), (0,255,0), thickness = self.thickness)
#             cv2.line(self.img, (self.location[self.click_count-1][0], 0), (self.location[self.click_count-1][0], height), (0,255,0), thickness = self.thickness)
#             cv2.imshow(self.title, self.img) 

   
    
#     def manual_pixel(self, curPth : str, keep_same = False, title = 'Select Diameter'):
#         self.keep_same = keep_same
#         self.title = title

#         print('you may use "w" "s" "a" "d" keys to change first point location.\n or you may click "r" to reset selction')
#         self.curPth = curPth
#         self.img = cv2.imread(curPth)

#         # Create a window and set the mouse callback function
#         cv2.namedWindow(title)
#         cv2.setMouseCallback(title, self.mouse_callback)

#         # Display the image and wait for a mouse event
#         while self.click_count<3:
#             if self.mouse_wheel:
#                 self.img = cv2.resize(cv2.imread(curPth), None, fx=self.current_zoom, fy=self.current_zoom)
#                 self.mouse_wheel = False

#             strXY = 'zoom = ' + str(self.current_zoom)
#             (text_width, text_height), baseline = cv2.getTextSize(strXY, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
#             x = 10 
#             y = text_height + 10
#             cv2.putText(self.img, strXY, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.8, get_bgr('w'), 2)
#             cv2.imshow(title, self.img)

#             if self.click_count == 2:
#                 bias = 20
#                 bbb = np.array(self.location)
#                 distancey  = np.linalg.norm(bbb[0]-bbb[-1])
#                 distancey = distancey/self.current_zoom
#                 if distancey.is_integer():
#                     distancey_str = f'{round(distancey)}'
#                 else:
#                     distancey_str =  f'{distancey:.3f}'

#                 strXY = 'distance : ' + distancey_str + ' [pixel]'
#                 cv2.putText(self.img, strXY, (self.location[self.click_count-1][0] + bias, self.location[self.click_count-1][1] - bias), cv2.FONT_HERSHEY_DUPLEX, 0.8, get_bgr('w'), 2)

#             key = cv2.waitKey(33)
#             if key == 27:  # Press 'Esc' to exit
#                 break
#             elif key == ord('w') or key == ord('W'):  # Press 'w' or up arrrow key for up
#                 self.location[self.click_count-1][1] = self.location[self.click_count-1][1] - 1
#                 self.update_line()
#             elif key == ord('s') or key == ord('S'):  # Press 's' or down arrrow key for down
#                 self.location[self.click_count-1][1] = self.location[self.click_count-1][1] + 1
#                 self.update_line()
#             elif key == ord('a') or key == ord('A'):  # Press 'a' or left arrow key for left
#                 self.location[self.click_count-1][0] = self.location[self.click_count-1][0] - 1
#                 self.update_line()
#             elif key == ord('d') or key == ord('D'):  # Press 'd' or right arrow key for right
#                 self.location[self.click_count-1][0] = self.location[self.click_count-1][0] + 1
#                 self.update_line()
#             elif key == ord('r') or key == ord('R'):  # Press 'r' to reset location
#                 self.click_count = 0
#                 self.location = []
#                 self.absolut_dx_dy=[0,0]
#                 self.current_zoom = 1
#                 # self.img.fill(0)
#                 self.img = cv2.imread(self.curPth)
#             elif key == 13:  # Press 'Enter' to confirm and exit
#                 if self.click_count == 2:
#                     break

#         # Close the window
#         cv2.destroyAllWindows()

#         b = np.array(self.location)
#         distance  = np.linalg.norm(b[0]-b[-1])/self.current_zoom

#         cv2.waitKey(33)
#         cv2.destroyAllWindows()
#         return distance



# def get_bgr(color_char):
#     # Define the BGR values for blue, green, red, and yellow
#     colors = {
#         'black': (0, 0, 0),     # black
#         'w': (255, 255, 255),         # white
#         'b': (255, 0, 0),       # Blue
#         'g': (0, 255, 0),       # Green
#         'r': (0, 0, 255),       # Red
#         'y': (0, 255, 255),     # Yellow
#         'cyan': (255, 255, 0),  # Cyan
#         'c': (255, 255, 0),     # Cyan
#     }
#     return colors.get(color_char.lower(), (0, 0, 0))  # Default to black if color_char is not recognized



# def select_the_object_cv(im_a_path : str, title : str = 'Select the Object for future analysis'):
#     first_image = cv2.imread(im_a_path)
#     x111, y111, w111, h111 = cv2.selectROI(title, first_image)
#     cv2.destroyAllWindows()
#     # do not delete, important
#     if w111%2 == 1:
#         w111 = w111 - 1
#     if h111%2 == 1:
#         h111 = h111 - 1
#     return x111, y111, w111, h111

# def display_images_as_video(image_folder : str | list, waiting_time : float = 0, scatter : Optional[list] = None, cut_pic = None, origin : str = 'top-left', figsize=(10,10)):
#     if type(image_folder) is str:
#         images = find_pictures(image_folder, True)
#     else:
#         images = image_folder
    
#     for ind, image in enumerate(images):
#         plt.figure(figsize=figsize)
#         print(os.path.basename(image))
#         IPython.display.clear_output(wait = True)
#         img = cv2.imread(image, cv2.COLOR_BGR2GRAY)

#         if not cut_pic is None:
#             if type(cut_pic) is tuple:
#                 img = img[cut_pic]
#             else:
#                 img = img[cut_pic[ind]]
#         if origin == 'bottom-left':
#             imgy = np.copy(img)[::-1]
#             plt.imshow(imgy, cmap='gray', origin='lower') # make sure the origin is left bottom corner
#         else:
#             plt.imshow(img)
#         if not scatter is None:
#             plt.scatter(scatter[ind][0], scatter[ind][1], c ='red') 
#         plt.show()

#         if waiting_time > 0:
#             time.sleep(waiting_time)
#     print('Done!')


# #######################
# # made for the new "select_circle" function
# #######################

# circle_selected = False
# def circle_selected_updater(state):
#     global circle_selected
#     circle_selected = state
# #######################


# def select_circle(image_path:str, title = 'Select Cirlcle', colored : bool = True):
#     r'''
#     select a circle using the mouse, for fine tuning please use the "a s d w + -" keys.
#     return center coordinates and diameter of the circle
#     '''
#     global circle_selected
#     def mouse_callback(event, x, y, flags, param):
#         nonlocal center, radius, drawing, orig
#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing = True
#             center = [x, y]

#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing = False
#             if radius == 0:
#                 radius = max(abs(center[0] - x), abs(center[1] - y))
#             circle_selected_updater(True)

#     drawing = False
#     circle_selected = False
#     center = [0, 0]
#     radius = 0
#     bias = 5 # pixel

#     image = cv2.imread(image_path)

#     #colored picture
#     if colored:
#         image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
#     orig = image.copy()
#     cv2.namedWindow(title)
#     cv2.setMouseCallback(title, mouse_callback)

#     while True:
#         image = orig.copy()
#         strXY = str(center[0]) + ', ' + str(center[1]) + ', Radius: ' + str(radius) +' [pixel]'
#         cv2.putText(image, strXY, (center[0] + radius + bias, center[1] - radius - bias), cv2.FONT_HERSHEY_DUPLEX, 0.8, get_bgr('w'), 2)
        

#         draw_circle(image, center, radius, drawing, circle_selected, title)
#         key = cv2.waitKey(33)

#         if key == 27:  # Press 'Esc' to exit
#             break
#         elif key == ord('+'):  # Press '+' to increase the radius
#             radius += 1
#         elif key == ord('-'):  # Press '-' to decrease the radius
#             radius = max(radius - 5, 0)
#         elif key == ord('w') or key == ord('W'):  # Press 'w' or up arrrow key for up
#             center[1] = center[1] - 1
#         elif key == ord('s') or key == ord('S'):  # Press 's' or down arrrow key for down
#             center[1] = center[1] + 1
#         elif key == ord('a') or key == ord('A'):  # Press 'a' or left arrow key for left
#             center[0] = center[0] - 1
#         elif key == ord('d') or key == ord('D'):  # Press 'd' or right arrow key for right
#             center[0] = center[0] + 1
#         elif key == 13:  # Press 'Enter' to confirm and exit
#             break

#     cv2.destroyAllWindows()
#     circle_selected_updater(False)
#     return center, radius*2




# def display_photos_as_video_cv(path : str, frame_rate : int = 120, scatter = None, title = 'shows photos in changing frame rate', return_img_number : bool = False):
#     photos_path = find_pictures(path, True)
#     the_ind = 0
#     def mouse_callback(event, x, y, flags, param):
#         nonlocal frame_rate
#         if event == cv2.EVENT_MOUSEWHEEL:
#             if flags > 0:  # Zoom in when scrolling up
#                 frame_rate = frame_rate + 1
#             else:  # Zoom out when scrolling down
#                 frame_rate = frame_rate - 2

#         nonlocal photo_path, ind, the_ind
#         if event == cv2.EVENT_LBUTTONDOWN:
#             print(os.path.basename(photo_path))
#             the_ind = ind

#     for ind, photo_path in enumerate(photos_path):
#         if keyboard.is_pressed("right arrow"):
#             time.sleep(0.3)
#             break
#         if the_ind>0 and return_img_number:
#             cv2.destroyAllWindows()
#             return the_ind

#         frame_rate = max(1, frame_rate)
#         image = cv2.imread(photo_path)

#         #scatter
#         if scatter is not None:
#             cv2.circle(image, scatter[ind], 5, (0, 0, 255), -1)

#         #Frame rate
#         strXY = 'frame rate = ' + str(frame_rate)
#         (text_width, text_height), baseline = cv2.getTextSize(strXY, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
#         x = 10 
#         y = text_height + 10
#         cv2.putText(image, strXY, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.8, get_bgr('w'), 2)
        
#         #File name
#         (text_width, text_height), baseline = cv2.getTextSize(os.path.basename(photo_path), cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
#         x = image.shape[1] - text_width - 10
#         y = text_height + 10
#         cv2.putText(image, os.path.basename(photo_path), (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, get_bgr('w'), 2)

#         cv2.imshow(title, image)
#         cv2.setMouseCallback(title, mouse_callback)
#         cv2.waitKey(round(1000 / frame_rate))  # Delay to achieve the desired frame rate

#     cv2.destroyAllWindows()
#     if return_img_number:
#         return the_ind




# def draw_circle(image, center, radius, drawing, circle_selected, title = 'Select Cirlcle', color = 'y'):
#     '''
#     helping function of select_circle
#     '''
#     temp_image = image.copy()

#     if drawing or circle_selected:
#         cv2.circle(temp_image, center, max(radius, 1), get_bgr(color), 2)

#     cv2.imshow(title, temp_image)

    # #######################
    # # End of OLD cv2 functions. has been replaced with napari
    # #######################

    # #######################
    # # End of OLD Sphere_diameter_calculator
    # #######################


# def Circle_Detect(imgy, min_radius = 30, max_radius = 44, param1=30, param2=50, plot = False, is_path = True):
#     r'''
#     detect circles in picture. not reliable, better to avoid this function. used to calculate diameter. NOT RECOMMENDED
#     '''
#     if is_path:
#         # Read image.
#         img = cv2.imread(imgy, cv2.IMREAD_COLOR)
#         # Convert to grayscale.
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         # Read image.
#         img = imgy
#         # Convert to grayscale.
#         gray = imgy

    
#     # Blur using 3 * 3 kernel.
#     gray_blurred = cv2.blur(gray, (3, 3))
    
#     # Apply Hough transform on the blurred image.
#     detected_circles = cv2.HoughCircles(gray_blurred, 
#                     cv2.HOUGH_GRADIENT, 1, 20, param1 = param1,
#                 param2 = param2, minRadius = min_radius, maxRadius = max_radius)
    
#     # Draw circles that are detected.
#     if detected_circles is not None:
    
#         # Convert the circle parameters a, b and r to integers.
#         detected_circles = np.uint16(np.around(detected_circles))
#         if plot:
    
#             for pt in detected_circles[0, :]:
#                 a, b, r = pt[0], pt[1], pt[2]
        
#                 # Draw the circumference of the circle.
#                 cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        
#                 # Draw a small circle (of radius 1) to show the center.
#                 cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
#                 cv2.imshow("Detected Circle", img)
#                 cv2.waitKey(33)
    
#         return detected_circles[0]
#     else:
#         return []



# def most_right_and_left_coordinares(pathyhy : str, show : bool = False, threshold1 : int = 200, threshold2 : int = 400, printy : bool = False, rotate_deg : int = 0):
#     r'''
#     uses nost right and most left coordinates
#     finds diameter of a sphere automaticly, working good on matplotlib circles,
#     need to check on real sphere images from the lab
#     '''
#     raw_image = rotate_image(read_image(pathyhy, False), deg = rotate_deg)
#     if show:
#         cv2.imshow('Original Image', raw_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
    
#     #mort
#     gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
#     bright_filtered = bright_objects(gray_image)


#     # bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
#     bilateral_filtered_image = cv2.bilateralFilter(bright_filtered, 5, 175, 175)

#     if show:
#         cv2.imshow('Bilateral', bilateral_filtered_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

    
#     edge_detected_image = cv2.Canny(bilateral_filtered_image, threshold1=threshold1, threshold2=threshold2)

#     if show:
#         cv2.imshow('Edge', edge_detected_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


#     contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     contour_list = []
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)
#         area = cv2.contourArea(contour)
#         if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
#             contour_list.append(contour)
    
#     if show:
#         cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
#         cv2.imshow('Objects Detected',raw_image)
#         cv2.waitKey(33)
#         cv2.destroyAllWindows()
        
#     xis = []
#     yis = []
#     for contour in contour_list:
#         for el in contour:
#             col = el[:, 0][0]
#             xis.append(col)
#             yis.append(el[:, 1][0])
  
#     max_index = xis.index(max(xis))
#     min_index = xis.index(min(xis))
#     most_right = (xis[max_index], yis[max_index])
#     most_left = (xis[min_index], yis[min_index])
#     if printy:
#         print(os.path.basename(pathyhy), np.linalg.norm(np.array(most_right)-np.array(most_left)))
#     return most_right, most_left



    # #######################
    # # End of OLD Sphere_diameter_calculator
    # #######################
