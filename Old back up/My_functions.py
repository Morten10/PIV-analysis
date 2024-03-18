import cv2
import os
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage import filters
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import correlate2d, fftconvolve
import imageio


def create_directory(location):
    '''
    creating folder if not exists. otherwise does nothing.
    '''
    if not os.path.isdir(location):
        os.makedirs(location)


def Circle_Detect(imgy, min_radius = 30, max_radius = 44, param1=30, param2=50, plot = False, is_path = True):
    '''
    detect circles in picture. not reliable, better to avoid this function.
    '''
    if is_path:
        # Read image.
        img = cv2.imread(imgy, cv2.IMREAD_COLOR)
        # Convert to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Read image.
        img = imgy
        # Convert to grayscale.
        gray = imgy

    
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = param1,
                param2 = param2, minRadius = min_radius, maxRadius = max_radius)
    
    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        if plot:
    
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
        
                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                cv2.imshow("Detected Circle", img)
                cv2.waitKey(33)
    
        return detected_circles[0]
    else:
        return []

def find_diameter(pathyhy, show=False, threshold1 = 200, threshold2 = 400, printy = False):
    '''
    finds diameter of a sphere automaticly, working good on matplotlib circles,
    need to check on real sphere images from the lab
    '''
    raw_image = cv2.imread(pathyhy)
    if show:
        cv2.imshow('Original Image', raw_image)
        cv2.waitKey(33)
    
    #mort
    gray_image =  cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
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
        cv2.waitKey(33)
        cv2.destroyAllWindows()


    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
            contour_list.append(contour)
    
    if show:
        cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
        cv2.imshow('Objects Detected',raw_image)
        cv2.waitKey(33)
        cv2.destroyAllWindows()
        
    xis = []
    yis = []
    for contour in contour_list:
        for el in contour:
            col = el[:, 0][0]
            xis.append(col)
            yis.append(el[:, 1][0])
  
    max_index = xis.index(max(xis))
    min_index = xis.index(min(xis))
    most_right = (xis[max_index], yis[max_index])
    most_left = (xis[min_index], yis[min_index])
    if printy:
        print(os.path.basename(pathyhy), np.linalg.norm(np.array(most_right)-np.array(most_left)))
    return most_right, most_left

class pixy:
    '''
    find the pixel distance between the line drawed in the cv2 picture.
    good for measuring diameter
    '''

    def __init__(self):
        self.click_count = 0
        self.img = None
        self.location = []
        self.curPth = None
        self.keep_same = False
        self.title = 'Select Diameter'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_count += 1
            
            if self.click_count == 1:
                self.location.append([x,y])
                height, width, channels = self.img.shape
                cv2.line(self.img, (0, y), (width,y), (0,255,0), 2)
                cv2.line(self.img, (x, 0), (x, height), (0,255,0), 2)
                cv2.imshow(self.title, self.img) 

            if self.click_count == 2:
                height, width, channels = self.img.shape

                if (self.keep_same == 'x' or self.keep_same == 'y'):
                    if self.keep_same =='y':
                        self.location.append([x, self.location[0][1]])
                        cv2.line(self.img, (0, self.location[0][1]), (width,self.location[0][1]), (0,255,0), 2)
                        cv2.line(self.img, (x, 0), (x, height), (0,255,0), 2)
                        cv2.imshow(self.title, self.img)
                    if self.keep_same =='x':
                        self.location.append([self.location[0][0], y])
                        cv2.line(self.img, (0, y), (width,y), (0,255,0), 2)
                        cv2.line(self.img, (self.location[0][0], 0), (self.location[0][0], height), (0,255,0), 2)
                        cv2.imshow(self.title, self.img)
                else:
                    cv2.line(self.img, (0, y), (width,y), (0,255,0), 2)
                    cv2.line(self.img, (x, 0), (x, height), (0,255,0), 2)
                    cv2.imshow(self.title, self.img) 
                    self.location.append([x,y])

    def update_line(self):
        self.img.fill(0)
        self.img = cv2.imread(self.curPth)
        height, width, channels = self.img.shape

        if self.click_count == 1:
            cv2.line(self.img, (0, self.location[self.click_count-1][1]), (width, self.location[self.click_count-1][1]), (0,255,0), 2)
            cv2.line(self.img, (self.location[self.click_count-1][0], 0), (self.location[self.click_count-1][0], height), (0,255,0), 2)
            cv2.imshow(self.title, self.img) 

        if self.click_count == 2:
            cv2.line(self.img, (0, self.location[self.click_count-2][1]), (width, self.location[self.click_count-2][1]), (0,255,0), 2)
            cv2.line(self.img, (self.location[self.click_count-2][0], 0), (self.location[self.click_count-2][0], height), (0,255,0), 2)
            cv2.line(self.img, (0, self.location[self.click_count-1][1]), (width, self.location[self.click_count-1][1]), (0,255,0), 2)
            cv2.line(self.img, (self.location[self.click_count-1][0], 0), (self.location[self.click_count-1][0], height), (0,255,0), 2)
            cv2.imshow(self.title, self.img) 


    def manual_pixel(self, curPth, keep_same = False, title = 'Select Diameter'):
        self.keep_same = keep_same
        self.title = title

        print('you may use "w" "s" "a" "d" keys to change first point location.\n or you may click "r" to reset selction')
        self.curPth = curPth
        self.img = cv2.imread(curPth)

        # Create a window and set the mouse callback function
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, self.mouse_callback)

        # Display the image and wait for a mouse event
        while self.click_count<3:
            cv2.imshow(title, self.img)

            key = cv2.waitKey(33)
            if key == 27:  # Press 'Esc' to exit
                break
            elif key == 119 or key == 38 or key == 87:  # Press 'w' or up arrrow key for up
                self.location[self.click_count-1][1] = self.location[self.click_count-1][1] - 1
                self.update_line()
            elif key == 115 or key == 40 or key == 83:  # Press 's' or down arrrow key for down
                self.location[self.click_count-1][1] = self.location[self.click_count-1][1] + 1
                self.update_line()
            elif key == 97 or key == 37 or key == 65:  # Press 'a' or left arrow key for left
                self.location[self.click_count-1][0] = self.location[self.click_count-1][0] - 1
                self.update_line()
            elif key == 100 or key == 39 or key == 68:  # Press 'd' or right arrow key for right
                self.location[self.click_count-1][0] = self.location[self.click_count-1][0] + 1
                self.update_line()
            elif key == 114 or key == 82:  # Press 'r' to reset location
                self.click_count = 0
                self.location = []

                self.img.fill(0)
                self.img = cv2.imread(self.curPth)
            elif key == 13:  # Press 'Enter' to confirm and exit
                break

        # Close the window
        cv2.destroyAllWindows()

        b = np.array(self.location)
        distance  = np.linalg.norm(b[0]-b[-1])

        cv2.waitKey(33)
        cv2.destroyAllWindows()
        return distance


# class pixy:
#     '''
#     find the pixel distance between the line drawed in the cv2 picture.
#     good for measuring diameter
#     '''

#     def __init__(self):
#         self.click_count = 0
#         self.img = None
#         self.line = None
#         self.location = []
#         self.curPth = None
#         self.keep_same = False
#         self.title = 'Select Diameter'

#     def mouse_callback(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if self.click_count == 0:
#                 height, width, channels = self.img.shape
#                 self.line = cv2.line(self.img, (0, y), (width,y), (0,255,0), 2)
#                 cv2.imshow(self.title, self.img) 
#                 self.line = cv2.line(self.img, (x, 0), (x, height), (0,255,0), 2)

#             self.click_count += 1
#             if (self.keep_same == 'x' or self.keep_same == 'y') and self.click_count == 2:
#                 if self.keep_same =='y':
#                     self.location.append([x, self.location[0][1]])
#                 if self.keep_same =='x':
#                     self.location.append([self.location[0][0], y])
#             else:
#                 self.location.append([x,y])

#     def update_line(self):
#         self.img.fill(0)
#         self.img = cv2.imread(self.curPth)
#         height, width, channels = self.img.shape

#         self.line = cv2.line(self.img, (0, self.location[0][1]), (width, self.location[0][1]), (0,255,0), 2)
#         cv2.imshow(self.title, self.img) 
#         self.line = cv2.line(self.img, (self.location[0][0], 0), (self.location[0][0], height), (0,255,0), 2)

#     def manual_pixel(self, curPth, keep_same = False, title = 'Select Diameter'):
#         self.keep_same = keep_same
#         self.title = title

#         print('you may use "w" "s" "a" "d" keys to change first point location.\n or you may click "r" to reset selction')
#         self.curPth = curPth
#         self.img = cv2.imread(curPth)

#         # Create a window and set the mouse callback function
#         cv2.namedWindow(title)
#         cv2.setMouseCallback(title, self.mouse_callback)

#         # Display the image and wait for a mouse event
#         while self.click_count<2:
#             cv2.imshow(title, self.img)

#             key = cv2.waitKey(33)
#             if key == 27:  # Press 'Esc' to exit
#                 break
#             elif key == 119 or key == 38:  # Press 'w' or up arrrow key for up
#                 self.location[0][1] = self.location[0][1] - 1
#                 self.update_line()
#             elif key == 115 or key == 40:  # Press 's' or down arrrow key for down
#                 self.location[0][1] = self.location[0][1] + 1
#                 self.update_line()
#             elif key == 97 or key == 37:  # Press 'a' or left arrow key for left
#                 self.location[0][0] = self.location[0][0] - 1
#                 self.update_line()
#             elif key == 100 or key == 39:  # Press 'd' or right arrow key for right
#                 self.location[0][0] = self.location[0][0] + 1
#                 self.update_line()
#             elif key == 114:  # Press 'r' to reset location
#                 self.click_count=0
#                 self.location = []

#                 self.img.fill(0)
#                 self.img = cv2.imread(self.curPth)
#             elif key == 13:  # Press 'Enter' to confirm and exit
#                 break

#         # Close the window
#         cv2.destroyAllWindows()

#         b = np.array(self.location)
#         distance  = np.linalg.norm(b[0]-b[-1])

#         cv2.waitKey(33)
#         cv2.destroyAllWindows()
#         return distance

def return_file_end_with(path, extenstion = 'txt'):
    txt_files = [f for f in os.listdir(path) if f.endswith('.' + extenstion)]
    return txt_files

def txt_to_list(path):
    '''
    openpiv saves data in txt. this function extract x,y,u,v
    recive Path of txt file
    return paramers as list
    '''
    with open(path, 'r') as f:
        data = f.read().split('\n')

    columns = [[] for _ in range(len(data[0].split('\t')))]

    for row in data:
        if row:
            for i, col in enumerate(row.split('\t')):
                columns[i].append(col)

    daty = []
    for i in range(4):
        holder = columns[i][1:]
        daty.append([])
        for el in holder:
            if el[0] == ' ':
                daty[i].append(float(el[1:]))
            else:
                daty[i].append(float(el))
    
    return np.array(daty[0]), np.array(daty[1]), np.array(daty[2]), np.array(daty[3])



def Img_manipulation(imgy, method = 'gaussian', filter_size = 3, threshold1 = 120, threshold2 = 500, is_path = True):
    '''
    taking an images and detect the edges, very nice.
    '''
    if method == 'gaussian':
        if is_path:
            img1 = plt.imread(imgy)
        else:
            img1 = imgy
        imcopy = np.copy(img1)
        background = gaussian_filter(median_filter(img1, filter_size), filter_size)
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
    

def find_pictures_subfolder(directory, extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff','.tif', '.ico', '.jfif', '.webp']):
    '''
    return a list of pictures in the folder and the subfolders
    '''
    pictures = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                pictures.append(os.path.basename(file))
    return sorted(pictures)

def find_pictures(directory, extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.ico', '.jfif', '.webp']):
    '''
    return a list of pictures in the current folder only
    '''
    pictures = []
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in extensions):
            pictures.append(os.path.basename(file))
    return sorted(pictures)

def save_cutted_picture(file_path, range_cut, where_to_save):
    '''
    file_path = r'c:\a\template\a.png'
    range cut should be -> [a,b,c,d] a, b for y, and c, d for x
    range_cut 2 ranges first y then x -> slice(220, 800), slice(180, 380)
    '''
    create_directory(where_to_save)
    a, b, c, d = range_cut
    range_cut = slice(a,b), slice(c,d)
    im_a = plt.imread(file_path)
    file_name = os.path.basename(file_path)
    save_path = os.path.join(where_to_save, 'cutted_'+file_name)
    cv2.imwrite(save_path, im_a[range_cut])


def track_object_series(image_dir, show = False, cutted = None, adjust_coordinate = True, title = 'select object'):
    '''
    recive folder directory
    return coordinates of the object, in cutted is known changing the coordinates acoordinaly,
    cutted should be in the form of cutted= [220, 800, 180, 380] #[y1, y2, x1, x2], just like slicing image in numpy
    show is to plot if wanted
    adjust_coordinate if True, changes the coordinates accordint to the sphere selction
    '''
    # Read the first image
    image_files = find_pictures(image_dir)
    first_image_path = os.path.join(image_dir, image_files[0])

    # Create a separate window for object selection
    first_image = cv2.imread(first_image_path)
    roi = cv2.selectROI(title, first_image)
    cv2.destroyAllWindows()
    x11, y11 , w11, h11 = roi
    x_roi, y_roi = x11 + w11/2, y11 + h11/2

    if adjust_coordinate:
        real_coord, _ = select_circle(first_image_path)
        x_real, y_real = real_coord[0], real_coord[1]
        bias_x, bias_y = x_roi - x_real, y_roi - y_real

    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    
    im_cropped = np.copy(first_image[int(y11) : int(y11+h11), int(x11) : int(x11+w11)])

    tracked_positions = []

    # Loop through the rest of the images
    for image_file in image_files[1:]:
        image_path = os.path.join(image_dir, image_file)
        x1, y1 = track_object(first_image_path, image_path, show = show, cutted = cutted, adjust_coordinate = True, sequence = [im_cropped, [bias_x, bias_y]])[0]
        tracked_positions.append((x1, y1))
    if not cutted is None:
        print('this coordinates are of the original photos, before cutting them')
    return tracked_positions

# #back up track_object_series
# def track_object_series_s(image_dir, show = False, cutted = None, adjust_coordinate = True):
#     '''
#     recive folder directory
#     return coordinates of the object, in cutted is known changing the coordinates acoordinaly,
#     cutted should be in the form of cutted= [220, 800, 180, 380] #[y1, y2, x1, x2], just like slicing image in numpy
#     show is to plot if wanted
#     adjust_coordinate if True, changes the coordinates accordint to the sphere selction
#     '''
#     # Read the first image
#     image_files = find_pictures(image_dir)
#     first_image_path = os.path.join(image_dir, image_files[0])

#     # Create a separate window for object selection
#     first_image = cv2.imread(first_image_path)
#     roi = cv2.selectROI('Click "Enter" to select', first_image)
#     cv2.destroyAllWindows()
#     x11, y11 , w11, h11 = roi

#     if adjust_coordinate:
#         real_coord, _ = select_circle(first_image_path)
#         x_real, y_real = real_coord[0], real_coord[1]

#     first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    
#     first_image = Image.open(first_image_path).convert('L')
#     first_image = np.array(first_image)
#     im_cropped = np.copy(first_image[int(y11) : int(y11+h11), int(x11) : int(x11+w11)])

#     if show:
#         plt.title('selected Image')
#         plt.imshow(im_cropped)
#         plt.show()

#     # List to store the tracked object positions
#     tracked_positions = [(x11+w11/2, y11+h11/2)]

#     # Loop through the rest of the images
#     for image_file in image_files[1:]:
#         image_path = os.path.join(image_dir, image_file)
#         imageyyyy = Image.open(image_path).convert('L')
#         x1,y1 = correlate2d_for_me(imageyyyy, im_cropped)
#         tracked_positions.append((x1-w11/2, y1-h11/2))

#         if show:
#             print('doesnt take into account "adjust_coordinate"')
#             plt.scatter(tracked_positions[-1][0],tracked_positions[-1][1])
#             plt.imshow(imageyyyy)
#             plt.show()

#     if adjust_coordinate:
#         bias_x, bias_y = tracked_positions[0][0]-x_real, tracked_positions[0][1]-y_real
#         tracked_positions = [(x - bias_x, y - bias_y) for (x, y) in tracked_positions]
    
#     if not cutted is None:
#         a, b, c, d = cutted
#         tracked_positions = [(x+c, y+a) for (x, y) in tracked_positions]
#         print('This is the coordinates of the ORIGINAL photos.')
    
#     return tracked_positions


def track_object(first_image_path, image_path, show = False, cutted = None, adjust_coordinate = True, sequence = None, title = 'select the object', circle_size = 100):
    '''
    recive:
      first_image_path the image from to selcted rectangle
      image_path the image we search the object
    return:
     coordinates of the object, in cutted is known changing the coordinates acoordinaly,
    cutted should be in the form of cutted= [220, 800, 180, 380] #[y1, y2, x1, x2], just like slicing image in numpy
    show is to plot if wanted
    adjust_coordinate if True, changes the coordinates accordint to the sphere selction
    '''

    # Create a separate window for object selection
    if sequence is None:
        first_image = cv2.imread(first_image_path)
        roi = cv2.selectROI(title, first_image)
        cv2.destroyAllWindows()
        x11, y11 , w11, h11 = roi

        first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    
        first_image = Image.open(first_image_path).convert('L')
        first_image = np.array(first_image)
        im_cropped = np.copy(first_image[int(y11) : int(y11+h11), int(x11) : int(x11+w11)])
        tracked_positions = [(x11+w11/2, y11+h11/2)]

        if adjust_coordinate:
            real_coord, _ = select_circle(first_image_path)
            x_real, y_real = real_coord[0], real_coord[1]
            bias_x, bias_y = tracked_positions[0][0] - x_real, tracked_positions[0][1] - y_real

    else:
        im_cropped = sequence[0]
        bias_x, bias_y = sequence[1]
        w11, h11 = im_cropped.shape
        tracked_positions = []

    if show:
        plt.title('selected Image single')
        plt.imshow(im_cropped)
        plt.show()

    # List to store the tracked object positions

    imageyyyy = Image.open(image_path).convert('L')
    x1, y1 = correlate2d_for_me(imageyyyy, im_cropped)
    tracked_positions.append((x1 - w11/2, y1 - h11/2))

    if not cutted is None:
        a, b, c, d = cutted
        tracked_positions = [(x+c, y+a) for (x, y) in tracked_positions]

    if adjust_coordinate:
        tracked_positions = [(x - bias_x, y - bias_y) for (x, y) in tracked_positions]
    
    if show:
        if adjust_coordinate:
            plt.title('taken into account adjust_coordinate')
        else:
            plt.title('Did not take into account adjust_coordinate')
        if not cutted is None:
            plt.scatter(tracked_positions[-1][0]-c, tracked_positions[-1][1]-a, s = circle_size, c = 'red')
        else:
            plt.scatter(tracked_positions[-1][0],tracked_positions[-1][1], s = circle_size, c = 'red')
        plt.imshow(imageyyyy)
        plt.show()
    return tracked_positions




# # back up track_object
# def track_object_ssssss(first_image_path, image_path, show = False, cutted = None, adjust_coordinate = True):
#     '''
#     recive:
#       first_image_path the image from to selcted rectangle
#       image_path the image we search the object
#     return:
#      coordinates of the object, in cutted is known changing the coordinates acoordinaly,
#     cutted should be in the form of cutted= [220, 800, 180, 380] #[y1, y2, x1, x2], just like slicing image in numpy
#     show is to plot if wanted
#     adjust_coordinate if True, changes the coordinates accordint to the sphere selction
#     '''

#     # Create a separate window for object selection
#     first_image = cv2.imread(first_image_path)
#     roi = cv2.selectROI('Click "Enter" to select', first_image)
#     cv2.destroyAllWindows()
#     x11, y11 , w11, h11 = roi

#     if adjust_coordinate:
#         real_coord, _ = select_circle(first_image_path)
#         x_real, y_real = real_coord[0], real_coord[1]

#     first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    
#     first_image = Image.open(first_image_path).convert('L')
#     first_image = np.array(first_image)
#     im_cropped = np.copy(first_image[int(y11) : int(y11+h11), int(x11) : int(x11+w11)])

#     if show:
#         plt.title('selected Image')
#         plt.imshow(im_cropped)
#         plt.show()

#     # List to store the tracked object positions
#     tracked_positions = [(x11+w11/2, y11+h11/2)]

#     imageyyyy = Image.open(image_path).convert('L')
#     x1,y1 = correlate2d_for_me(imageyyyy, im_cropped)
#     tracked_positions.append((x1-w11/2, y1-h11/2))

#     if not cutted is None:
#         a, b, c, d = cutted
#         tracked_positions = [(x+c, y+a) for (x, y) in tracked_positions]
#         print('This is the coordinates of the ORIGINAL photos.')

#     if adjust_coordinate:
#         bias_x, bias_y = tracked_positions[0][0]-x_real, tracked_positions[0][1]-y_real
#         tracked_positions = [(x - bias_x, y - bias_y) for (x, y) in tracked_positions]
#     if show:
#         if adjust_coordinate:
#             plt.title('taken into account adjust_coordinate')
#         else:
#             plt.title('Did not take into account adjust_coordinate')
#         plt.scatter(tracked_positions[-1][0],tracked_positions[-1][1])
#         plt.imshow(imageyyyy)
#         plt.show()

#     return tracked_positions


def track_object_multi(pattern, image_path, bias = [1, 1] ,show = False, showy = False, cutted = None, adjust_coordinate = True, prior_knowledge = False):
    '''
    recive:
      pattern - the cutted rectangle img, a 2d arrray grayscale
      bias - the pixels change in x and y for the real center of circle.
      image_path the image we search the object

      ###
      CUTTED HASEN'T BEEN TESTED YET
      ####
    return:
     coordinates of the object, in cutted is known changing the coordinates acoordinaly,
    cutted should be in the form of cutted= [220, 800, 180, 380] #[y1, y2, x1, x2], just like slicing image in numpy
    show is to plot if wanted
    adjust_coordinate if True, changes the coordinates accordint to the sphere selction
    '''
    if showy:
        plt.title('selected Image')
        plt.imshow(pattern)
        plt.show()

    # List to store the tracked object positions
    tracked_positions = []

    image = Image.open(image_path).convert('L')
    imageyyyy = np.array(image)
    if prior_knowledge is not False:
        imageyyyy = imageyyyy[prior_knowledge]

    x1, y1 = correlate2d_for_me(imageyyyy, pattern)
    if prior_knowledge is not False:
        y1 = y1 + prior_knowledge[0].start
        x1 = x1 + prior_knowledge[1].start
    
  
    tracked_positions.append((int(x1 - pattern.shape[1]/2), int(y1 - pattern.shape[0]/2)))
    
    if not cutted is None:
        a, b, c, d = cutted
        tracked_positions = [(x+c, y+a) for (x, y) in tracked_positions]
        print('This is the coordinates of the ORIGINAL photos.')

    if adjust_coordinate:
        tracked_positions = [(int(x - bias[0]), int(y - bias[1])) for (x, y) in tracked_positions]
    
    if show:
        if adjust_coordinate:
            plt.title('taken into account adjust_coordinate')
        else:
            plt.title('Did not take into account adjust_coordinate')
        plt.scatter(tracked_positions[-1][0],tracked_positions[-1][1])
        plt.imshow(image)
        plt.show()

    return tracked_positions[0]


def find_the_slice(image_path, object_image, path = False, prior_knowledge = False):
    '''
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
    if prior_knowledge is not False:
        first_image = first_image[prior_knowledge]

    x1, y1 = correlate2d_for_me(first_image, object_image)
    shapey = object_image.shape
    if prior_knowledge is not False:
        y1 = y1 + prior_knowledge[0].start
        x1 = x1 + prior_knowledge[1].start
    return y1-shapey[0], y1, x1-shapey[1], x1


def correlate2d_for_me_test_fftconvolve_not_working(img, template, show = False):
    '''
    make sure both images are 2d - Grayscale images.
    correlation between 2 images
    return:
     the coordinated of the center of the best match
    '''

    img = np.array(img)
    template = np.array(template)
    img = img - img.mean()
    template = template - template.mean()
    
    corr = fftconvolve (img, template, mode = 'same')
    # corr1 = correlate2d (img, template)
    # print(corr, corr1)


    max_coords = np.where(corr == np.max(corr))
    x = max_coords[1][0]
    y = max_coords[0][0]

    if show:
        plt.plot(max_coords[1], max_coords[0],'c*', markersize=5)
        plt.imshow(corr, cmap='hot')
        plt.show()
    return x, y


def correlate2d_for_me(img, template, show = False):
    '''
    make sure both images are 2d - Grayscale images.
    correlation between 2 images
    return:
     the coordinated of the center of the best match
    '''

    img = np.array(img)
    template = np.array(template)
    img = img - img.mean()
    template = template - template.mean()
    
    corr = correlate2d(img, template)

    # max_coords = np.where(corr == np.max(corr))
    # x = max_coords[1][0]
    # y = max_coords[0][0]

    max_coords = np.argmax(corr)
    y, x = np.unravel_index(max_coords, corr.shape)

    if show:
        plt.plot(max_coords[1], max_coords[0],'c*', markersize=5)
        plt.imshow(corr, cmap='hot')
        plt.show()
    return x, y

def select_circle(image_path, title = 'Select Cirlcle'):
    '''
    select a circle using the mouse, for fine tuning please use the "a s d w + -" keys.
    return center coordinates and diameter of the circle
    '''
    def mouse_callback(event, x, y, flags, param):
        nonlocal center, radius, drawing, circle_selected

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            center = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            radius = max(abs(center[0] - x), abs(center[1] - y))
            circle_selected = True

    drawing = False
    circle_selected = False
    center = [0, 0]
    radius = 0

    image = cv2.imread(image_path)
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_callback)

    while True:
        draw_circle(image, center, radius, drawing, circle_selected, title)
        key = cv2.waitKey(33)

        if key == 27:  # Press 'Esc' to exit
            break
        elif key == 43:  # Press '-' to increase the radius
            radius += 1
        elif key == 45:  # Press '+' to decrease the radius
            radius = max(radius - 5, 0)
        elif key == 119 or key == 38 or key == 87:  # Press 'w' or up arrrow key for up
            center[1] = center[1] - 1
        elif key == 115 or key == 40 or key == 83:  # Press 's' or down arrrow key for down
            center[1] = center[1] + 1
        elif key == 97 or key == 37 or key == 65:  # Press 'a' or left arrow key for left
            center[0] = center[0] - 1
        elif key == 100 or key == 39 or key == 68:  # Press 'd' or right arrow key for right
            center[0] = center[0] + 1
        elif key == 13:  # Press 'Enter' to confirm and exit
            break
    cv2.destroyAllWindows()

    if circle_selected:
        return center, radius*2
    else:
        return None

def draw_circle(image, center, radius, drawing, circle_selected, title = 'Select Cirlcle'):
    '''
    helping function of select_circle
    '''
    temp_image = image.copy()

    if drawing or circle_selected:
        cv2.circle(temp_image, center, max(radius, 1), (0, 0, 255), 2)

    cv2.imshow(title, temp_image)

def bright_objects(image, filter_size=7):
    '''
    recive:
    gray image - an image of (x,y,1) dimentions.
    return the bright ocbect, masked.
    Ideal for finding diameter.
    '''
    imcopy = np.copy(image)
    background = gaussian_filter(median_filter(image, filter_size), filter_size)
    mask = background > filters.threshold_otsu(background)
    imcopy[mask] = 254
    return imcopy


def add_black_strip(abb, position='right', strip_width=100):
    """
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
    imgy = Image.open(abb)
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
    imageio.imwrite(new_filename, new_img)


# def add_black_strip(abb):
#     base_name, extension = os.path.splitext(abb)
#     new_base_name = base_name + '_1'
#     new_filename = os.path.join(os.path.dirname(abb), new_base_name + extension)

#     # Load the image
#     imgy = Image.open(abb)
#     img = np.array(imgy)

#     # Create a new array with the desired size
#     asada = (img.shape[0], img.shape[1] + 100, img.shape[2])
#     new_img = np.zeros(asada, dtype=np.uint8)

#     # Copy the original image data into the top left corner of the new array
#     new_img[:img.shape[0], :img.shape[1], :] = img

#     # Determine the number of channels in the original image
#     num_channels = img.shape[2]

#     # Create a black strip with the same height as the original image and 100 pixels width
#     black_strip = np.zeros((img.shape[0], 100, num_channels), dtype=np.uint8)

#     # Set the black strip to be all black
#     black_strip[:, :, :] = [0] * num_channels

#     # Concatenate the black strip to the right side of the original array
#     new_img[:, img.shape[1]:, :] = black_strip

#     # Save the new array as the new image
#     imageio.imwrite(new_filename, new_img)


##
# FOR COMTROL VOLUME
##

def find_closest_values_range_in_array(arr, target):
    arr = np.array(arr)
    diff = np.abs(arr - target)
    closest_index = np.argmin(diff)
    target = arr[closest_index]
 
    start = end = closest_index
    while start > 0 and arr[start - 1] == target:
        start -= 1
    while end < len(arr) - 1 and arr[end + 1] == target:
        end += 1
      
    return start, end

def find_closest_values_in_array(arr, target):
    arr = np.array(arr)
    diff = np.abs(arr - target)
    closest_index = np.argmin(diff)
    target = arr[closest_index]
    return closest_index
