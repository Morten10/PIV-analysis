# import My_functions as Mf
# pic = r'c:\\Users\\Morten\\OneDrive - mail.tau.ac.il\\Thesis\\Python Thesis\\Analysis\\Picture\\alot - PTV'
# sssss = [[650, 349],
#        [650, 349],
#        [650, 350]]

# import napari
# import numpy as np

# def display_images_with_points_napari(directory, points, scaling_factor, pysical : bool = False):
#     r'''
#     pysical is to convert pixel to meter.
#     '''
#     viewer = napari.Viewer(axis_labels=['y', 'x'])
#     image_paths = Mf.find_pictures(directory, True)[:3]
#     images = [Mf.read_image(path) for path in image_paths]
#     images_stack = np.stack(images)
#     if pysical:
#         viewer.add_image(images_stack, scale=[1/scaling_factor, 1/scaling_factor])

#         points = np.array(points)/scaling_factor
#         pointss = Mf.add_index_to_points(points)
#         pointss = Mf.replace_elements(pointss)
#         viewer.add_points(pointss, size=10/scaling_factor, face_color='red')
#         viewer.scale_bar.unit = "m"

#     else:
#         viewer.add_image(images_stack)
#         pointss = Mf.add_index_to_points(points)
#         pointss = Mf.replace_elements(pointss)
#         viewer.add_points(pointss, size=10, face_color='red')
#         viewer.scale_bar.unit = "pixel"


#     viewer.axes.visible = True
#     viewer.scale_bar.visible = True
#     viewer.scale_bar.colored = True
#     # viewer.scale_bar.color = 'violet'

#     napari.run()

# display_images_with_points_napari(pic, sssss, 3538.7297166591284, True)



import My_functions as Mf
import os
parent_folder = os.path.dirname(os.getcwd()) #on notebok only. in .py file its different


pic_folder = os.path.join(os.getcwd(), 'Analysis', 'Picture', 'alot')

# Mf.transform_coordinates_pysical([[150,60], [150,61], [150,62], [150,63]], pic_folder, 10, [10,10])

# Mf.velocity_from_location(sphere_location = [[10, 10], [11, 11], [12, 12], [15, 15]], dt = [1,1,3])

# print(Mf.display_images_with_points_napari(pic_folder))

# print(Mf.calculate_distance_between_points_napari_notebook_orig(Mf.find_pictures(pic_folder)[0]))

# print(Mf.select_area_napari_notebook(Mf.find_pictures(pic_folder)[0]))

# print(Mf.replace_elements([[1,2],[1,2]]))

# Mf.Change_contrast(Mf.select_directory())


# print(Mf.select_circle_napari_advance(pic_folder))
# img_number, real_cor, obj, PTV = Mf.select_circle_napari_advance(pic_folder)
# print(PTV)





# aaaa = r'C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Python Thesis\Analysis\Picture\contrast test'
# Mf.Change_contrast(aaaa)




# image edge detectiopn
                    # # import the necessary packages
                    # import imutils
                    # import cv2
                    # iterations = 5
                    # # load the image, convert it to grayscale, and blur it slightly
                    # path = r"C:\Users\Morten\OneDrive - mail.tau.ac.il\Thesis\Python Thesis\Analysis\Picture\snapshot diameter 9.5252 2023 11 01\C001H001S0040000001.tif"
                    # image = cv2.imread(path)
                    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    # # threshold the image, then perform a series of erosions +
                    # # dilations to remove any small regions of noise
                    # thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
                    # thresh = cv2.erode(thresh, None, iterations=iterations)
                    # thresh = cv2.dilate(thresh, None, iterations=iterations)
                    # # find contours in thresholded image, then grab the largest
                    # # one
                    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # cnts = imutils.grab_contours(cnts)
                    # c = max(cnts, key=cv2.contourArea)

                    # # determine the most extreme points along the contour
                    # extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    # extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # extTop = tuple(c[c[:, :, 1].argmin()][0])
                    # extBot = tuple(c[c[:, :, 1].argmax()][0])

                    # # draw the outline of the object, then draw each of the
                    # # extreme points, where the left-most is red, right-most
                    # # is green, top-most is blue, and bottom-most is teal
                    # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
                    # cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
                    # cv2.circle(image, extRight, 8, (0, 255, 0), -1)
                    # cv2.circle(image, extTop, 8, (255, 0, 0), -1)
                    # cv2.circle(image, extBot, 8, (255, 255, 0), -1)
                    # # show the output image
                    # cv2.imshow("Image", image)
                    # cv2.waitKey(0)











                    # print('here')

                    # # import the necessary packages
                    # # from pyimagesearch.transform import four_point_transform
                    # # from skimage.filters import threshold_local
                    # import numpy as np
                    # import argparse
                    # import cv2
                    # import imutils

                    #                 # # construct the argument parser and parse the arguments
                    #                 # ap = argparse.ArgumentParser()
                    #                 # ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
                    #                 # args = vars(ap.parse_args())
                    #                 # # load the image and compute the ratio of the old height
                    #                 # # to the new height, clone it, and resize it
                    #                 # # image = cv2.imread(args["image"])


                    # image = cv2.imread(path)
                    # ratio = image.shape[0] / 500.0
                    # orig = image.copy()
                    # image = imutils.resize(image, height = 500)
                    # # convert the image to grayscale, blur it, and find edges
                    # # in the image
                    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    # edged = cv2.Canny(gray, 75, 200)
                    # # show the original image and the edge detected image
                    # print("STEP 1: Edge Detection")
                    # cv2.imshow("Image", image)
                    # cv2.imshow("Edged", edged)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # # find the contours in the edged image, keeping only the
                    # # largest ones, and initialize the screen contour
                    # cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # cnts = imutils.grab_contours(cnts)
                    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

                    # # loop over the contours
                    # for c in cnts:
                    # 	# approximate the contour
                    # 	peri = cv2.arcLength(c, True)
                    # 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    # 	print(approx)
                    # 	# if our approximated contour has four points, then we
                    # 	# can assume that we have found our screen
                    # 	if len(approx) == 2:
                    # 		screenCnt = approx
                    # 		break
                    # fixed_cnts = []
                    # print(len(fixed_cnts))
                    # # Iterate through contours
                    # for contour in cnts:
                    #     # Calculate the length of the contour using arc length
                    #     length = cv2.arcLength(contour, closed=True)
                        
                    #     # Check if the length is greater than 200 pixels
                    #     if length > 200:
                    #         # Add the contour to the fixed_cnts list
                    #         fixed_cnts.append(contour)
                    # print(len(fixed_cnts))
                        
                    # # show the contour (outline) of the piece of paper
                    # print("STEP 2: Find contours of paper")
                    # print(fixed_cnts)
                    # fixed_cnts = fixed_cnts[0]
                    # # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
                    # # print(cnts)

                    # # print(len(cnts))
                    # # for ind in range(len(cnts)):
                    # # 	if ind == 0:
                    # # 	    cv2.drawContours(image, [cnts[ind]], -1, (0, 255, 0), 2)


                    # # determine the most extreme points along the contour
                    # extLeft = tuple(fixed_cnts[fixed_cnts[:, :, 0].argmin()][0])
                    # extRight = tuple(fixed_cnts[fixed_cnts[:, :, 0].argmax()][0])
                    # extTop = tuple(fixed_cnts[fixed_cnts[:, :, 1].argmin()][0])
                    # extBot = tuple(fixed_cnts[fixed_cnts[:, :, 1].argmax()][0])

                    # # draw the outline of the object, then draw each of the
                    # # extreme points, where the left-most is red, right-most
                    # # is green, top-most is blue, and bottom-most is teal
                    # # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
                    # cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
                    # cv2.circle(image, extRight, 8, (0, 255, 0), -1)
                    # cv2.circle(image, extTop, 8, (255, 0, 0), -1)
                    # cv2.circle(image, extBot, 8, (255, 255, 0), -1)


                    # cv2.drawContours(image, fixed_cnts, -1, (0, 255, 0), 2)
                    # cv2.imshow("Outline", image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()











                    #             # # Assuming f# ixed_cnts is already defined as you mentioned in your code

                    #             # # Initialize variables to store extreme coordinates
                    #             # most_left = float('inf')
                    #             # most_right = float('-inf')
                    #             # topmost = float('inf')
                    #             # bottommost = float('-inf')

                    #             # # Iterate through contours in fixed_cnts
                    #             # for contour in fixed_cnts:
                    #             #     for point in contour:
                    #             #         # Get x and y coordinates of the point
                    #             #         x, y = point[0]

                    #             #         # Update most left coordinate
                    #             #         most_left = min(most_left, x)

                    #             #         # Update most right coordinate
                    #             #         most_right = max(most_right, x)

                    #             #         # Update topmost coordinate
                    #             #         topmost = min(topmost, y)

                    #             #         # Update bottommost coordinate
                    #             #         bottommost = max(bottommost, y)

                    #             # # Now most_left, most_right, topmost, and bottommost contain the extreme coordinates
                    #             # print("Most Left:", most_left)
                    #             # print("Most Right:", most_right)
                    #             # print("Topmost:", topmost)
                    #             # print("Bottommost:", bottommost)

import My_functions as Mf


# print(Mf.click_and_creat_graph())

# density_at_y = [1119.6774193548388, 1119.7379032258066, 1119.7983870967744, 1119.858870967742, 1119.858870967742, 1119.7983870967744, 1119.9798387096776, 1120.100806451613, 1120.2217741935485, 1120.2217741935485, 1120.1612903225807, 1120.1612903225807, 1120.2217741935485, 1120.2217741935485, 1120.1612903225807, 1120.2217741935485, 1120.342741935484, 1120.5846774193549, 1121.0685483870968, 1121.4314516129034, 1122.0967741935485, 1122.8225806451615, 1123.4274193548388, 1123.9112903225807, 1124.2137096774195, 1124.758064516129, 1125.1814516129034, 1125.7862903225807, 1126.2701612903227, 1126.6935483870968, 1126.9959677419356, 1127.2983870967744, 1127.782258064516, 1127.9032258064517, 1128.4475806451615, 1128.75, 1128.9314516129034, 1129.2943548387098, 1129.3548387096776, 1129.3548387096776, 1129.475806451613, 1129.5362903225807, 1129.5362903225807, 1129.657258064516, 1129.657258064516, 1129.657258064516, 1129.8387096774195, 1129.8387096774195, 1130.0201612903227, 1130.0806451612905, 1130.0806451612905, 1130.0806451612905, 1129.9596774193549, 1130.0201612903227, 1130.2016129032259]
# y_location = [0.5863636363636364, 0.5668831168831169, 0.5490259740259741, 0.5311688311688312, 0.5198051948051948, 0.4938311688311689, 0.4840909090909092, 0.4662337662337663, 0.45, 0.4353896103896105, 0.41266233766233773, 0.40616883116883123, 0.3883116883116884, 0.38344155844155847, 0.36396103896103904, 0.3477272727272728, 0.3347402597402598, 0.3266233766233767, 0.31850649350649357, 0.31688311688311693, 0.3120129870129871, 0.3071428571428572, 0.3006493506493507, 0.2990259740259741, 0.2990259740259741, 0.29740259740259745, 0.2957792207792208, 0.2892857142857143, 0.28766233766233773, 0.28441558441558445, 0.28116883116883123, 0.2795454545454546, 0.27142857142857146, 0.26818181818181824, 0.2633116883116884, 0.2535714285714286, 0.24707792207792212, 0.23409090909090913, 0.2243506493506494, 0.21785714285714292, 0.21136363636363642, 0.20487012987012992, 0.19512987012987015, 0.1837662337662338, 0.17402597402597408, 0.16915584415584423, 0.14480519480519483, 0.1204545454545455, 0.09772727272727276, 0.08636363636363638, 0.07012987012987014, 0.05551948051948054, 0.04090909090909094, 0.027922077922077945, 0.014935064935064954]
# print(Mf.click_and_get_y(density_at_y, y_location))


import numpy as np

rho_function = Mf.density_function(amplitude=4.1, center_of_interface=0.2)
x_values = np.linspace(0.1, 0.5, 1000)
y_values = rho_function(x_values)

# print(Mf.click_and_get_y(x_values, y_values, label='Density Function'))


# plt.plot(y_values, x_values, label='Density Function')
# plt.ylabel('Height [m]', rotation=30)
# # plt.ylabel(r'$Density [\frac{kg}{m^3}]$\mathit{\frac{v} {v_{1}}}$', fontsize = 20, rotation=0)
# plt.xlabel(r'Density $[\frac{kg}{m^3}]$', rotation=0)
# # plt.ylabel(r'$\mathit{\frac{kg}{m^3}}$', fontsize = 20, rotation=0)
# plt.title('Density Interpolation')
# plt.legend()
# plt.show()






# import matplotlib.pyplot as plt



# rho_function = Mf.density_function(amplitude = 1, interface_width = 1, 
#                                    center_of_interface = 0, mean_density_value = 0, normalizer=6)



# x_values = np.linspace(-1, 1, 1000)
# y_values = rho_function(x_values)
# # plt.title('Density vs height graph')
# plt.plot(y_values, x_values, label='Density Function', color='red')
# plt.ylabel('Height [m]', rotation=2)
# plt.xlabel(r'Density $ \left[ \frac {kg} {m^3} \right]$', rotation=0)
# plt.legend(fancybox=True, framealpha=0)
# ylabel = plt.ylabel('Height [m]', rotation=0)
# ylabel.set_verticalalignment('bottom')  # Align at the bottom of the label
# ylabel.set_y(ylabel.get_position()[1] - 0.1)
# # plt.savefig(os.path.join(Mf.select_directory(), 'a.svg'), transparent=True, bbox_inches='tight')
# plt.show()

scaling_factor = 3149.606299212598
dt = 1/60
wanted_velocity_uncertainty = 0.001 #[m/s]
scaling_factor_uncertainty = 10 #[pixel/m]
location_uncertainty_pixel = 3 #[pixel]

dt, scaling_factor,  = 0.016666666666666666, 3149.606299212598 


skipped_dt = Mf.calc_skipped_dt_from_acceptable_error(dt, scaling_factor, 
                                          wanted_velocity_uncertainty, 
                                          scaling_factor_uncertainty, 
                                          location_uncertainty_pixel)
print(skipped_dt/dt)
