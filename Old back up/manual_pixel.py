import cv2
import numpy as np

class pixy:

    def __init__(self):
        self.click_count = 0
        self.img = None
        self.a = []

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_count == 0:
                height, width, channels = self.img.shape
                imagey = cv2.line(self.img, (0, y), (width,y), (0,255,0), 2)
                cv2.imshow('image', imagey) 
                imagey = cv2.line(self.img, (x, 0), (x, height), (0,255,0), 2)

            self.click_count += 1
            self.a.append((x,y))
    
    def manual_pixel(self, curPth):
        self.img = cv2.imread(curPth)

        # Create a window and set the mouse callback function
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.mouse_callback)


        # Display the image and wait for a mouse event
        while self.click_count<2:
            cv2.imshow('image', self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Close the window
        cv2.destroyAllWindows()

        b = np.array(self.a)
        distance  = np.linalg.norm(b[0]-b[-1])

        cv2.waitKey(10)
        cv2.destroyAllWindows()

        return distance




#######
# Old and Working.
########
# import cv2
# import numpy as np


# # Initialize the click count
# click_count = 0

# # Initialize psitions
# a =[]

# def manual_pixel(curPth, show_coordinates = False):
#     # click_count = 0
#     global click_count
#     img = cv2.imread(curPth)

#     def mouse_callback(event, x, y, flags, param):
#         global click_count, a

#         if event == cv2.EVENT_LBUTTONDOWN:
#             if click_count == 0:
#                 height, width, channels = img.shape
#                 imagey = cv2.line(img, (0,y), (width,y), (0,255,0), 2)
#                 cv2.imshow('image', imagey) 
#                 imagey = cv2.line(img, (x,0), (x, height), (0,255,0), 2)

#             click_count += 1
#             a.append((x,y))

#             if show_coordinates:
#                 print(f'Clicked at ({x}, {y})')


#     # Create a window and set the mouse callback function
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image', mouse_callback, param=click_count)


#     # Display the image and wait for a mouse event
#     while click_count<2:
#         cv2.imshow('image', img)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     # Close the window
#     cv2.destroyAllWindows()

#     b = np.array(a)
#     distance  = np.linalg.norm(b[0]-b[-1])

#     cv2.waitKey(10)
#     cv2.destroyAllWindows()

#     click_count = 0
#     return distance

