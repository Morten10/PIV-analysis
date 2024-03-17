import My_functions as Mf
import os
from openpiv import tools, scaling, pyprocess, validation, filters, preprocess
import pathlib
import numpy as np
import datetime


master_pciture_folder = Mf.get_folder_names(Mf.select_directory())

parent_folder = '.\\'
print(parent_folder)
print(os.path.abspath(parent_folder))

pickle_save_folder = '.\\output'
Mf.create_directory(pickle_save_folder)



Masked = False
if False:
    data_for_PIV = []
    for ind, pic_folder in enumerate(master_pciture_folder):
        print(ind+1)
        origin = [0, 1024]
        
        picture_list = Mf.find_pictures(pic_folder)
        img_number, real_cor, object_range_cut, prior_knowledge, contrast = Mf.select_circle_napari_advance(pic_folder,
                                                                                                            image_number = 53,
                                                                                                            sphere_location = [469, 157],
                                                                                                            Object_data = [[435, 123], [435, 196], [505, 196], [505, 123]],
                                                                                                            PTV_data = [[375, 0], [375, 1024], [608, 1024], [608, 0]])
        
        bias_x, bias_y = object_range_cut[1].start + (object_range_cut[1].stop - object_range_cut[1].start)/2 - real_cor[0], object_range_cut[0].start + (object_range_cut[0].stop - object_range_cut[0].start)/2 - real_cor[1]
        im_a_path = picture_list[img_number]
        the_selected_image = Mf.read_image(im_a_path)
        object_array = the_selected_image[object_range_cut]
        PIV_Range = slice(0,1024), slice(0,1024)
        data_for_PIV.append([img_number, real_cor, object_range_cut, prior_knowledge, contrast, bias_x, bias_y, the_selected_image, object_array, PIV_Range])

    Expirement_date_pickle = Mf.convert_date(Mf.read_data_from_cihx(pic_folder, '<date>'))

    Mf.save_pickle(os.path.join(pickle_save_folder, f'inital_data_{Expirement_date_pickle}.pkl'), data_for_PIV)

print(f'\n\n\n\n\nStarting Data Processing Date: {datetime.date.today().strftime("%Y/%m/%d")}, Data Processing time: {datetime.datetime.now().strftime("%H:%M:%S")}')



the_big_pickle_path = '.\\Excel Files\\The Big Pickle.pkl'
object_pattern, bias, contrast, prior_knowledge, real_cor, object_range_cut, img_number, contrast, prior_knowledge, real_cor, object_range_cut, img_number, contrast, prior_knowledge, real_cor, object_range_cut, img_number = Mf.load_pickle(the_big_pickle_path)['Object To Follow']
bias_x, bias_y = bias
object_array = object_pattern
PIV_Range = slice(0,1024), slice(0,1024)
chossed_output_path = r'C:\Users\Morten\Desktop\PIV output'


Use_fft = True
first_folder_number = int(os.path.basename(master_pciture_folder[0])[-4:])
for ind, pic_folder in enumerate(master_pciture_folder, start = first_folder_number-1):
    Expirement_date_pickle = Mf.convert_date(Mf.read_data_from_cihx(pic_folder, '<date>'))
    # img_number, real_cor, object_range_cut, prior_knowledge, contrast, bias_x, bias_y, the_selected_image, object_array, PIV_Range = Mf.load_pickle(the_big_pickle_path)['Object To Follow']
        # os.path.join(pickle_save_folder, f'inital_data_{Expirement_date_pickle}.pkl'))[ind]
    
    # We need only this: and they all can be constant.
    # prior_knowledge, bias_x, bias_y, object_array, PIV_Range

    Expirement_date = Mf.read_data_from_cihx(pic_folder, '<date>')
    Record_number = int(os.path.splitext(Mf.files_in_folder(pic_folder, 'cihx')[0])[0][-2:])

    save_folder = os.path.join(chossed_output_path, 'output', f'{Expirement_date.replace("/", " ")} - Rec {Record_number}')
    save_folder_multy = os.path.join(save_folder, 'Multy')

    multy_cut_image = 'multy_cut_image'
    if Masked:
        multy_cut_image = multy_cut_image+ '_masked'
    multy_cut_image_folder = os.path.join(save_folder_multy, multy_cut_image)
    Mf.create_directory(multy_cut_image_folder)

    picture_list = Mf.find_pictures(pic_folder)
    print(f'Expirement date = {Expirement_date}  .   Record number - {Record_number} number of pictures in folder {len(picture_list)}')
   

    unit = Mf.read_data_from_cihx(pic_folder, '<distanceUnit>')
    scaling_factor = 1/Mf.read_data_from_cihx(pic_folder, '<sizeOfPixel>', return_number = True) #[pixel/mm]
    if 'mm' in unit:
        scaling_factor *= 1000
    scaling_factor # [pixel/m]

    clean_locations = []
    clean_locations_plus_frame = []
    relative_locations = []

    high_pass_filter = True
    edge_enhancement = True
    apply_static_manual_mask = True
    sphere_diameter_pixel = 26
    see_sphere_location = False
    to_print = False


    frame_rate = Mf.read_data_from_cihx(pic_folder, return_number=True, return_int=True) # fps
    dt = 1/frame_rate # sec

    window_size = 32 # pixels
    search_area_size = window_size # pixels 
    overlap = round(search_area_size/2) # pixels
    scale = 0.01 # scale defines here the arrow length
    width = 0.002 # width is the thickness of the arrow
    threshold = 1.1
    u_thresholds = 100


    display_plot = True # display plot
    



    def funcy( args ):
        """A function to process each image pair."""

        # this line is REQUIRED for multiprocessing to work
        # always use it in your custom function

        file_a, file_b, counter = args

        #####################
        # Here goes you code
        #####################

        number = int(os.path.splitext(os.path.basename(file_a))[0][-5:])
        the_file_name = 'multy_%04d.txt'
        if Masked:
            the_file_name = the_file_name[:-4] + '_Mask.txt'
        if to_print:
            #Prints
            print(file_a)
            print(file_b)
            print(the_file_name % number)

        # read images into numpy arrays
        im_a = tools.imread(file_a)
        im_b = tools.imread(file_b)


        #finding sphere coordinates
        #cor a cnd corr b are coordinaes of thesphere in each of the images, found by he coorelaton function. in [pixel] with respect the origin (0,0) is top left corner
        corr_a = Mf.track_object(object_array, file_a, fft = Use_fft, bias = [bias_x, bias_y], prior_knowledge = prior_knowledge, show = see_sphere_location)

        #the real spheres location is the interpolation between those two images.
        clean_locations.append(corr_a)
        clean_locations_plus_frame.append([corr_a, number])
        
        frame_aa = im_a
        frame_bb = im_b
        
        if PIV_Range is not None:
            frame_aa = im_a[PIV_Range]
            frame_bb = im_b[PIV_Range]

            #Optional part apply high pass filter if manualy contrasnt has not been specified:
        if high_pass_filter:
            if contrast[0] == 0 and contrast[1] == 255:
                frame_aa = Mf.high_pass_filter(frame_aa)
                frame_bb = Mf.high_pass_filter(frame_bb)
            else: # apply contrast from napari
                frame_aa = Mf.Change_contrast(frame_aa, contrast)
                frame_bb = Mf.Change_contrast(frame_bb, contrast)
        
        if edge_enhancement:
            frame_aa = Mf.edge_enhancement(frame_aa)
            frame_bb = Mf.edge_enhancement(frame_bb)

        #Optional part: static mask
        if apply_static_manual_mask: #manual
            frame_aa = Mf.static_manual_mask(frame_aa, x = corr_a[0] - PIV_Range[1].start, y = corr_a[1] - PIV_Range[0].start, radius = sphere_diameter_pixel)
            
            corr_b = Mf.track_object(object_array, file_b, bias = [bias_x, bias_y], prior_knowledge = prior_knowledge, show = see_sphere_location)
            frame_bb = Mf.static_manual_mask(frame_bb, x = corr_b[0] - PIV_Range[1].start, y = corr_b[1] - PIV_Range[0].start, radius = sphere_diameter_pixel)

        
        
        ##autumatic
        # if apply_static_mask:
            # static_mask_a = Mf.static_mask(frame_aa)
            # static_mask_b = Mf.static_mask(frame_bb)
            # frame_aa[static_mask_a] = 0
            # frame_bb[static_mask_b] = 0

        #Optional part: dynamic mask
        if Masked:
            frame_aa, _ = preprocess.dynamic_masking(frame_aa, method='intensity', filter_size=7, threshold=0.01)
            frame_bb, _ = preprocess.dynamic_masking(frame_bb, method='intensity', filter_size=7, threshold=0.01)
        
        # save image
        the_image_name = 'multy_%04d.tif'
        if Masked:
            the_image_name = the_image_name[:-4] + '_Mask.tif'

        relative_locations.append([corr_a[0] - PIV_Range[1].start, corr_a[1] - PIV_Range[0].start])
        Mf.save_image(os.path.join(multy_cut_image_folder, the_image_name % number), frame_aa)
        

        # PIV analysis
        u, v, sig2noise = pyprocess.extended_search_area_piv(frame_aa.astype(np.int32), 
                                                        frame_bb.astype(np.int32), 
                                                        window_size = window_size, 
                                                        overlap = overlap, 
                                                        dt = dt, 
                                                        search_area_size = search_area_size, 
                                                        sig2noise_method = 'peak2peak')
        
        x, y = pyprocess.get_coordinates(image_size = frame_aa.shape, 
                                        search_area_size = search_area_size, 
                                        overlap = overlap )


        flags_g = validation.global_val(u, v, (-u_thresholds, u_thresholds), (-u_thresholds, u_thresholds))
        flags_s2n = validation.sig2noise_val(sig2noise, threshold = threshold)
        flags = flags_g | flags_s2n
        u, v = filters.replace_outliers(u, v, flags, method='localmean', max_iter = 5, kernel_size = 3)

        # Turning everything to SI
        x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = scaling_factor)

        # 0,0 bottom left, positive rotation rate is counterclockwise
        x, y, u, v = tools.transform_coordinates(x, y, u, v)


        # save to a file
        tools.save(os.path.join(save_folder_multy, the_file_name % number), x, y, u, v, flags, None, fmt='%.16e', delimiter='\t')

    task = tools.Multiprocesser(data_dir = pathlib.Path(pic_folder), pattern_a = 'C*.tif', pattern_b = '(1+2),(2+3)')
    task.run(func = funcy, n_cpus = 1)
    clean_locations.append(Mf.track_object(object_array, picture_list[-1], bias = [bias_x, bias_y], prior_knowledge = prior_knowledge))

    display_plot = True
    if display_plot:
        cutted_txt_list = Mf.files_in_folder(save_folder_multy, 'txt', full_path=True)
        multy_cut_list = Mf.find_pictures(multy_cut_image_folder)
        for i in range(len(cutted_txt_list)):
            the_vector_folder = Mf.save_vector_field_as_image(cutted_txt_list[i], multy_cut_list[i], scaling_factor=scaling_factor, vector_scale=scale, window_size=window_size,width=width, sphere_loc=relative_locations[i])
    Mf.eliminate_folder(multy_cut_image_folder)

print(f'\n\n\n\nDone, please check "{pickle_save_folder}"     folder tee see the output files\nThe time is {datetime.date.today().strftime("%Y/%m/%d")}  {datetime.datetime.now().strftime("%H:%M:%S")}')