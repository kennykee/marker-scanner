import ais_dataset 
import roiv2 as roi

###### Image Profile ######
# [T11,T12,T13,T15,T16,T17,T18] = Trimmable Image + dimension ratio + black white ratio + Highest template matching 
# [T2,T3,T4,T5,T6,T7,T14,T19,T20] = Detectable markers with frame for template matching (min + highest) + Full_Marker with frame
# [T1,T8,T9,T10] = Non detectable markers for SIFT + Full_Marker (min + highest)

def identifyMarker(source_image):
    found_key = ""

    #1) All can detect corner and extract marker
    (trimmed_image, image_with_marker, black_count, white_count) = roi.getROI(source_image) 

    if trimmed_image is not None:
        ################# Part 1 #################
        # Trimmable dataset
        keyset = ["T11","T12","T13","T15","T16","T17","T18"]
        dataset = {key: value for key, value in ais_dataset.data.items() if key in keyset}
        dataset = roi.compareFillRatio(dataset, black_count / white_count if white_count != 0 else 0)
        dataset = roi.compareImageRatio(dataset, trimmed_image)

        if len(dataset) == 1:
            found_key = list(dataset.keys())[0]    
        elif len(dataset) > 1:
            found_key = roi.compareContour(image_with_marker, dataset)

        ################# Part 2 #################
        # Detectable markers with frame for template matching (min + highest) 
        if len(dataset) == 0:
            keyset = ["T2","T3","T4","T5","T6","T7","T14","T19","T20"]
            dataset = {key: value for key, value in ais_dataset.data.items() if key in keyset}
        # Template matching higher than 50% and first found
        if found_key == "" and roi.getContourCount(source_image) < 6000:
            found_key = roi.compareTemplateMatching(image_with_marker, dataset, "Base_Marker_With_Background")
            # if found_key:
            #     print("Part-2", found_key)

    ################# Optional Prefiltering TODO #################
    # =================
    # 1) if number of contours in captured image exceeds ["T1","T8","T9","T10"] max contour
    # count by 100%, it means camera too far, return false.
    # 2) Optional = front end, instead of automatically capture and send,
    # change to press capture button to capture and send.
    # 3) Determine number of contour using marker with background
    # =================
    ##############################################################

    ################# Part 3 #################
    # Cannot find matching - use SIFT for selected captured mark. 30% to match
    if found_key == "" and roi.getContourCount(source_image) < 6000:
        keyset = ["T1","T8","T9","T10"]
        dataset = {key: value for key, value in ais_dataset.data.items() if key in keyset}
        found_key = roi.detect_object(source_image, dataset)
        # if found_key:
        #     print("Part-3", found_key)   

    ################# Part 4 #################
    # Contour matching
    if found_key == "" and roi.getContourCount(source_image) < 6000:
        keyset = ["T1","T8","T9","T10"]
        dataset = {key: value for key, value in ais_dataset.data.items() if key in keyset}
        found_key = roi.compareContour(source_image, dataset)
        # if found_key:
        #     print("Part-4", found_key)   

    ################ Return Tuple ################
    # found_key
    # image_with_marker or source_image

    return found_key, source_image if image_with_marker is None else image_with_marker
