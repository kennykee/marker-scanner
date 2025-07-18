from math import ceil
import cv2 as cv
import numpy as np
import os
from pathlib import Path
import base64
from typing import Tuple

ROI_WIDTH_RATIO = 0.2346939
ROI_HEIGHT_RATIO = 0.2263158

def getROI(original_image) -> Tuple[cv.Mat|None, cv.Mat|None, int, int]:

    NOMATCH = (None, None, 0, 0) 

    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    (bounding_coordinate, final_contour, top_x, top_y, bottom_x, bottom_y) = getBoundingMarker(original_image)

    if len(bounding_coordinate) == 0:
        return NOMATCH

    capture = cv.GaussianBlur(original_image, (7, 7), 0)
    capture = cv.threshold(capture, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    if top_x == capture.shape[1]:
        corner_width = bounding_coordinate["top_right"][
            2] if "top_right" in bounding_coordinate else bounding_coordinate["bottom_right"][2]
        top_x = bottom_x - 1 / ROI_WIDTH_RATIO * corner_width

    if top_y == capture.shape[0]:
        corner_height = bounding_coordinate["bottom_left"][
            3] if "bottom_left" in bounding_coordinate else bounding_coordinate["bottom_right"][3]
        top_y = bottom_y - 1 / ROI_HEIGHT_RATIO * corner_height

    if bottom_x == 0:
        corner_width = bounding_coordinate["top_left"][
            2] if "top_left" in bounding_coordinate else bounding_coordinate["bottom_left"][2]
        bottom_x = top_x + 1 / ROI_WIDTH_RATIO * corner_width

    if bottom_y == 0:
        corner_height = bounding_coordinate["top_left"][
            3] if "top_left" in bounding_coordinate else bounding_coordinate["top_right"][3]
        bottom_y = top_y + 1 / ROI_HEIGHT_RATIO * corner_height

    # Remove marker contours
    cv.drawContours(capture, list(final_contour.values()), -
                    1, (255, 255, 255), -1)

    marker_width = bottom_x - top_x
    marker_height = bottom_y - top_y

    # Clear all 4 corners. Clear whole adjacent lines.
    if "top_left" not in bounding_coordinate:
        fill_contours = np.array([
            [top_x, top_y],
            [bottom_x, top_y],
            [bottom_x, top_y + ROI_HEIGHT_RATIO * marker_height / 3],
            [top_x + ROI_WIDTH_RATIO * marker_width / 3,
                top_y + ROI_HEIGHT_RATIO * marker_height / 3],
            [top_x + ROI_WIDTH_RATIO * marker_width / 3, bottom_y],
            [top_x, bottom_y]
        ]).astype(int)
        cv.fillPoly(capture, pts=[fill_contours], color=(255, 255, 255))

    if "top_right" not in bounding_coordinate:
        fill_contours = np.array([
            [top_x, top_y],
            [bottom_x, top_y],
            [bottom_x, bottom_y],
            [bottom_x - ROI_WIDTH_RATIO * marker_width / 3, bottom_y],
            [bottom_x - ROI_WIDTH_RATIO * marker_width / 3,
                top_y + ROI_HEIGHT_RATIO * marker_height / 3],
            [top_x, top_y + ROI_HEIGHT_RATIO * marker_height / 3]
        ]).astype(int)
        cv.fillPoly(capture, pts=[fill_contours], color=(255, 255, 255))

    if "bottom_left" not in bounding_coordinate:
        fill_contours = np.array([
            [top_x, top_y],
            [top_x + ROI_WIDTH_RATIO * marker_width / 3, top_y],
            [top_x + ROI_WIDTH_RATIO * marker_width / 3,
                bottom_y - ROI_HEIGHT_RATIO * marker_height / 3],
            [bottom_x, bottom_y - ROI_HEIGHT_RATIO * marker_height / 3],
            [bottom_x, bottom_y],
            [top_x, bottom_y]
        ]).astype(int)
        cv.fillPoly(capture, pts=[fill_contours], color=(255, 255, 255))

    if "bottom_right" not in bounding_coordinate:
        fill_contours = np.array([
            [bottom_x - ROI_WIDTH_RATIO * marker_width / 3, top_y],
            [bottom_x, top_y],
            [bottom_x, bottom_y],
            [top_x, bottom_y],
            [top_x, bottom_y - ROI_HEIGHT_RATIO * marker_height / 3],
            [bottom_x - ROI_WIDTH_RATIO * marker_width / 3,
                bottom_y - ROI_HEIGHT_RATIO * marker_height / 3],
        ]).astype(int)
        cv.fillPoly(capture, pts=[fill_contours], color=(255, 255, 255))

    # Crop
    top_x, top_y, bottom_x, bottom_y = int(top_x), int(
        top_y), int(bottom_x), int(bottom_y)
    capture = capture[top_y:bottom_y, top_x:bottom_x]
    original_image = original_image[top_y:bottom_y, top_x:bottom_x]

    # Trim Capture
    gray = 255*(capture < 160).astype(np.uint8)
    coords = cv.findNonZero(gray)
    x, y, w, h = cv.boundingRect(coords)
    capture = capture[y:y+h, x:x+w]
    
    trim_original_image = original_image[y:y+h, x:x+w]

    if w > 1 and h > 1:
        trim_original_image = cv.resize(
            trim_original_image, (400, int(400/w*h)), interpolation=cv.INTER_NEAREST)

    return (trim_original_image, original_image, np.sum(capture == 0), np.sum(capture == 255))


def getBoundingMarker(original_image):

    # Remove noise
    capture = cv.GaussianBlur(original_image, (7, 7), 0)
    capture = cv.threshold(
        capture, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    thresh = cv.threshold(capture, 127, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Choose biggest bounding rectangle
    # top_left, top_right, bottom_left, bottom_right (0,0)
    bounding_coordinate = {}
    final_contour = {}

    top_x = capture.shape[1]
    top_y = capture.shape[0]
    bottom_x = 0
    bottom_y = 0

    # Don't have child
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][2] == -1:

            marker_result = getMarkerPosition(contour, capture.shape[1], capture.shape[0])  # {p,x,y,w,h}
            x, y, w, h = marker_result["x"], marker_result["y"], marker_result["w"], marker_result["h"]

            if marker_result["position"] == "top_left":
                if "top_left" in bounding_coordinate:
                    if x - bounding_coordinate["top_left"][0] + y - bounding_coordinate["top_left"][1] < 0:
                        bounding_coordinate["top_left"] = (x, y, w, h)
                        final_contour["top_left"] = contour
                else:
                    bounding_coordinate["top_left"] = (x, y, w, h)
                    final_contour["top_left"] = contour

                top_x = x if x < top_x else top_x
                top_y = y if y < top_y else top_y

            elif marker_result["position"] == "top_right":
                if "top_right" in bounding_coordinate:
                    if x - bounding_coordinate["top_right"][0] - (y - bounding_coordinate["top_right"][1]) > 0:
                        bounding_coordinate["top_right"] = (x, y, w, h)
                        final_contour["top_right"] = contour
                else:
                    bounding_coordinate["top_right"] = (x, y, w, h)
                    final_contour["top_right"] = contour

                top_y = y if y < top_y else top_y
                bottom_x = (x + w) if (x + w) > bottom_x else bottom_x

            elif marker_result["position"] == "bottom_left":
                if "bottom_left" in bounding_coordinate:
                    if -(x - bounding_coordinate["bottom_left"][0]) + (y - bounding_coordinate["bottom_left"][1]) > 0:
                        bounding_coordinate["bottom_left"] = (x, y, w, h)
                        final_contour["bottom_left"] = contour
                else:
                    bounding_coordinate["bottom_left"] = (x, y, w, h)
                    final_contour["bottom_left"] = contour

                top_x = x if x < top_x else top_x
                bottom_y = (y+h) if (y+h) > bottom_y else bottom_y

            elif marker_result["position"] == "bottom_right":
                if "bottom_right" in bounding_coordinate:
                    if x - bounding_coordinate["bottom_right"][0] + y - bounding_coordinate["bottom_right"][1] > 0:
                        bounding_coordinate["bottom_right"] = (x, y, w, h)
                        final_contour["bottom_right"] = contour
                else:
                    bounding_coordinate["bottom_right"] = (x, y, w, h)
                    final_contour["bottom_right"] = contour

                bottom_x = (x+w) if (x+w) > bottom_x else bottom_x
                bottom_y = (y+h) if (y+h) > bottom_y else bottom_y

    return (bounding_coordinate, final_contour, top_x, top_y, bottom_x, bottom_y)


def getMarkerPosition(contour, capturedWidth, captureHeight):

    x, y, w, h = cv.boundingRect(contour)
    result = {"position": "None", "x": x, "y": y, "w": w, "h": h}

    min_x = x
    min_y = y
    max_x = x+w
    max_y = y+h

    # 1) Make sure w and h has 90% similarities because it is a square
    if abs(1-w/h) > 0.2 or w < 10:
        return result
    
    # Make sure marker is at least 1/5 of screen width and height 
    if w < capturedWidth * ROI_WIDTH_RATIO / 5:
        return result

    if h < captureHeight * ROI_HEIGHT_RATIO / 5:
        return result
    
    # Make sure marker is at max min and height
    if w > capturedWidth * ROI_WIDTH_RATIO:
        return result
    
    if h > captureHeight * ROI_HEIGHT_RATIO:
        return result


    # 2) Occupies at least 1/3 of rectangle of each edge. Ensure 90% of coordinates fall into this range else not valid contour.
    # Use counter measurement. Find 2/3 of the areas that has only a max of 10% of total coordinates.

    start_x = min_x + int(round(w/3))
    end_x = min_x + (2 * int(round(w/3)))
    start_y = min_y + int(round(h/3))
    end_y = min_y + (2 * int(round(h/3)))

    max_permitted = len(contour) * 0.1

    # At least 90 to return true
    empty_area_count = {
        "top_left": 0,
        "top_right": 0,
        "bottom_left": 0,
        "bottom_right": 0
    }

    for point in contour:

        if point[0][0] in range(start_x, max_x+1) and point[0][1] in range(start_y, max_y+1):
            empty_area_count["top_left"] += 1
        if point[0][0] in range(min_x, end_x+1) and point[0][1] in range(start_y, max_y+1):
            empty_area_count["top_right"] += 1
        if point[0][0] in range(start_x, max_x+1) and point[0][1] in range(min_y, end_y+1):
            empty_area_count["bottom_left"] += 1
        if point[0][0] in range(min_x, end_x+1) and point[0][1] in range(min_y, end_y+1):
            empty_area_count["bottom_right"] += 1

    # Remove element above max_permitted and get min value in dictionary
    for k, v in empty_area_count.items():
        if v <= max_permitted and (result["position"] == "None" or v < empty_area_count[result["position"]]):
            result["position"] = k

    return result

def removeBackground(original_image):
    # Step 1: Convert to HSV
    hsv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_image)

    # Step 2: Identify lighter colors (high Value) and darken their Hue
    # Define a threshold for "light" colors (e.g., V > 200)
    light_threshold = 100
    dark_hue_shift = 240  # Shift hue to a darker tone (adjustable)

    # Create a mask for light areas
    light_mask = v > light_threshold

    # Replace hue in light areas with a darker hue
    # For simplicity, reduce hue value (darker hues are typically lower, e.g., towards blue/red)
    h_modified = h.copy()
    h_modified[light_mask] = np.clip(h[light_mask] - dark_hue_shift, 0, 180)

    # Step 3: Reduce brightness (Value) of light areas to enhance darkening
    v_modified = v.copy()
    v_modified[light_mask] = np.clip(v[light_mask] - 50, 0, 255)

    # Step 4: Merge modified HSV channels
    hsv_modified = cv.merge([h_modified, s, v_modified])
    modified_image = cv.cvtColor(hsv_modified, cv.COLOR_HSV2BGR)

    # Step 5: Define HSV range for foreground detection
    lower_saturation = np.array([0, 30, 0])    
    upper_saturation = np.array([180, 255, 255])  

    # Step 6: Create mask for foreground
    mask = cv.inRange(hsv_modified, lower_saturation, upper_saturation)

    # Step 7: Refine mask with morphological opening
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Step 8: Invert mask for background
    inverted_mask = cv.bitwise_not(mask)

    # Step 9: Create white background
    white_background = np.full_like(original_image, 255)

    # Step 10: Extract foreground and background
    foreground = cv.bitwise_and(modified_image, modified_image, mask=mask)
    background = cv.bitwise_and(white_background, white_background, mask=inverted_mask)

    # Step 11: Combine foreground and white background
    result = cv.add(foreground, background)

    return result

def compareImageRatio(dataset, image):

    if image.shape[0] <= 1:
        return {}

    data = {}
    image_ratio = image.shape[1] / image.shape[0]

    for x, y in dataset.items():
        if (abs(image_ratio-y["dimension_ratio"]) / y["dimension_ratio"]) < 0.1:
            data[x] = y
    return data

def compareFillRatio(dataset, ratio):
    data = {}
    for x, y in dataset.items():
        if (abs(ratio-y["fill_ratio"]) / y["fill_ratio"]) < 0.2:
            data[x] = y
    return data

def compareTemplateMatching(image_with_marker, dataset, marker_type):

    height, width = image_with_marker.shape[:2]

    if width < 50 or height < 50:
        return ""
    
    image_resized = resize_to_width(image_with_marker, 400)

    best_match_val = 0.0
    best_match_id = ""

    for x, y in dataset.items():
        marker_file = os.path.join(Path.cwd(), marker_type, x + ".jpg")
        marker_image = cv.imread(marker_file, cv.IMREAD_GRAYSCALE)
        marker_image = resize_to_width(marker_image, 400)

        result = cv.matchTemplate(image_resized, marker_image, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(result)
        if max_val > best_match_val and max_val > 0.5:
            best_match_val = max_val
            best_match_id = x
    
    return best_match_id


def resize_to_width(image, width):
    aspect_ratio = width / float(image.shape[1])
    dimensions = (width, int(image.shape[0] * aspect_ratio))
    return cv.resize(image, dimensions, interpolation=cv.INTER_NEAREST)

def compareSIFT(dataset, input_image):
    data = {}

    # Resize input to 600 width
    width_input = input_image.shape[1]
    height_input = input_image.shape[0]

    if width_input > 1 and height_input > 1:
        resized_height_input = int(600/width_input*height_input)
        input_image = cv.resize(
            input_image, (600, resized_height_input), interpolation=cv.INTER_NEAREST)
    else:
        return {}

    # Initiate SURF detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIRF
    input_dsc = sift.detectAndCompute(input_image, None)[1]

    # BFMatcher with default params
    bf = cv.BFMatcher()

    for x, y in dataset.items():
        marker_file = os.path.join(
            Path.cwd(), "data", "scout_marker", y["file_name"])
        marker_image = cv.imread(marker_file, 0)
        marker_kp, marker_dsc = sift.detectAndCompute(marker_image, None)
        matches = bf.knnMatch(input_dsc, marker_dsc, k=2)
        good = 0
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good += 1
        similarity = good / len(marker_kp) * 100
        print(str(similarity) + "% - " + str(x))

        # More than 25 and highest ranked
        if similarity > 25:
            y["score"] = similarity
            data[x] = y

    data = dict(
        sorted(data.items(), key=lambda item: item[1]['score'], reverse=True))

    if data:
        data = {list(data.keys())[0]: list(data.values())[0]}

    return data

def detect_object(source, dataset, ratio_thresh=0.68, match_percent_thresh=0.2):

    best_match_val = 0.0
    best_match_id = ""

    sift = cv.SIFT_create()

    image_resized = resize_to_width(source, 500)
    _, descriptors_image = sift.detectAndCompute(image_resized, None)

    for x, y in dataset.items():

        # Compare with Base_Marker_With_Background + Full_Marker, take highest

        marker_file_1 = os.path.join(Path.cwd(), "Base_Marker_With_Background", x + ".jpg")
        marker_file_2 = os.path.join(Path.cwd(), "Full_Marker", x + ".jpg")

        marker_image_1 = cv.imread(marker_file_1, cv.IMREAD_GRAYSCALE)
        marker_image_2 = cv.imread(marker_file_2, cv.IMREAD_GRAYSCALE)

        marker_image_1 = resize_to_width(marker_image_1, 500)
        marker_image_2 = resize_to_width(marker_image_2, 500)

        keypoints_template_1, descriptors_template_1 = sift.detectAndCompute(marker_image_1, None)
        keypoints_template_2, descriptors_template_2 = sift.detectAndCompute(marker_image_2, None)
    
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

        matches_1 = bf.knnMatch(descriptors_image, descriptors_template_1, k=2)
        matches_2 = bf.knnMatch(descriptors_image, descriptors_template_2, k=2)

        good_matches_1 = [m for m, n in matches_1 if m.distance < ratio_thresh * n.distance]
        good_matches_2 = [m for m, n in matches_2 if m.distance < ratio_thresh * n.distance]

        match_percentage_1 = len(good_matches_1) / len(keypoints_template_1) if keypoints_template_1 else 0
        match_percentage_2 = len(good_matches_2) / len(keypoints_template_2) if keypoints_template_2 else 0
        
        if match_percentage_1 >= match_percent_thresh and match_percentage_1 > best_match_val:
            best_match_val = match_percentage_1
            best_match_id = x

        if match_percentage_2 >= match_percent_thresh and match_percentage_2 > best_match_val:
            best_match_val = match_percentage_2
            best_match_id = x

    return best_match_id


def compareContour(image, dataset):
    
    match_rate = {}

    image = cv.Canny(image, 100, 200)

    contours,_ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contour_lengths = [(cv.arcLength(contour, True), contour) for contour in contours]
    contour_lengths.sort(key=lambda x: x[0], reverse=True)

    top_contours = [contour for _, contour in contour_lengths[:min(20, len(contour_lengths))]]
    contours_length = len(top_contours)

    # get up to longest 20 arc, match at least 80% of total arc compared
    for x, y in dataset.items():
        marker_file = os.path.join(Path.cwd(), "Base_Marker_With_Background", x + ".jpg")
        marker_image = cv.imread(marker_file, cv.IMREAD_GRAYSCALE)
        marker_image = cv.threshold(marker_image, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        marker_image = cv.threshold(marker_image, 127, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
        marker_contours, _ = cv.findContours(marker_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        marker_contour_lengths = [(cv.arcLength(contour, True), contour) for contour in marker_contours]
        marker_contour_lengths.sort(key=lambda x: x[0], reverse=True)

        top_marker_contours = [contour for _, contour in marker_contour_lengths[:min(20, len(marker_contour_lengths))]]
        marker_contours_length = len(top_marker_contours)

        if not (False if contours_length == 0 or marker_contours_length == 0 else abs(contours_length - marker_contours_length) / max(contours_length, marker_contours_length) * 100 <= 10):
            continue
        
        min_match = ceil(0.80*len(marker_contours))
        match_count = 0

        for marker_contour in marker_contours:
            matched_index = []
            for index, contour in enumerate(contours):
                match_score = cv.matchShapes(marker_contour, contour, cv.CONTOURS_MATCH_I1, 0.0)
                if match_score < 0.5 and index not in matched_index and match_score != 0:
                    match_count += 1
                    matched_index.append(index)
                    break
        if match_count > 0 and (match_count >= min_match or match_count >= len(marker_contours) - 1):
            match_rate[x] = match_count

    if(len(match_rate) > 0):
        return max(match_rate, key=lambda k: match_rate[k])
    else:
        return ""
    
def getThumbnail(input_image):

    height, width = input_image.shape[:2]

    if width < 50 or height < 50:
        return ""

    # 72px width
    width = 72
    height = int((width / input_image.shape[1]) * input_image.shape[0])  # Maintain aspect ratio
    resized_image = cv.resize(input_image, (width, height), interpolation=cv.INTER_LINEAR)
    
    # Encode the resized image into JPEG and convert it to a byte object
    _, buffer = cv.imencode('.jpg', resized_image)
    buffer_bytes = buffer.tobytes()  # Convert ndarray to bytes
    
    # Encode to base64 and decode to string
    return base64.b64encode(buffer_bytes).decode()

def getContourCount(input_image):
    gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, thresh = cv.threshold(binary, 127, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)