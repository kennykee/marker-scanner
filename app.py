import cv2
import roiv2 as roi
import tool
from pathlib import Path

for set in range(1,5):
    for unit in range(1,21):
        source_image = cv2.imread(f'Test_Case//Set_{set}//T{unit}.jpg') 
        result, _ = tool.identifyMarker(source_image)
        if (result == "" and roi.getContourCount(source_image) < 6000) or result == "T16": 
            # Special case, T15 light blue identified as white, remove background and try again.
            source_image = roi.removeBackground(source_image)
            result, _ = tool.identifyMarker(source_image)
            
        print(f"Set {set} - T{unit} = {result} {' - Wrong' if result != f'T{unit}' else ''}")