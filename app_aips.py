import os
import time
from flask import Flask, request
from werkzeug.utils import secure_filename
from pathlib import Path
from flask import jsonify
import roiv2 as roi
from flask_cors import CORS
import tool
import ais_dataset
import cv2

app = Flask(__name__)
CORS(app)


@app.route('/')
def alive():
    return 'Alive'


@app.route('/scan', methods=['POST'])
def scan():

    jsonObj = {"success": 0, "data": "",
               "message": "Request method not allowed"}

    if request.method == 'POST':
        if "file" in request.files:
            f = request.files['file']
            temp_filename = str(time.time()) + "_" + secure_filename(f.filename if f.filename is not None else "")
            temp_file = os.path.join(Path.cwd(), "cache", temp_filename)
            f.save(temp_file)

            data = {"match": 0, "route": "", "image": ""} 

            ############################### Detect Section###############################
            source_image = cv2.imread(temp_file)
            
            result, rectangle_image = tool.identifyMarker(source_image)
            if (result == "" and roi.getContourCount(source_image) < 6000) or result == "T16": 
                # Special case, T15 light blue identified as white, remove background and try again.
                source_image = roi.removeBackground(source_image)
                result, rectangle_image = tool.identifyMarker(source_image)

            if result:
                data["match"] = 1 
                data["route"] = ais_dataset.data[result]["id"]
                data["image"] = roi.getThumbnail(rectangle_image)

            ############################### Detect Section###############################

            # Cleanup temp file
            os.remove(temp_file)

            jsonObj["success"] = 1
            jsonObj["data"] = data
            jsonObj["message"] = ""
        else:
            jsonObj["message"] = "Please upload a file"
    return jsonify(jsonObj)

if __name__ == "__main__":
    app.run()
