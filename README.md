# AR Detection Engine – Ahmad Ibrahim Primary School Science Wall

## Project Description

This is the detection engine powering the **Ahmad Ibrahim Primary School AR Science Wall** - an interactive mobile web application designed to enrich student learning through augmented reality.

The app transforms passive printed environmental displays into engaging learning tools. By scanning AR markers embedded in the wall, users instantly receive relevant educational content on their devices, deepening their understanding of topics like sustainability and environmental systems.

Visit [SchoolApp.sg](https://schoolapp.sg/project/ahmad-ibrahim-primary-school-ar-science-wall) for project images.

### Key Learning Themes:
- Sustainability
- Waste management
- Environmental systems
- Natural ecosystems

## Detection Engine Responsibilities

This repository specifically handles:
- Detecting and identifying image-based AR markers sent from user's device captured image
- Return detected marker name or false in JSON format

## Tech Stack

- **Python 3**
- **OpenCV** (for image processing and marker recognition)
- Flask as the API server 
- Platform-agnostic architecture (designed for web deployment)

## Installation

```bash
python app_aips.py
```

## License

This project is licensed under the MIT License.  
See the [`LICENSE`](./LICENSE) file for full details.
