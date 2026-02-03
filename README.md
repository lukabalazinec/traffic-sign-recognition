### Traffic Sign Recognition
This project implements a traffic sign recognition system based on deep learning using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system is built around a convolutional neural network (CNN) trained to classify traffic signs from images and video streams. In addition to model training and evaluation, the project includes graphical user interfaces for recognizing traffic signs from individual images, folders of images, video files, and live webcam input. The model is trained using supervised learning with image preprocessing, region-of-interest extraction, data normalization, and progressive data augmentation techniques. The application demonstrates practical usage of computer vision and machine learning concepts, including image classification, real-time inference and user-friendly visualization of predictions.

### Instructions for Running the Application
1. Clone the repository or download it as a ZIP archive
2. Create and activate a Python virtual environment: `python -m venv .venv` and `source .venv/bin/activate`
3. Install the required Python packages (see Dependencies): `python -m pip install tensorflow numpy pandas opencv-python scikit-learn matplotlib pillow tkinterdnd2`
4. Download the GTSRB dataset from Kaggle and extract it so `Train.csv`, `Test.csv`, `Meta.csv` and the `Train/`, `Test/`, `Meta/` folders are in the project root. Dataset page: `https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`
5. Run training with `python projekt.py` (this will save `traffic_sign_model.h5`)
6. Run the GUI demo with `python predict_gui.py` or the video demo with `python video_recognition.py`

### Required Dependencies
- Python 3.8+
- TensorFlow (2.x)
- NumPy
- Pandas
- OpenCV
- scikit-learn
- Matplotlib
- Pillow
- tkinter (usually included with Python on Windows)
- tkinterdnd2 (for drag-and-drop support)
