import PyInstaller.__main__
import os
import sys

# Get the path to the Python interpreter being used
python_path = sys.executable

# Define PyInstaller arguments
pyinstaller_args = [
    'GDA_ML.py',  # Your main script
    '--name=General_Data_Reader_&_Analyzer',  # Name of the output executable
    '--onefile',  # Create a single executable file
    '--windowed',  # Don't show console window
    '--add-data={}/*;.'.format(os.path.dirname(python_path)),
    '--hidden-import=numpy',

    '--hidden-import=pandas',
    '--hidden-import=matplotlib',
    '--hidden-import=PyQt5',
    '--hidden-import=matplotlib.backends.backend_qt5agg',
    '--hidden-import=scipy.ndimage',
    '--hidden-import=joblib',
    '--hidden-import=huggingface-hub',
    '--hidden-import=xgboost',
]

# Run PyInstaller
PyInstaller.__main__.run(pyinstaller_args)