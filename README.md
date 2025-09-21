# cnn-for-mammograms
Deep learning and CNNs for X-rays mammograms for breast cancer prediction

# Notes
<b>If you need to experiment with the models or techniques:</b>
The images are not stored in Git Hub due to storage constrains. 
This folder should contained the downloaded data (DICOM, CSVs) files.
Due to the size of the folder containing the DICOM images, the folder is
stored locally. Please dowload the data from:
https://www.cancerimagingarchive.net/collection/cbis-ddsm/
Please note that the CSV files are tracked in GitHub. But, if needed, 
new files can be downloaded as well from the previous website. 

If using the tf.Data.Dataset image iterator(fastest and better for applying preporcessing before resizing), 
you will need to 
1. Decompress images using <b>"P0_Decompress_Data"</b>
2. Apply basic preprocesing to data files using <b>"P1_Basic_Preprocessing"</b>
3. Apply preprocessing to images locally using the set of techniques using ablation or
   by adding one at a time(both in same file) <b>"P2_0_Local_Preprocess"<b>

All models from P2 to P5, except the first model in first iteration, use the tf.Data.Dataset.

## Running Locally and Installation
Make sure you have:
```bash
Python 3.11.11
```

1. Clone repository
```bash
git clone https://github.com/mayramtv/cnn-for-mammograms.git
cd cnn-for-mammograms
```
2. Make sure that you install Git LFS if you want to use the optimal model stored in Git LFS
```bash
git lfs install 
```
4. Create a virtual environment
```bash
virtualenv venv
source venv/bin/activate
```
4. Install packages
```bash
pip install -r requirements.txt
```
6. Install jupyter lab
```bash
pip install jupyter lab
```
6. Lunch jupyter lab
```bash
jupyter lab
```
7. Open http://localhost:8888/ in web browser.
8. Inside jupyter lab you can see the directory tree. All jupyter notebooks can be run in isolation.
Inside the Utils, all the functions and code modulation is stores for experimentation.
The last jupyter file "P4_Inference" can be used in isolation to make predictions. It uses the project
optimal model stored in Models/VGGModel (only this model was kept due to storage constarins.
However the models can be run and stored locally for further experimentation.    



