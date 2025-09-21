# cnn-for-mammograms
Deep learning and CNNs for X-rays mammograms for breast cancer prediction

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
3. Create a virtual environment
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
pip install jupyterlab
```
6. Lunch jupyter lab
```bash
jupyter lab
```
7. Open http://localhost:8888/ in web browser.
8. Inside jupyter lab you can see the directory tree. All jupyter notebooks can be run in isolation. Inside the Utils, all the functions and code modulation is stores for experimentation.
9. The last jupyter file "P4_Inference" can be used in isolation to make predictions. It uses the project optimal model stored in Models/VGGModel (only this model was kept due to storage constarins. However the models can be run and stored locally for further experimentation.    



