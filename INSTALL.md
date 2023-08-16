# Installation

This repository is built in PyTorch 2.0.1 and tested on Window 11 environment (Python3.11.4, CUDA11.8, cuDNN8.9.3).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/alsco1234/Image_Denoising.git
cd Restormer
```

2. Make conda environment
```
conda create -n restormer python=3.7
conda activate restormer
```

3. Install dependencies
```
conda env create --file environment.yml 
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```
