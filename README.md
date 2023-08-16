
# Fine-Tuning Restormer for Medical Image Denoising

The origin model is [Restormer](https://github.com/swz30/Restormer)   
Fine-tuned by [alsco1234](https://github.com/alsco1234/)

<hr />

> With the recent development of deep learning technology, a method of removing noise from medical images using neural networks is required. In particular, a technology is needed to remove real noise generated by the irregular flow of electrons in X-ray images without artifacts. Therefore, in this project, the Restoration Transformer (Restoration Transformer) model, which showed excellent performance in the grayscale image denoising task, was improved and applied to medical images. Unlike CNN, Restormer calculates not only local information but also global information, and has the advantage of not limiting the input size. However, there is a disadvantage that the size of the model is large. Considering this, this project reduced the number of blocks of Transformer during fine-tuning and learned only the attention layer, allowing only noise to be removed without creating artifacts.
<hr />

## Network Architecture

<img src = "https://i.imgur.com/ulLoEig.png"> 

## Results
Experiments are performed for different image processing tasks including, image deraining, single-image motion deblurring, defocus deblurring (both on single image and dual pixel data), and image denoising (both on Gaussian and real data). 

<details>
<summary><strong>Image Deraining</strong> (click to expand) </summary>

<img src = "https://i.imgur.com/mMoqYJi.png"> 
</details>

<details>
<summary><strong>Single-Image Motion Deblurring</strong> (click to expand) </summary>

<p align="center"><img src = "https://i.imgur.com/htagDSl.png" width="400"></p></details>

<details>
<summary><strong>Defocus Deblurring</strong> (click to expand) </summary>

S: single-image defocus deblurring.
D: dual-pixel defocus deblurring.

<img src = "https://i.imgur.com/sfKnLG2.png"> 
</details>


<details>
<summary><strong>Gaussian Image Denoising</strong> (click to expand) </summary>

Top super-row: learning a single model to handle various noise levels.
Bottom super-row: training a separate model for each noise level.

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/4vzV8Qy.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/Sx986Xs.png" width="500"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Grayscale</b></p></td>
    <td><p align="center"><b>Color</b></p></td>
  </tr>
</table>
</details>

<details>
<summary><strong>Real Image Denoising</strong> (click to expand) </summary>

<img src = "https://i.imgur.com/6v5PRxj.png">
</details>

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.

## Test

To test the Fine-Tuned Restormer models of 255*255 one patch
```
python test_if_training.py
```
To test the Fine-Tuned Restormer models of one image
```
python prepare_dataset.py
python test_one.py
python blending.py
```

## Training
To Training no pretraiend weight
```
python train.py
```
To Fine-tuning with pretraiend weight
```
python Fine-tuning.py
```


## Citation

    @inproceedings{Zamir2021Restormer,
        title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
                and Fahad Shahbaz Khan and Ming-Hsuan Yang},
        booktitle={CVPR},
        year={2022}
    }


## Contact
Should you have any question, please contact alsco4321@gmail.com


## My Related Works
- Grayscale Image Denoising | [Code](https://github.com/alsco1234/Image_Denoising) 