# Coronaries-Arteries-Diseases-Weakly supervised Learning based method 
this porject is supported by Jordan University of science and Technology , alonge side with this research we explored a paper that introduced [Multi-instance Learning approach based on Transformer](https://arxiv.org/abs/2106.00908) in our mission we Re-design PPGE method for more efficient training by improvig **Convolution with Fast Fourier Transform** which called the method Fast Fourier Postional encoding **FFPE**

**Notation:** the implementation still under progres as long as we are trying to collect dataset of Coronaries-Arteries-Diseases
now tried to test the approach on Data from [Kaggle RSNA Screening Mammography Breast Cancer Detection](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge)

* Setup the ENV:
     - Create the environment 

            conda create --name TransFFT-MIL python=3.6
    - install the requirements
    
            pip install -r requirements.txt  

* Run the code :
    -  training model </br>
        **Note** in our experiment we Re-Developed two approaches based Positional Encodings methods **FFTPEG** and **FF_ATPEG**
        that can be changed in TransFFPEG.py file 

```python
     python train.py --stage 'train' --gpus 0 --Epochs 200
```       
