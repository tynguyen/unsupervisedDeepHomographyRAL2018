### Model trained on Synthetic Data
Download at 
```bash 
https://drive.google.com/drive/folders/1Y9oNgbJTrAdkgf5-T1xONtU9n2ZqwDta?usp=sharing
```
Then, store the synthetic_models to current folder


### Model trained on Aerial Image Data
Download at 
```bash 
https://drive.google.com/drive/folders/16RI7R0EVayiXfYoP2Ahhl4yN2sWhG76Z?usp=sharing
```
Then, store the synthetic_models to current folder. \ 

Note: you need to format your image data in a correct size in order to make use of this trained model. Please refer to section "Generate aerial dataset" in README to get how to format the raw images. 


## Test model with synthetic dataset
### Supervised
```bash 
python homography_synthetic.py --mode test --lr 5e-4 --loss_type h_loss 
``` 

### Unsupervised
```bash 
python homography_synthetic.py --mode test --lr 1e-4 --loss_type l1_loss
``` 
