# Ghost-Free HDR Imaging in Dynamic Scenes via High-Low Frequency Decomposition

It will be continuously updated！



## Pipeline
![pipeline](https://github.com/chengeng0613/HL-HDR_Plus/blob/main/pic/overview.png)




## Usage

### Requirements
* Python 3.7.0
* CUDA 10.0 on Ubuntu 18.04

Install the require dependencies:
```bash
conda create -n hlhdr python=3.7
conda activate hlhdr
pip install -r requirements.txt
```

### Dataset
1. Download the dataset (include the training set and test set) from [Kalantari17's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
2. Move the dataset to `./data` and reorganize the directories as follows:
```
./data/Training
|--001
|  |--262A0898.tif
|  |--262A0899.tif
|  |--262A0900.tif
|  |--exposure.txt
|  |--HDRImg.hdr
|--002
...
./data/Test (include 15 scenes from `EXTRA` and `PAPER`)
|--001
|  |--262A2615.tif
|  |--262A2616.tif
|  |--262A2617.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
|--BarbequeDay
|  |--262A2943.tif
|  |--262A2944.tif
|  |--262A2945.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
```
3. Prepare the corpped training set by running:
```
cd ./dataset
python gen_crop_data.py
```

### Training & Evaluaton
```
cd HL-HDR
```
To train the model, run:
```
python train.py --model_dir experiments
```
To test, run:
```
python fullimagetest.py
```

## Results




## Citation


## Contact
If you have any questions, feel free to contact Genggeng Chen at chengeng0613@gmail.com.

## Checkpoints
The following links are the weights of the Kalantari dataset , the Hu dataset and Tel dataset:https://drive.google.com/drive/folders/1JID13dIOqYdpVbvxZtqeR2bZYigR9eVe?usp=sharing
# HL-HDR_Plus
