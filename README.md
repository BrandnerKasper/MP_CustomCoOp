# MP_CustomCoOp
Tweaking [CoOp](https://github.com/KaiyangZhou/CoOp) for a multilangual object class detection. 
Therefore, we like to use other pretrained models from [open clip](https://github.com/mlfoundations/open_clip) and 
use other datasets more suited for a multilangual setting. 

## Setup
This repository builds upon [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).
Here are the steps to get you going:
1. Clone the Dassl repo
```
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
```
2. Create a conda environment
```
conda create -y -n dassl python=3.10
```
3. Activate the environment
```
conda activate dassl
```
4. Install torch (requires version >= 1.8.1) and torchvision (Please refer to https://pytorch.org/ if you need a different cuda version)
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
5. Install dependencies
```
pip install -r requirements.txt
```
6. Install this library (no need to re-build if the source code is modified)
```
python setup.py develop
```
7. Switch to this repository and install dependecies
```
cd ..
cd MP_CustomCoOp
pip install -r requirements.txt
```

## Dataset
For the moment only the Caltech dataset was tested, the original repository supported multiple datasets, including imagenet, food101, eurosat, oxford pets etc.
To get this running with Caltech simply:
- Create a folder named `caltech-101/` under `$DATA`. (`$DATA` the data folder can be placed where ever you like.)
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

(For other datasets look into the CoOp repo and the DATASETS.md`)

## How to run
At the moment this repository only supports CoOp (and not CoCoOp!).
Two examples of how to use this repo: (ALWAYS SUBSTITUTE `$DATA` with either the absolute path to the datafolder on your system or relativ to this folder!)
1. With the original clip models (to get the same results as in the CoOp paper):
```
python train.py $DATA rn50
```
2. With the new open clip models:
```
python train.py $DATA roberta-vit-b32 --shots 16 --open_Clip --pretrained laion2b_s12b_b32k
```
You can also use the same models as the original clip but from the open clip repository
```
python train.py $DATA rn50 --open_Clip 
```
(Note for the same trained models we do not need to set a pretrained tag, since it defaults to openai)

!! Checkout the list available models script to check which models with which pretrained tags are supported!!
```
python list_available_models.py 
```

## Evaluation
Substituting the clip models with "the same" open clip models we get the following results:

To evaluate your model simply call:
```
python eval.py 
```
This evaluates all files in the output folder.
In our case we got the following results:
```
Test and train accuracy with standard deviation of different runs.
open_clip_roberta_vit_b32_openai_1.txt test: 36.6% +- 3.8% and train: 63.2% +- 33.4
open_clip_xlm_roberta_base_vit_b_32_openai_1.txt test: 38.3% +- 2.3% and train: 64.6% +- 34.9
clip_rn_50_openai_1.txt test: 83.3% +- 3.1% and train: 87.3% +- 9.8
clip_vit_b_32_openai_1.txt test: 87.4% +- 0.9% and train: 90.4% +- 7.4
open_clip_vit_b_32_openai_1.txt test: 73.6% +- 5.9% and train: 75.7% +- 27.5
open_clip_vit_b_32_laion2b_s34b_b79k.txt test: 85.1% +- 1.5% and train: 87.4% +- 20.0
clip_rn_101_openai_1.txt test: 85.4% +- 3.1% and train: 88.1% +- 8.9
open_clip_rn_50_openai_16.txt test: 88.5% +- 0.7% and train: 58.4% +- 25.5
open_clip_rn_101_openai_1.txt test: 80.2% +- 1.2% and train: 86.0% +- 20.1
open_clip_rn_50_openai_1.txt test: 4.5% +- 0.0% and train: 3.6% +- 3.0
open_clip_roberta-vit-b32_laion2b_s12b_b32k_16.txt test: 88.6% +- 0.5% and train: 70.4% +- 18.0
open_clip_vit_b_16_openai_1.txt test: 86.1% +- 2.1% and train: 87.4% +- 16.5
clip_vit_b_16_openai_1.txt test: 89.7% +- 1.8% and train: 90.5% +- 7.2
```