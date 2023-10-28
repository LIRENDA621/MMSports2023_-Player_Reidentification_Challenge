

[[Paper]](https://dl.acm.org/doi/10.1145/3606038.3616168)[[Challenge]](http://mmsports.multimedia-computing.de/mmsports2023/challenge.html)

**Please note that the MMSports 2022 championship scheme is our baseline and our code is built based on this [[repository]](https://github.com/DeepSportradar/2022-winners-player-reidentification-challenge/tree/master).**

Code for the paper: `Exploring Loss Function and Rank Fusion for Enhanced Person Re-identification`

## Setup environment
Clone the repo:
  ```bash
  git clone https://github.com/LIRENDA621/MMSports2023_-Player_Reidentification_Challenge.git
  cd MMSports2023_-Player_Reidentificatio_Challenge
  ```  
Create conda environment:
```bash
pip install -r requirements.txt
```


 

## Approach

![CLIP](network_arch.png)

## Usage

Steps for Training and Evaluation:

1. get data: `download_data.py`
2. create DataFrames: `preprocess_data.py`
3. training:
   
   `train_our_post.py`
   
   The **main training script** will output two post-processing results during each verification, one is last year's championship solution, and the other is our implementation. The differences can be found in the paper.

   `train_our_post_resnet.py`

   This training script facilitates the use of the ResNet model to conduct relevant experiments in the paper.
   
5. evaluation: `evaluate.py`
   
7. final predictions:
   
    `predict.py`
   
   
   `predict_fortestset.py`
   
   
   The script is convenient for performing ablation experiments on the test set and does not need to be used.

## Model Fuse

1. similarity fuse: `my_py/sim_fuse.py`

2. dissimilarity fuse: `my_py/dissim_fuse.py`

**In our solution, we first perform similarity fusion and dissimilarity fusion on multiple models respectively, and then perform similarity fusion on the two results.**

All settings are done by the configuration dataclass at the beginning of the scripts.

`download_data.py` downloads and unzips the challenge data from the provided [challenge toolkit](https://github.com/DeepSportRadar/player-reidentification-challenge).
