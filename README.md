

[[Paper]](https://dl.acm.org/doi/10.1145/3606038.3616168)[[Challenge]](http://mmsports.multimedia-computing.de/mmsports2023/challenge.html)

Code for the paper: `Exploring Loss Function and Rank Fusion for Enhanced Person Re-identification`


## Approach

![CLIP](docs/CLIP-ReIdent.png)

Reformulation of the contrastive language-to-image pre-training approach of CLIP to a contrastive image-to-image training approach using the InfoNCE loss as training objective.



## Usage

Steps for Training and Evaluation:

1. get data: `download_data.py`
2. create DataFrames: `preprocess_data.py`
3. training: `train.py`
4. evaluation: `evaluate.py`
5. final predictions: `predict.py`


All settings are done by the configuration dataclass at the beginning of the scripts.

`download_data.py` downloads and unzips the challenge data from the provided [challenge toolkit](https://github.com/DeepSportRadar/player-reidentification-challenge).
