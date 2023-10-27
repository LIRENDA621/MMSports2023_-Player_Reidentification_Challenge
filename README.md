

[[Paper]](https://doi.org/10.1145/3552437.3555698)[[Challenge]](http://mmsports.multimedia-computing.de/mmsports2023/challenge.html)

Code for the paper: `CLIP-ReIdent: Contrastive Training for Player Re-Identification`


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


##

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{habel2022clipreident,
  title={CLIP-ReIdent: Contrastive Training for Player Re-Identification},
  author={Habel, Konrad and Deuser, Fabian and Oswald, Norbert},
  booktitle={Proceedings of the 5th International ACM Workshop on Multimedia Content Analysis in Sports (MMSports’22), October 10, 2022, Lisboa, Portugal},
  year={2022}
}

```
