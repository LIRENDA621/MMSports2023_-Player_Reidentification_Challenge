import os
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

from clipreid.model import TimmModel, OpenClipModel
from clipreid.transforms import get_transforms
from clipreid.dataset import TestDataset, ChallengeDataset
from clipreid.evaluator import predict, compute_dist_matrix, compute_scores, write_mat_csv, postprocess_distance
from clipreid.utils import print_line

import timm.models.eva

@dataclass
class Configuration:
    '''
    --------------------------------------------------------------------------
    Open Clip Models:
    --------------------------------------------------------------------------    
    - ('RN50', 'openai')
    - ('RN50', 'yfcc15m')
    - ('RN50', 'cc12m')
    - ('RN50-quickgelu', 'openai')
    - ('RN50-quickgelu', 'yfcc15m')
    - ('RN50-quickgelu', 'cc12m')
    - ('RN101', 'openai')
    - ('RN101', 'yfcc15m')
    - ('RN101-quickgelu', 'openai')
    - ('RN101-quickgelu', 'yfcc15m')
    - ('RN50x4', 'openai')
    - ('RN50x16', 'openai')
    - ('RN50x64', 'openai')
    - ('ViT-B-32', 'openai')
    - ('ViT-B-32', 'laion2b_e16')
    - ('ViT-B-32', 'laion400m_e31')
    - ('ViT-B-32', 'laion400m_e32')
    - ('ViT-B-32-quickgelu', 'openai')
    - ('ViT-B-32-quickgelu', 'laion400m_e31')
    - ('ViT-B-32-quickgelu', 'laion400m_e32')
    - ('ViT-B-16', 'openai')
    - ('ViT-B-16', 'laion400m_e31')
    - ('ViT-B-16', 'laion400m_e32')
    - ('ViT-B-16-plus-240', 'laion400m_e31')
    - ('ViT-B-16-plus-240', 'laion400m_e32')
    - ('ViT-L-14', 'openai')
    - ('ViT-L-14', 'laion400m_e31')
    - ('ViT-L-14', 'laion400m_e32')
    - ('ViT-L-14-336', 'openai')
    - ('ViT-H-14', 'laion2b_s32b_b79k')
    - ('ViT-g-14', 'laion2b_s12b_b42k')
    --------------------------------------------------------------------------
    Timm Models:
    --------------------------------------------------------------------------
    - 'convnext_base_in22ft1k'
    - 'convnext_large_in22ft1k'
    - 'vit_base_patch16_224'
    - 'vit_large_patch16_224'
    - ...
    - https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
    --------------------------------------------------------------------------
    '''

    # Model
    # model: str = ('ViT-L-14', 'openai')   # ('name of Clip model', 'name of dataset') | 'name of Timm model'
    model: str = 'beitv2_large_patch16_224.in1k_ft_in22k_in1k'   # ('name of Clip model', 'name of dataset') | 'name of Timm model'
    remove_proj = True                    # remove projection for Clip ViT models
    
    # Settings only for Timm models
    img_size: int = (224, 224)            # follow above Link for image size of Timm models
    # img_size: int = (448, 448)            # follow above Link for image size of Timm models
    mean:   float = (0.485, 0.456, 0.406) # mean of ImageNet
    std:    float = (0.229, 0.224, 0.225) # std  of ImageNet
    
    # Eval
    batch_size: int = 64                 # batch size for evaluation
    normalize_features: int = True        # L2 normalize of features during eval  
    
    # Split for Eval
    fold: int = -1                        # -1 for given test split | int >=0 for custom folds 
             
    # Checkpoints: str or tuple of str for ensemble (checkpoint1, checkpoint2, ...)
    # checkpoints: str = ("./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth",
    #                     "./model/ViT-L-14_openai/all_data_seed_1/weights_e4.pth")

    # checkpoints: str = 'model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth'
    checkpoints: str = "model/beitv2_large_patch16_224.in1k_ft_in22k_in1k_colorjitter/fold-1_seed_1/weights_e3.pth"
    
    # Dataset
    data_dir: str = "/home/data1/lrd/mmsport/2022-winners-player-reidentification-challenge-master/data_reid"
    
    # show progress bar
    verbose: bool = True 
  
    # set num_workers to 0 if OS is Windows
    num_workers: int = 0 if os.name == 'nt' else 8  
    
    # use GPU if available
    device: str = 'cuda:6' if torch.cuda.is_available() else 'cpu' 

    # postprocess
    k1: int = 20       
    k2: int = 6      
    lamda: float = 0.7
    
    # whether challenge
    challenge: bool = False
    
    dist_csv: str = 'model_fuse/result.csv'
    
#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#  
config = Configuration()

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  
if isinstance(config.model, tuple):

    model = OpenClipModel(config.model[0],
                          config.model[1],
                          remove_proj=config.remove_proj
                          )
    
    img_size = model.get_image_size()
    
    mean=(0.48145466, 0.4578275, 0.40821073)
    std=(0.26862954, 0.26130258, 0.27577711)
       
else:
    model = TimmModel(config.model,
                      pretrained=False)

    img_size = config.img_size
    mean = config.mean
    std = config.std
    
    
dist_matrix_list = []
dist_matrix_rerank_list = []

if not isinstance(config.checkpoints, list) and not isinstance(config.checkpoints, tuple):
    checkpoints = [config.checkpoints]
else:
    checkpoints = config.checkpoints


for checkpoint in checkpoints: 
    
    
    
    #------------------------------------------------------------------------------------------------------------------#  
    # DataLoader                                                                                                       #
    #------------------------------------------------------------------------------------------------------------------#  
    
    # Transforms
    val_transforms, train_transforms = get_transforms(img_size, mean, std)
    
    # Dataframes
    df = pd.read_csv("data_reid/train_df.csv")
    df_challenge = pd.read_csv("data_reid/challenge_df.csv")
     
    if config.fold == -1:
        # Use given test split
        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]
    else:
        # Use custom folds
        df_train = df[df["fold"] != config.fold]
        df_test = df[df["fold"] == config.fold]
        
    
    # Validation
    test_dataset = TestDataset(img_path="./data_reid",
                                     df=df_test,
                                     image_transforms=val_transforms)


    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             shuffle=False,
                             pin_memory=True)
    

    
    #------------------------------------------------------------------------------------------------------------------#  
    # Test                                                                                                             #
    #------------------------------------------------------------------------------------------------------------------# 
    print_line(name="Eval Fold: {}".format(config.fold), length=80)
    
    dist_df = pd.read_csv(config.dist_csv,header=None, skiprows=1, usecols=range(1, 910))
    
    
    # 将数据转换为NumPy数组
    dist_matrix_test_rerank = dist_df.to_numpy(dtype=np.float32)
    
    # save with re-ranking
    print("\nwith re-ranking:")
    mAP = compute_scores(dist_matrix_test_rerank,
                         test_dataset.query,
                         test_dataset.gallery,
                         cmc_scores=True)
    
    
    
    
    