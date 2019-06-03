from train import Trainer
import os
from PIL import Image
from pathlib import Path
from utils import *
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Config:
    lr = 0.00001
    batch_size = 64 
    step = 0
    epoch = 10
    epoch_num = 0
    triplet_alpha = 0.2
    beta = 0.3
    use_pca = False
    data_list_file = Path('../dfw_prot/triplet_list_random6.txt')
    dataset_path = Path('/data/xiangyang/dataset/dfw_112x112')
    masked_matrix = Path('../dfw_prot/test_matrix.txt')
    threshold_num = 400
    #save_matrix :save distance matrix
    save_matrix = True
    save_root = Path('/home/xiangyang/dfw/model/9955/random6_alpha={}_beta={}'.format(triplet_alpha,beta))

    #model_path = Path('/home/xiangyang/dfw/model/alpha=0.3_beta=0/epoch:5.pth')
    model_path = Path('./model.pth')
    print_freq = 5
    evaluate_able = True


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.train()
    #trainer.evaluate()