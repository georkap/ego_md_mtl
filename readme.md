# Multi-dataset Multitask Egocentric Action Recognition

Code for paper Multi-dataset Multitask Egocentric Action recognition (https://ieeexplore.ieee.org/document/9361177)

## Abstract
For egocentric vision tasks such as action recognition, there is a relative scarcity of labeled data. This increases the risk of overfitting during training. In this paper, we address this issue by introducing a multitask learning scheme that employs related tasks as well as related datasets in the training process. Related tasks are indicative of the performed action, such as the presence of objects and the position of the hands. By including related tasks as additional outputs to be optimized, action recognition performance typically increases because the network focuses on relevant aspects in the video. Still, the training data is limited to a single dataset because the set of action labels usually differs across datasets. To mitigate this issue, we extend the multitask paradigm to include datasets with different label sets. During training, we effectively mix batches with samples from multiple datasets. Our experiments on egocentric action recognition in the EPIC-Kitchens, EGTEA Gaze+, ADL and Charades-EGO datasets demonstrate the improvements of our approach over single-dataset baselines. On EGTEA we surpass the current state-of-the-art by 2.47%. We further illustrate the cross-dataset task correlations that emerge automatically with our novel training scheme.

## Requirements
torch==1.6.0  
dsntnn==0.5.3  
opencv_python==4.4.0.44  
matplotlib==3.3.1  
scipy==1.5.0  
numpy==1.19.1  
pandas==0.24.2  
torchvision==0.7.0  
scikit_learn==0.24.1  

## Citation
If you do, please cite our paper. 

G. Kapidis, R. Poppe and R. C. Veltkamp, "Multi-Dataset, Multitask Learning of Egocentric Vision Tasks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2021.3061479.

@ARTICLE{9361177,  
  author={G. {Kapidis} and R. {Poppe} and R. C. {Veltkamp}},  
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
  title={Multi-Dataset, Multitask Learning of Egocentric Vision Tasks},  
  year={2021},  
  volume={},  
  number={},  
  pages={1-1},  
  doi={10.1109/TPAMI.2021.3061479}}  
