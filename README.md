[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/decorrelated-adversarial-learning-for-age/age-invariant-face-recognition-on-cacdvs)](https://paperswithcode.com/sota/age-invariant-face-recognition-on-cacdvs?p=decorrelated-adversarial-learning-for-age)
# Decorrelated-Adversarial-Learning

This is an unofficial pytorch implementation for the 2019 CVPR paper 'Decorrelated Adversarial Learning for Age-Invariant Face Recognition' from Tencent AI Lab'

## How to train on your own dataset

If you want to start fresh training, follow the procedures.
(or use the backbone and train with your own scritps)

### Prepare your data
#### Folder structure
Since it's under pytorch framework, your dataset should follow ImageFolder requirements. Furthermore, age label for every image is needed. That is, the dataset should look like
```
 path/to/dataset/train/
                       id0/              
                          000_[age].jpg
                          001_[age].jpg
                          ...
                       id1/              
                          000_[age].jpg
                          001_[age].jpg
                          002_[age].jpg
                          ...
```
where `[age]` is the age of that face image. If true age labels are not available, DEX age prediction network can be used to predict apparent ages.

#### Image sizing

The paper (and this implementation) uses (h, w) = (112, 96). Any face alignment method will do.
 
### Add your dataset metainfo in meta.py
```
your_dataset_name = {
    'train_root': '/data/your_dataset_name/train',      # root dir for training set
    'val_root': '/data/your_dataset_name/val',          # root dir for validation set (optional, set to None if not available).
    'pat': '_|\.',                                      # pattern for re.split(), which split 000_12.jpg to [000, 12, jpg]
    'pos': 1,                                           # index for `[age]` after split
    'n_cls': 82                                         # number of classes
}
```
### Modify main.py
Open main.py, import your dataset from meta.py, and change `dataset` to your dataset. Change `save` to the dir where you want to save your trained weights. By default, weights are saved every epoch.

### Off we go
Run the following cmd to start training:
`python3 main.py ctx epochs /path/to/model_weights`
where
```
ctx: device id, ctx < 0 for cpu, ctx-th gpu otherwise.
epochs: how many epochs to train.
/path/to/model_weights: path to trained model weights if you want to resume training.
```
## Performance
I trained on a dataset composed by many public cross-age datasets and VGG-face2 general dataset. Best verification accuracy on CSCD_VS was 98.8%.

## Keeps updating...
