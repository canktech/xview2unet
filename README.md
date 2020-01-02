### AI FOR HUMANITARIAN ASSISTANCE AND DISASTER RELIEF 
### SEGMENTATION AND CLASSIFICATION OF BUILDING DAMAGE 

### OPENSOURCE XVIEW2 SUBMISSION USING MODIFIED UNET (MORE STAGES AND PAIRED IMAGE INPUTS) 

### TESTED ON UBUNTU 18.04, PYTHON 3.7.5, USING RTX GPU WITH HIGH VRAM

### Get XView2 Data

Untar train.tar and tier3.tar and test.tar from https://xview2.org/dataset

Arrange data as follows:

```bash
data
├── test
│   └── images
└── train
    ├── images1024
    ├── labels1024
    └── targets1024
```

### INSTRUCTIONS 

You may need to modify batch sizes in ```trainlocunet.py``` and ```traindamgeunet.py```

```bash
pip install requirements.txt 
python preprocess.py
python trainlocunet.py
python traindamageunet.py
tree workspace # to see your commits
python testdamage.py
```

Results will be in 
```bash
results
├── predictions
└── vizpredictions
```

### Score
Just missed out on top 50 leaderboard despite joining the competition very late and only made submissions on last day

(weighted overall, loc, dmg)  .68 / .78 / .63
