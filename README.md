<h1>Balanced kfold</h1>

The implementation of "UNO: Underwater Non-Natural Object dataset" k-folding.

<h2>Details</h2>

A script that splits a video dataset into k well-balanced folds for object detector nested cross-validation purposes.

The goal is to split n videos into k groups, such as the total number of frames and the total number of labels of each group are approximatively equal. Moreover, every frame belonging to one video has to be part of a unique group.

At this time, it only supports the YOLOv5 label format (https://github.com/ultralytics/yolov5/issues/2293) with a unique class.

<h2>Citation</h2>
Soon available.

<br>

<h2>Prerequisite</h2>
```bash
conda create --name balanced_kfold --file requirements.txt
```
<h2>Usage</h2>

<h3>K-folding results of the paper</h3>
```bash
python balancedKFolding.py --input_dir dummy_UNO_dataset --output_dir uno_kfold --k 5 --ite 100000000
```

<h3>K-folding results for random video dataset</h3>
1. Generate a random video dataset<br>
```bash
python generateDummyVideoDataset.py --video_nr 100 --frames_mean 50 --frames_std 30 --labels_mean 10 --labels_std 30
```
2. Create k-folds<br>
```bash
python balancedKFolding.py
```



