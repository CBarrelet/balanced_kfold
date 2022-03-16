<h1>Balanced kfold</h1>

The implementation of "UNO: Underwater Non-Natural Object dataset" k-folding.

Paper link: (Soon available)<br>
Website link: (Soon available)

<h2>Details</h2>

A script that splits a video dataset into k well-balanced folds for object detector nested cross-validation purposes.

The goal is to split n videos into k groups, such as the total number of frames and the total number of labels of each group are approximatively equal. Moreover, every frame belonging to one video has to be part of a unique group.

At this time, it only supports the <a href="https://github.com/ultralytics/yolov5/issues/2293">YOLOv5 label format</a> with a unique class.

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
python balancedKFolding.py  --input_dir dummy_UNO_dataset
                            --output_dir uno_kfold
                            --k 5 # Number of folds
                            --ite 100000000 # Number of iterations
```

<h3>K-folding results for random video dataset</h3>

1. Generate a random video dataset<br>
```bash
python generateDummyVideoDataset.py --video_nr 100    # Number of videos
                                    --frames_mean 50  # Frames mean per video
                                    --frames_std 30   # Frames standard deviation per video
                                    --labels_mean 10  # Labels mean per frame
                                    --labels_std 30   # Labels standard deviation per frame
```
2. Create k-folds<br>
```bash
python balancedKFolding.py
```



