# Monoloco

> We tackle the fundamentally ill-posed problem of 3D human localization from monocular RGB images. 
Driven by the limitation of neural networks outputting point estimates, 
we address the ambiguity in the task with a new neural network  predicting confidence intervals through a 
loss function based on the Laplace distribution. 
Our architecture  is a  light-weight feed-forward  neural network which  predicts the 3D  coordinates given 2D human pose.  
The design is particularly well suited for small training data and cross-dataset generalization. 
Our experiments show that (i) we outperform state-of-the art results on KITTI and nuScenes datasets, 
(ii) even outperform stereo for far-away pedestrians,  and (iii) estimate meaningful confidence intervals. 
We further share insights on our model of uncertainty in case of limited observation and out-of-distribution samples.

```
@article{bertoni2019monoloco,
  title={MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation},
  author={Bertoni, Lorenzo and Kreiss, Sven and Alahi, Alexandre},
  journal={arXiv preprint arXiv:1906.06059},
  year={2019}
}
```
The article is available on [ArXiv](https://arxiv.org/abs/1906.06059v1)

A video with qualitative results is available on [YouTube](https://www.youtube.com/watch?v=ii0fqerQrec)

![overview_paper](docs/pull.png)
# Setup

### Install
Python 3 is required. Python 2 is not supported. 
Do not clone this repository and make sure there is no folder named monoloco in your current directory.

`pip install monoloco`

Live demo is available, we recommend to try our **Webcam** functionality. More info in the webcam section.


For development of the monoloco source code itself, you need to clone this repository and then:
```
pip install openpifpaf nuscenes-devkit tabulate
```
Python 3.6 or 3.7 is required for nuScenes development kit. Python 3 is required for openpifpaf. 
All details for Pifpaf pose detector at [openpifpaf](https://github.com/vita-epfl/openpifpaf).



### Data structure

    Data         
    ├── arrays                 
    ├── models
    ├── kitti
    ├── nuscenes
    ├── logs
    

Run the following to create the folders:
```
mkdir data
cd data
mkdir arrays models kitti nuscenes logs
```

### Pre-trained Models
* Download a MonoLoco pre-trained model from 
[Google Drive](https://drive.google.com/open?id=1F7UG1HPXGlDD_qL-AN5cv2Eg-mhdQkwv) and save it in `data/models` 
(default) or in any folder and call it through the command line option `--model <model path>`
* Pifpaf pre-trained model will be automatically downloaded at the first run. 
Three standard, pretrained models are available when using the command line option 
`--checkpoint resnet50`, `--checkpoint resnet101` and `--checkpoint resnet152`.
Alternatively, you can download a Pifpaf pre-trained model from [openpifpaf](https://github.com/vita-epfl/openpifpaf)
 and call it with `--checkpoint  <pifpaf model path>`


# Interfaces
All the commands are run through a main file called `main.py` using subparsers.
To check all the commands for the parser and the subparsers (including openpifpaf ones) run:

* `python3 -m monoloco.run --help`
* `python3 -m monoloco.run predict --help`
* `python3 -m monoloco.run train --help`
* `python3 -m monoloco.run eval --help`
* `python3 -m monoloco.run prep --help`
or check the file `monoloco/run.py`
              
# Predict
The predict script receives an image (or an entire folder using glob expressions), 
calls PifPaf for 2d human pose detection over the image
and runs Monoloco for 3d location of the detected poses.
The command `--networks` defines if saving pifpaf outputs, MonoLoco outputs or both.
You can check all commands for Pifpaf at [openpifpaf](https://github.com/vita-epfl/openpifpaf).


Output options include json files and/or visualization of the predictions on the image in *frontal mode*, 
*birds-eye-view mode* or *combined mode* and can be specified with `--output_types`


### Ground truth matching
* In case you provide a ground-truth json file to compare the predictions of MonoLoco,
 the script will match every detection using Intersection over Union metric. 
 The ground truth file can be generated using the subparser `prep` and called with the command `--path_gt`.
 Check preprocess section for more details or download the file from 
 [here](https://drive.google.com/open?id=1F7UG1HPXGlDD_qL-AN5cv2Eg-mhdQkwv).
 
* In case you don't provide a ground-truth file, the script will look for a predefined path. 
If it does not find the file, it will generate images
with all the predictions without ground-truth matching.

Below an example with and without ground-truth matching. They have been created (adding or removing `--path_gt`) with:
`python3 -m monoloco.run predict --networks monoloco --glob docs/002282.png --output_types combined --scale 2 
--model data/models/monoloco-190513-1437.pkl --n_dropout 100 --z_max 30`
 
With ground truth matching (only matching people):
![predict_ground_truth](docs/002282.png.combined_1.png)

Without ground_truth matching (all the detected people): 
![predict_no_matching](docs/002282.png.combined_2.png)

### Images without calibration matrix
To accurately estimate distance, the focal length is necessary. 
However, it is still possible to test Monoloco on images where the calibration matrix is not available. 
Absolute distances are not meaningful but relative distance still are. 
Below an example on a generic image from the web, created with:
`python3 -m monoloco.run predict --networks monoloco --glob docs/surf.jpg --output_types combined --model data/models/monoloco-190513-1437.pkl --n_dropout 100 --z_max 25`

![no calibration](docs/surf.jpg.combined.png)


# Webcam
<img src="docs/webcam_short.gif" height=350 alt="example image" />

MonoLoco can run on personal computers with only CPU and low resolution images (e.g. 256x144) at ~2fps.
It support 3 types of visualizations: `front`, `bird` and `combined`.
Multiple visualizations can be combined in different windows.

The above gif has been obtained running on a Macbook the command:

`python3 -m monoloco.run predict --webcam --scale 0.2 --output_types combined --z_max 10 --checkpoint resnet50`

# Preprocess

### Datasets

#### 1) KITTI dataset
Download KITTI ground truth files and camera calibration matrices for training
from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and
save them respectively into `data/kitti/gt` and `data/kitti/calib`. 
To extract pifpaf joints, you also need to download training images soft link the folder in `
data/kitti/images`

#### 2) nuScenes dataset
Download nuScenes dataset from [nuScenes](https://www.nuscenes.org/download) (either Mini or TrainVal), 
save it anywhere and soft link it in `data/nuscenes`


### Annotations to preprocess
MonoLoco is trained using 2D human pose joints. To create them run pifaf over KITTI or nuScenes training images. 
You can create them running the predict script and using `--networks pifpaf`.

### Inputs joints for training
MonoLoco is trained using 2D human pose joints matched with the ground truth location provided by
nuScenes or KITTI Dataset. To create the joints run: `python3 -m monoloco.run prep` specifying:
1. `--dir_ann` annotation directory containing Pifpaf joints of KITTI or nuScenes. 

2. `--dataset` Which dataset to preprocess. For nuscenes, all three versions of the 
dataset are supported: nuscenes_mini, nuscenes, nuscenes_teaser.

### Ground truth file for evaluation
The preprocessing script also outputs a second json file called **names-<date-time>.json** which provide a dictionary indexed
by the image name to easily access ground truth files for evaluation and prediction purposes.



# Train
Provide the json file containing the preprocess joints as argument. 

As simple as `python3 -m monoloco.run --train --joints <json file path>`

All the hyperparameters options can be checked at `python3 -m monoloco.run train --help`.

### Hyperparameters tuning
Random search in log space is provided. An example: `python3 -m monoloco.run train --hyp --multiplier 10 --r_seed 1`.
One iteration of the multiplier includes 6 runs.


# Evaluation
Evaluate performances of the trained model on KITTI or Nuscenes Dataset.
### 1) nuScenes
Evaluation on nuScenes is already provided during training. It is also possible to evaluate an existing model running
`python3 -m monoloco.run eval --dataset nuscenes --model <model to evaluate>`

### 2) KITTI
### Baselines
We provide evaluation on KITTI for models trained on nuScenes or KITTI. We compare them with other monocular 
and stereo Baselines: 

[Mono3D](https://www.cs.toronto.edu/~urtasun/publications/chen_etal_cvpr16.pdf), 
[3DOP](https://xiaozhichen.github.io/papers/nips15chen.pdf), 
[MonoDepth](https://arxiv.org/abs/1609.03677) and our 
[Geometrical Baseline](monoloco/eval/geom_baseline.py).

* **Mono3D**: download validation files from [here](http://3dimage.ee.tsinghua.edu.cn/cxz/mono3d) 
and save them into `data/kitti/m3d`
* **3DOP**: download validation files from [here](https://xiaozhichen.github.io/) 
and save them into `data/kitti/3dop`
* **MonoDepth**: compute an average depth for every instance using the following script 
[here](https://github.com/Parrotlife/pedestrianDepth-baseline/tree/master/MonoDepth-PyTorch) 
and save them into `data/kitti/monodepth`
* **GeometricalBaseline**: A geometrical baseline comparison is provided. 
The best average value for comparison can be created running `python3 -m monoloco.run eval --geometric`

#### Evaluation
First the model preprocess the joints starting from json annotations predicted from pifpaf, 
runs the model and save the results
in txt file with format comparable to other baseline. 
Then the model performs evaluation.

The following graph is obtained running:
`python3 -m monoloco.run eval --dataset kitti --generate --model data/models/monoloco-190513-1437.pkl 
--dir_ann <folder containing pifpaf annotations of KITTI images>`
![kitti_evaluation](docs/results.png)

