
# Perceiving Humans: from Monocular 3D Localization to Social Distancing

> Perceiving humans in the context of Intelligent Transportation Systems (ITS) 
often relies on multiple cameras or expensive LiDAR sensors. 
In this work, we present a new cost- effective vision-based method that perceives humans’ locations in 3D 
and their body orientation from a single image.
We address the challenges related to the ill-posed monocular 3D tasks by proposing a deep learning method 
that predicts confidence intervals in contrast to point estimates. Our neural network architecture estimates 
humans 3D body locations and their orientation with a measure of uncertainty. 
Our vision-based system (i) is privacy-safe, (ii) works with any fixed or moving cameras,
 and (iii) does not rely on ground plane estimation. 
 We demonstrate the performance of our method with respect to three applications: 
 locating humans in 3D, detecting social interactions, 
 and verifying the compliance of recent safety measures due to the COVID-19 outbreak. 
 Indeed, we show that we can rethink the concept of “social distancing” as a form of social interaction 
 in contrast to a simple location-based rule. We publicly share the source code towards an open science mission.

## Preprocessing

# Kitti
Annotations from a pose detector needs to be stored in a folder.
For example by using [openpifpaf](https://github.com/vita-epfl/openpifpaf):
```
python -m openpifpaf.predict \
--glob "<kitti images directory>/*.png" \
--json-output <directory to contain predictions> 
--checkpoint=shufflenetv2k30 \
--instance-threshold=0.05 --seed-threshold 0.05 --force-complete-pose 
```
Once the step is complete:
`python -m monstereo.run prep --dir_ann <directory that contains predictions> --monocular`


### Collective Activity Dataset
To evaluate on of the [collective activity dataset](http://vhosts.eecs.umich.edu/vision//activity-dataset.html)
 (without any training) we selected 6 scenes that contain people talking to each other. 
 This allows for a balanced dataset, but any other configuration will work. 

THe expected structure for the dataset is the following:

    collective_activity         
    ├── images                 
    ├── annotations
    
where images and annotations inside have the following name convention:

IMAGES: seq<sequence_name>_frame<frame_name>.jpg
ANNOTATIONS: seq<sequence_name>_annotations.txt

With respect to the original datasets the images and annotations are moved to a single folder 
and the sequence is added in their name. One command to do this is:

`rename -v -n 's/frame/seq14_frame/'  f*.jpg`

which for example change the name of all the jpg images in that folder adding the sequence number
 (remove `-n` after checking it works)

Pifpaf annotations should also be saved in a single folder and can be created with:

```
python -m openpifpaf.predict \
--glob "data/collective_activity/images/*.jpg"  \
--checkpoint=shufflenetv2k30 \
--instance-threshold=0.05 --seed-threshold 0.05 --force-complete-pose\
--json-output /data/lorenzo-data/annotations/collective_activity/v012 
```

Finally, to evaluate activity using a MonoLoco++ pre-trained model trained either on nuSCENES or KITTI:
```
python -m monstereo.run eval --activity \ 
--net monoloco_pp --dataset collective \
--model <MonoLoco++ model path>  --dir_ann <pifpaf annotations directory>
```

