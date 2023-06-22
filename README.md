# multimodal_representation
Adjustment of https://arxiv.org/abs/1810.10191 papers code for "IN2107 Seminar on Robotics Science and Systems Intelligence" of TUM

# Multimodal Representation 

Code for Making Sense of Vision and Touch. 
https://sites.google.com/view/visionandtouch

Code written by: Matthew Tan, Michelle Lee, Peter Zachares, Yuke Zhu 

## requirements
`pip install -r requirements.txt`

## get dataset

```
cd multimodal/dataset
./download_data.sh
```

## OPTIONAL: Check System settings for number of allowed open files
`launchctl limit maxfiles`
## outputs: maxfiles    SOFTLIMIT       HARDLIMIT

## OPTIONAL: IF first execution of the training process failed because of "OSError: [Errno 24] Too many open files"
## --> Set systems softlimit to hardlimit (influences the limit for the current shell session only)
ulimit -n `ulimit -Hn`
## To check if it worked, use the follwoing command:
ulimit -n


## run training

`python mini_main.py --config configs/training_default.yaml`


## ROBOT DATASET
----
action                   Dataset {50, 4}\
contact                  Dataset {50, 50}\ 
depth_data               Dataset {50, 128, 128, 1}\
ee_forces_continuous     Dataset {50, 50, 6}\
ee_ori                   Dataset {50, 4}\
ee_pos                   Dataset {50, 3}\
ee_vel                   Dataset {50, 3}\
ee_vel_ori               Dataset {50, 3}\
ee_yaw                   Dataset {50, 4}\
ee_yaw_delta             Dataset {50, 4}\
image                    Dataset {50, 128, 128, 3}\
joint_pos                Dataset {50, 7}\
joint_vel                Dataset {50, 7}\
optical_flow             Dataset {50, 128, 128, 2}\
proprio                  Dataset {50, 8}\
