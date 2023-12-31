# Multimodal Representation 

Adjustment of https://arxiv.org/abs/1810.10191 papers code for "IN2107 Seminar on Robotics Science and Systems Intelligence" of TUM

Code for Making Sense of Vision and Touch. 
https://sites.google.com/view/visionandtouch

Code was written by: Matthew Tan, Michelle Lee, Peter Zachares, Yuke Zhu

Adjusted and extended by: Johannes Decker

## requirements
The original list of required packages is contained in `requirements.txt`.

To get this repo up and running conda-forge/miniforge package management was used\
(Installation: https://kirenz.github.io/codelabs/codelabs/miniforge-setup/#0).

The packages and their versions listed in `requirements.txt` have been updated manually to their newest version to execute the training process on macOS with an apple silicon arm chip (which is my working device).

All the finally used packages are listed in the `2_environment.yml`. After conda was set up successfully, this file can be used to recreate the conda environment for the execution of the training process.

`conda env create --name envname --file=2_environment.yml`

(Replace envname with the name you want to give the conda environment)

Maybe the used packages in the `2_environment.yml` must be adjusted if you want to execute the training process on other systems/chips such as Windows, Linux, x64, and x86.

## structure of the repo
The repo provides two different ways to execute the training process:
1. Via the distributed file structure with `mini_main.py` as the main file, what is basically the original adjusted repository
2. Via the `SUPERFILE.py` file, it contains all the relevant code for the training process of the model in one single file

The distributed file structure and the `SUPERFILE.py` file contain exactly the same code. The `SUPERFILE.py` file was created only for the purpose of adjusting and extending the repository to be able to define and train the model with a varying size of the first dimension of the latent space vector.\
For both ways to execute the training process, a configuration file exists in the `multimodal/configs/` directory. They are identical to the file paths that are specified in these files.\
The model can be trained with varying sizes of the first dimension of the latent space vector (depth) by changing the `zdepth` parameter in the config file. The default size of the first dimension of the latent space vector is 2. In the cause of the design of the model `zdepth` can be changed to even numbers greater than zero only.

## conda environment
`conda activate envname`

To turn back to your standard (base) environment: `conda deactivate`

## get dataset
```
cd multimodal/dataset
./download_data.sh
```

`cd ..`

## run training
`python mini_main.py --config configs/training_default.yaml`

OR

`cd ..`\
`python SUPERFILE.py --config multimodal/configs/SUPERFILE_training_default.yaml`

## visualization of results
The `Tensorboard` package was used to see how well the trained model performs compared to models with varying sizes of the first dimension of the latent space vector.

`tensorboard --logdir logging` or `tensorboard --logdir multimodal/logging/`

Then you can find the results at `http://localhost:6006/`

## OPTIONAL: Check System settings for number of allowed open files
`launchctl limit maxfiles` 

outputs: maxfiles    SOFTLIMIT       HARDLIMIT

## OPTIONAL: IF first execution of the training process failed because of "OSError: [Errno 24] Too many open files"
--> Set systems soft limit to hard limit (influences the limit for the current shell session only)

`ulimit -n 'ulimit -Hn'`

## To check if it worked, use the following command:
`ulimit -n`

## ROBOT DATASET
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
proprio                  Dataset {50, 8}
