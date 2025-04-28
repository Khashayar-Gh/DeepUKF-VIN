
# DeepUKF-VIN

 Comprehensive guide for setting up and running the DeepUKF-VIN project.

 Ghanizadegan, K., & Hashim, H. A. (2025). **DeepUKF-VIN: Adaptively-tuned Deep Unscented Kalman Filter for 3D Visual-Inertial Navigation based on IMU-Vision-Net**. _Expert Systems with Applications_, 126656. DOI: [10.1016/j.eswa.2025.126656](https://doi.org/10.1016/j.eswa.2025.126656)

## Requirements

This project requires the following dependencies and setup to run successfully:

### Python Version

- **Python 3.9** is recommended.

### Libraries

The following Python libraries are required:

- **NumPy**
- **Pandas**
- **PyTorch**
- **PyTorch3D**
- **Torchvision**
- **OpenCV (cv2)**
- **Matplotlib**
- **PyYAML**
- **Pickle**
- **Pathlib**
- **Concurrent Futures**
- **Copy**
- **OS and Glob**

### Hardware

- **GPU Support**: A CUDA-enabled GPU is strongly recommended for faster computations.
- **RAM**: At least 8 GB of RAM is recommended. 16 GB is needed if training.

### Operating System

- The code has been tested on **Linux Ubuntu 24.04.1 LTS**

### Download

- Download the large files from [this google drive link](https://drive.google.com/drive/folders/1NZO6AFfWv14sFrqbGstD_rR8Htif0Ab2?usp=sharing")

- Download the dataset files from [V1_02_medium]("http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip") and [V2_02_medium]("http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip"), and unzip them.

- Place all the downloaded files to the same directory as the python scripts.

## How to run

### Train the model

To train the model, run ``` AIEKF_main_VI_all.py ``` by:

```bash
python AIEKF_main_VI_all.py
```

The logfile name will be printed after the training is done. In that file, you'll find the training loss at each epoch, as well as the name of the files containing IMU and Vision Net trained weights. Note that in our testing with an RTX 6000Ada GPU, it takes about 20 hours for 30 epochs.

If you wish to skip this step, refer to ```logfile_AIEKF_Both_10 22 16:48:58_all_cri_0_m3.txt``` as the log file. This is the log file of the model that is used in the paper. Note the trained models are stored at ```imu_model_10 22 23:31:06.pth``` and ```vision_model_10 22 23:31:06.pth```.

### Test the model

To test the model, refer to ```UKF_main_batch_AI_Validation.py```.
If using the pertained model, simply running the model will test the DeepUKF-VIN in ```V1_02_medium``` scenario. If whishing to test the non-AI version, go to line 46 and change the ```use_AI``` flag to False.

```Python
use_AI = False
```

For testing the ```V2_02_medium``` scenario, uncomment lines 43,44, and 371. Comment lines 40, 41 and 372. After modifications, the script should look like:

```Python
    # with open('log_obj_EKF_20241015143753.pkl', 'rb') as file: #for V1_02
    #     loaded_data = pickle.load(file)

    with open('log_obj_UKF_20241111165454.pkl', 'rb') as file: #for V2_02
        loaded_data = pickle.load(file)
    .
    .
    .
    dataset_path = 'V2_02_medium'
    # dataset_path = 'V1_02_medium'
```

Don't forget to adjust ```use_AI``` to use or not use the Adaptation Mechanism.

If wishing to test your own trained models, go to lines 71, and 72, and updated the paths to your own. They are currently set to the trained models used in the paper:

```Python
    vision_model_path = "vision_model_10 22 23:31:06.pth"
    imu_model_path = "imu_model_10 22 23:31:06.pth"
```

After each run, the results are saved in pickle files. The unique name of each pickle file will be printed at the end of each run. If you wish to skip this step, you may use the file names already available in the next script.

### Visulaizing and comparing

To reproduce the results presented in the paper, run the ```paper_comparison.py``` script without any change. If you wish to use your own tested results, replace line 12 of the script with your ```V1_02_medium``` scenario result and line 20, with your ```V2_02_medium``` scenario results. They are currently set to the results presented in the paper:

```Python
with open('log_obj_UKF_20241023162924.pkl', 'rb') as file:
    AI_UKF_loaded_data = pickle.load(file)
.
.
.
with open('log_obj_UKF_20241111185410.pkl', 'rb') as file:
    V2_AI_UKF_loaded_data = pickle.load(file)
```

The results presented in tables will be printed out. The graphs will be saved in ```v1_ai_ukf_errors_orientation_position_velocity.png``` and ```trajectory_and_errors_v1_ai_ukf.png```

## TODO
- Improve this README with detailed usage and setup instructions.
- Clean and refactor the code for clarity.

## Citation

If you use this repo, please cite:

```bibtex
@article{ghanizadegan2025deepukf,
  title={DeepUKF-VIN: Adaptively-tuned Deep Unscented Kalman Filter for 3D Visual-Inertial Navigation based on IMU-Vision-Net},
  author={Ghanizadegan, Khashayar and Hashim, Hashim A},
  journal={Expert Systems with Applications},
  pages={126656},
  year={2025},
  publisher={Elsevier}
}
```
