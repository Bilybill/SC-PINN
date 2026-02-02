#!/bin/bash
set -e

# ========================
# File Paths Configuration
# ========================

# Path to the initial velocity model (starting point for inversion)
INITVELPATH=./DATA_FOLDER/linearinit.rsf

# Path to the true velocity model (used for calculating errors/metrics)
TRUEVELPATH=./DATA_FOLDER/marmtrue.rsf

# Path to the observed travel time table (label data)
LABELTIMETABLE=./DATA_FOLDER/labeltimetable.rsf

# Path to the stacked seismic profile (input seismic image)
INITSTACKPROFILE=./DATA_FOLDER/linearstackprofile.rsf

# Configuration file containing hyperparameters (learning rate, epochs, etc.)
# This file will be loaded by the Python script to set experiment parameters
CONFIG_FILE=./configjson/PINN.json

# ========================
# Run Training Script
# ========================
# --inittimetablepath: Path to the observed travel time table used as input/label
# --expiter: Experiment iteration number (used for naming output directories/files)
# --config_file: JSON configuration file path defined above
# --truevelpath: True velocity model path for calculating RMSE/accuracy metrics

python ./ApplyScripts/TrainPINN.py \
    --initvelpath $INITVELPATH \
    --inittimetablepath $LABELTIMETABLE \
    --expiter 0 \
    --config_file $CONFIG_FILE \
    --truevelpath $TRUEVELPATH \
    --SaveEpoch 0 50 150 200 300 350 399