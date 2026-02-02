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
CONFIG_FILE=./configjson/SCPINN.json

# ========================
# Run Training Script
# ========================
# --initvelpath: Initial velocity model
# --inittimetablepath: Observed travel times
# --expiter: Experiment iteration number (for naming results)
# --config_file: JSON config file path
# --truevelpath: True velocity model for testing
# --stackprofilepath: Input seismic profile
# --SaveEpoch: Epochs to save model checkpoints
# --amp: Enable Automatic Mixed Precision for faster training
python ./ApplyScripts/TrainSCPINN.py \
    --initvelpath $INITVELPATH \
    --inittimetablepath $LABELTIMETABLE \
    --expiter 0 \
    --config_file $CONFIG_FILE \
    --truevelpath $TRUEVELPATH \
    --stackprofilepath $INITSTACKPROFILE \
    --SaveEpoch 0 50 150 200 300 350 399 \
    --amp