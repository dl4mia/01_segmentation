#!/bin/bash

# this seems necessary for the activate call to work
source /localscratch/miniforge3/etc/profile.d/mamba.sh

# Create environment name based on the exercise name
mamba create -n segmentation python=3.10 -y
conda activate segmentation
if [[ "$CONDA_DEFAULT_ENV" == "segmentation" ]]; then
    echo "Environment activated successfully"
    # Install additional requirements
    mamba install -c pytorch -c nvidia -c conda-forge --file requirements.txt -y
else
    echo "Failed to activate the environment"
fi

# Build the notebooks
sh prepare-exercise.sh

# Download and extract data, etc. (Unnecessary because the TAs set it up, but leaving here for records)
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp" -O kaggle_data.zip && rm -rf /tmp/cookies.txt
# gdown -O kaggle_data.zip 1ahuduC_4Ex84R7qKNRzAY6PiLRWX_J3I
# unzip -u -qq kaggle_data.zip && rm kaggle_data.zip


# Return to base environment
mamba deactivate

