# this seems necessary for the activate call to work
eval "$(conda shell.bash hook)"
# Create environment name based on the exercise name
mamba create -n segmentation python=3.10 -y
mamba activate segmentation
# Install additional requirements
mamba install -c pytorch -c nvidia --file requirements.txt -y
# Build the notebooks
sh prepare-exercise.sh

# Download and extract data, etc.
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp" -O kaggle_data.zip && rm -rf /tmp/cookies.txt
gdown -O kaggle_data.zip 1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp
unzip -u -qq kaggle_data.zip && rm kaggle_data.zip

# Download and extract data for the instance segmentation exercise
gdown -O kaggle_data_instance.zip 1I-TLOwwwNVdNc-AKjUw799j2A6cP-qaZ 
unzip -u -qq kaggle_data_instance.zip && rm kaggle_data_instance.zip


# Return to base environment
mamba deactivate

