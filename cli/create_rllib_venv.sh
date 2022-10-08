#!/bin/sh
# create and activate rllib venv
conda create -n myenv python=3.7.12 -y
source activate myenv
pip install -U "ray[rllib]"
pip install tensorflow==2.4.0
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3
# replay
pip install pandas-market-calendars

# Note:
# make executable: $chmod u+x create_rllib_venv.sh
# can be executed in terminal with $./create_rllib_venv.sh