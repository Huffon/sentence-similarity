# Set virtual environment
conda create -n test2
eval "$(conda shell.bash hook)"
conda activate test2

# Install required libraries
conda install tensorflow
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
