# Set virtual environment
conda create -n sensim
eval "$(conda shell.bash hook)"
conda activate sensim

# Install required libraries
conda install tensorflow
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
