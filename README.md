# neural-feature-extraction

contact: agnes.luhtaru@ut.ee or slwanna@utexas.edu

## Setup Instructions
1. Find the `tf_and_torch_env.yml` file and point the `prefix` path to your conda installation.
- ```$ conda env create -f tf_and_torch_env.yml```
- ```$ conda activate tf-and-torch-gpu```

Make sure you link the cuda install of conda for tensorflow. You will have to do this
for every new terminal session you begin.

- ```$ export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"``` 
- ```$ make dev_install```

## How to Run
This package assumes you also have the ```dataset-download-scripts``` package. This is because line 33 in this package's ```__main__.py``` references the ```metadata.json``` file in the ```dataset-download-scripts``` package.

## Before you commit!

1. Ensure you code "works" otherwise save to an appropriate branch
2. run ```$ make format``` 
3. run ```$ make lint```   