# neural-feature-extraction

contact: agnes.luhtaru@ut.ee or slwanna@utexas.edu

## Setup Instructions
- ```$ conda env create -f environment.yml```
- ```$ conda activate neural-feature-extraction```
- ```$ make dev_install```

## How to Run
This package assumes you also have the ```dataset-download-scripts``` package. This is because line 33 in this package's ```__main__.py``` references the ```metadata.json``` file in the ```dataset-download-scripts``` package.

## Before you commit!

1. Ensure you code "works" otherwise save to an appropriate branch
2. run ```$ make format``` 
3. run ```$ make lint```   