#!/bin/bash

# List of values for the model parameter
model_values=("clip" "sbert" "sonar" "use")

# List of values for the dataset parameter
dataset_values=("alfred" "bridge" "rt1" "scout" "tacoplay")

# Loop through the model and dataset values and modify config.ini each time
for model in "${model_values[@]}"
do
  for dataset in "${dataset_values[@]}"
  do
    # Modify the config.ini file (assuming parameters are 'model_name' and 'dataset_name')
    sed -i "s/^model_name = .*/model_name = $model/" config.ini
    sed -i "s/^dataset_name = .*/dataset_name = $dataset/" config.ini
    
    # Run the Python script
    python -m neural_feature_extraction
    
    # Optionally: you can echo the parameters to track the process
    echo "Ran neural_feature_extraction with model_name: $model and dataset_name: $dataset"
  done
done
