
## Initial Data Split

src/dataprep

put code inside above folder.

data/train.csv

pass the above data file in yaml file.

create 3 splits, train, val, test

will integrate the train and val splits with huggingface trainer.

will use the test split for inference


## Coding Requirements

Class Based.
Hydra Yaml config.
Outputs inside hydra run dir
add a limit to the config that will tell how many rows of data to work with.
-1 or not being provided should mean that i want to work with full data.