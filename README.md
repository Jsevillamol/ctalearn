# CTALearn Keras
A proof-of-concept project reimagining the architectural design of the [CTLearn project](https://github.com/ctlearn-project/ctlearn).

Major changes with respect to the original repository are:

* Use of the Keras API for training and prediction
* Use of Pandas for managing the data spread across multiple files
* Stronger division of the model building, training and prediction responsibilities
* the ImageMapper and DataProcessor classes have been hidden from the user

![UML diagram](CTALearn.png)

# Installing the CTAlearn framework

1. Install the prerequisites
2. Download this repo
3. Run `pip install upgrade .` from the root folder

## Basic functionality: How to build and train a model

1. Create a folder to be your training folder, cd to it
2. Create a `config.yml` file. This file should have the following sections:
      * `model_config`, specifying how the model should be built
      * `data_config`, specifying what kind of data to load and how to preprocess it
      * `train_config`, specifying training options
   You can check the params to be specified in each section in [`example_config.yaml`](configuration_examples/example_config.yaml).
3. Run `python PATH/TO/PROJECT/ctalearn/train.py config.yml`. The outputs of a training session are:
      * `model.h5` file containing the results of the training
      * `model_summary.txt` file containing a summary of the model architecture
      * `session.log` file containing a log of messages during training
      * `training_history.csv` containing the training history
      * `logs` folder containing tensorboard events. Read them running `tensorboard --logdir=logs` from the training folder.
      * `png` plots of the training history progress

### Training a preexisting model

Do as before, but run `python PATH/TO/PROJECT/ctalearn/train.py config.yml PATH/TO/MODEL/model.h5`, 
where `PATH/TO/MODEL/model.h5`is the relative path to the model you want to use for training.

### Performing grid search

CTALearn has built-in capabilities to enable grid search hyperparameter search.

To use it, just add the prefix `multi_` to the parameters where you want to perform grid search 
in the `yaml` config file, and specify a range of values to try via a list.

The training script will automatically recognize the `multi_` options and perform grid search, 
generating as many folders as possible combinations of parameters are possible.

Each folder will contain the config file specifying what parameters were used for that run, 
plus the usual output of a run.

Additionally, the script will generate a summary of all the run results and comparative `png` plots.

### Generating the pixel_pos files
In order to work properly, CTALearn relies on the generation of some files containing
information needed to perform the mapping from the raw image of the telescopes to 2D images.