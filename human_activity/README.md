# Experiments with NODE-based models on Human Activity

## Training
The code must be run from main directory of the repository, not from this directory. For example, if your working directory is ```human_activity```, first go up one level
```
cd ..
```
Then run the training process using this command:
```
python human_activity/run_models.py --gpu <choose your GPU number> --dataset activity --node <model-name>
```
For example, if you want to run the experiment with NesterovNODE with GPU 0 and let the training log file name be nesterovnode, then the command is as follows:
```
python human_activity/run_models.py --gpu 0 --dataset activity --ode-rnn --node NesterovNODE --save "output"
```
Modify this command if you want to run with other models.

# Visualization
The code for plotting the results is in ```visualization/Human_visualization.ipynb```. If the training process of the model is done successfully, follow the cells in this ```.ipynb``` file to reproduce the plots.