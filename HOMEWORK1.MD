## HW 1: Pytorch, Pytorch Lightning, Model monitoring, Hyper-parameter optimization

The goal of this homework will be to familiarize yourself with Pytorch Lightning which serves as a tool for organization of all-things related to training (this is a semi-quote) and perform hyper-parameter optimization.
(Including model monitoring with the help of a chosen model monitoring tool). 

If it comes to Pytorch Lightning this will be one time, where it is required to use this particular package without option to chose any other, as I feel there are no other alternatives, (at least for Pytorch). Of course I'm open to proving me wrong.

### Example flow to complete the homework:

* Prepare a Dataset class and Dataloader class using Pytorch for a arbitrary dataset of your choice.
* Prepare a training script of any chosen model in Pytorch Lightning. To achieve that you need to create two modules: Data Module and Lightning Module 
* Add model monitoring using built-in Pytorch Lightning callback for a chosen monitoring tool, for instance Wandb or MLFlow.
* Validate Model Learning: Confirm the model does in-fact learn by comparison of training to validation loss curve.
* Implement optimization of any chosen (or group of parameters) using a chosen optimization package. (I recommend Optuna or RayTune)

### Additional organization information:
* If possible, please don’t copy outright whole solutions, like the one linked below.
* Please upload your code to a repository. 
* Feel free to use models from any of your previous or future (like Master's degree) projects.

### Acceptance Criteria:
Very quick presentation (no slides!!) showcasing:
* Purpose of your model (Machine learning task, data type)
* Lightning Module
* Main problems encountered (for others to learn from your mistakes)
* Results of training in chosen monitoring tool with training and validation curves
* Results of hyper-parameter optimization: Showcasing what parameter have you optimized and how it improved validation metrics?

This homework yields 1 point. (Passing grade 3 points, max points: 5)

### Deadline:
* Classes on 31.03.2026 
Submitting the homework late yields one time penalty of 0.25 points.

### Useful links:
* https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning 
* https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html 
* https://freedium.cfd/https://pub.aimind.so/optimizing-hyperparameters-with-optuna-a-hands-on-tutorial-bb8e2ffd1801  
* https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py 
* https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html 
