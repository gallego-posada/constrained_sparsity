# Controlled Sparsity via Constrained Optimization

## About

This repository contains the official implementation for the paper [Controlled Sparsity via Constrained Optimization or: How I Learned to Stop Tuning Penalties and Love Constraints](https://arxiv.org/abs/2208.04425). This code enables the training of sparse neural networks using *constrained* $L_0$ regularization.

We use the [Cooper](https://github.com/cooper-org/cooper) library for implementing and solving the constrained optimization problems.

## Setup

Create an environment using your favorite tool and install the required packages.

```pip install -r requirements.txt```

### Dataset paths

The `utils/paths.json` file must be set up with the location of datasets before
running. See [Utils](#utils) below for details.

### Configs

We provide YAML files containing all the hyper-parameters choices made for each of the experiments presented in the paper. You can find these files under the [configs](#configs) folder. In the [Examples](#examples) section, we demonstrate how to use these configs to trigger runs.

### Weights and Biases

If you don't have a Weights and Biases account, or prefer not to log metrics to 
their servers, you can use the flag `-wboff`.

## Examples

**Constrained** experiments can be triggered by providing the `--target_density`
(or `-tdst`) arg, along with a) 1 number for model-wise constraints, or
b) 1 number for each of the sparsifiable layers of the network. The target 
densities are expected to be floats between 0 and 1. **Penalized** experiments are triggered in a similar way using the `--lmbdas` arg.

### Constrained WideResNet-28-10 on CIFAR-10

```
python core_exp.py --yaml_file configs/defaults/cifar_defaults.yml -tdst 0.7
```

### Penalized ResNet18 on TinyImageNet

```
python core_exp.py --yaml_file configs/defaults/tiny_imagenet_r18_defaults.yml --lmbdas 1e-3
```

## Project structure

```
constrained_l0
├── bash_scripts
├── configs
├── custom_experiments
├── get_results
├── sparse
├── tests
├── utils
├── core_exp.py         # Script used for running experiments
├── l0cmp.py            # Constrained minimization problem
├── README.md
├── requirements.txt
```

### Sparse
The `sparse` folder contains the main components used to construct $L_0$ sparsifiable networks.

Our implementation of $L_0$-sparse models is based on the [code](https://github.com/AMLab-Amsterdam/L0_regularization) by Louizos et al. for the paper *Learning Sparse Neural Networks through L0 Regularization*, ICLR, 2018.

Fully connected and convolutional $L_0$ layers are found inside of `l0_layers.py`, as well as `L0BatchNorm`, a batch normalization layer compatible with output sparsity layers. The modules `models.py`, `resnet_models.py` and `wresnet_models.py` implement various models composed both of standard Pytorch layers and $L_0$ layers. These include MLPs, LeNet5s, WideResNet28-10s, ResNet18s and ResNet50s. The code is general enough to be easily extensible to variations of these architectures.

```
├── sparse
│   ├── l0_layers.py
│   ├── models.py
│   ├── resnet_models.py
│   ├── wresnet_models.py
│   ├── purged_models.py
│   ├── purged_resnet_models.py
│   ├── utils.py
│   ...
```
### Utils

`utils` contains various project utils.

```
├── utils
│   ├── imagenet            # Utils for ImageNet dataloader
│   ├── basic_utils.py
│   ├── constraints.py      # Constraint schedulers
│   ├── datasets.py         # Dataloaders
│   ├── exp_utils.py        # Utils for core_exp.py, e.g. train and val loops
│   ├── paths.json
```

The `paths.json` file **must** be setup by the user to indicate the location of folders associated with different datasets. For instance,

```
{
    "mnist": "~/data/mnist",
    "cifar10": "~/data/cifar10",
    "cifar100": "~/data/cifar100",
    "tiny_imagenet": "~/data/tiny-imagenet-200",
    "imagenet": "~/data/imagenet"
}
```
### Configs

`configs` contains YAML files with basic configurations for the experiments presented throughout our paper. For instance, `mnist_defaults.yml` indicates the batch size, learning rates, optimizers and other details used for our MNIST experiments.

These YAML files were designed to be used in conjunction with the scripts in 
the `bash_scripts` folder. Arguments that are required to trigger a run, but 
were *not* specified in the YAML file are marked explicitly. You can find these
values in the corresponding `bash_scripts` file, as well as the appendix in the
paper.

```
├── configs
│   ├── defaults
│   │   ├── mnist_defaults.yml
│   │   ├── cifar_defaults.yml
│   │   ├── imagenet_defaults.yml
│   │   ...
```

### Custom Experiments

We implement two experiments which serve as baselines for comparison with our constrained $L_0$ approach. These are:

* `bisection`: training a model to achieve a pre-defined sparsity target via the 
penalized approach. The search over penalization coefficients $\lambda_{pen}$ is 
done via the bisection method.
* `pretrained`: comparison with magnitude pruning starting from a 
Pytorch-pretrained `torchvision.models.resnet50`. Also includes a loop for  
fine-tuning the remaining weights.

```
    ├── custom_experiments
    │   ├── bisection
    │   ├── pretrained
    │   │   ├── resnet_magnitude_pruning.py
```

### Tests

Automated tests implemented on Pytest are included in the `tests` folder. Besides the automated tests (runnable from the root folder with `python -m pytest`), we provide YAML files under the `configs` folder to test the `core_exp.py` script for different model architectures for 2 epochs. These must be triggered manually, as described in the [Examples](#examples) section.

```
├── test
│   ├── helpers
│   ├── configs
│   ...
```

### Bash scripts

We executed our experiments on a computer cluster with Slurm job management. The bash scripts used for triggering these runs are contained in the `bash_scripts` folder.

The experiments are split across different subfolders. For instance, the `tiny_imagenet` folder contains scripts for the TinyImagenet control table and plots.

```
├── bash_scripts
│   ├── bash_helpers.sh
│   ├── run_basic_exp.sh
│   ├── mnist
│   ├── cifar
│   ├── tiny_imagenet
│   ├── imagenet
│   ├── detach_gates
│   ├── magnitude_pruning
```

### Results: plots and tables

The `get_results` folder includes scripts for producing the plots and tables found in the paper. These scripts depend on calls to the Weights and Biases API for retrieving logged metrics. 

```
├── get_results
│   ├── neurips             # Scripts for producing tables and plots
│   ├── saved_dataframes    # Tables saved as csv
│   ├── wandb_utils.py      # Utils to access WandB API
```
