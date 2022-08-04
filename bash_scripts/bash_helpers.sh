
regularization_list() {
  # Genearates a list of regularization parameters for model- or layer-wise
  # experiments
  #
  # Args:
  # $1: (str) reg_type; one of "layer" or "model"
  # $2: (float) value of target_density or lmbda
  # $3: (int) number of layers in the model

  if [ "$1" = "model" ];
  then
    # Model-wise case (only one number)
    _reg_list=$2
  else
    # Layer-wise case
    _reg_list=""
    for i in $(seq $3);
    do
      _reg_list="$_reg_list $2"
    done
  fi

  echo $_reg_list
}


mnist_regularization_list() {
  # Genearates a list of regularization parameters for model- or layer-wise
  # experiments
  #
  # Args:
  # $1: (str) reg_type; one of "layer" or "model"
  # $2: (float) value of target_density or lmbda
  # $3: (str) model_type; one of "MLP" or "LeNet"

    # Set up tdst
  if [ "$1" = "model" ];
  then
      # Model-wise case (only 1 tdst)
      _reg_list="$2"
  else
      # Separate (layer-wise) case (one tdst per layer)
      if [ "$3" = "LeNet" ];
      then
          _reg_list="$2 $2 $2 $2"
      else
          _reg_list="$2 $2 $2"
      fi
  fi

  echo $_reg_list
}


load_modules_and_env() {
  # Loads SLURM modules and Python environment
    module load python/3.8
    source ~/venvs/constl0/bin/activate
}

create_wandb_arg(){
  # Creates string arg for WandB config
  #
  # Args:
  # $1: (str) $use_wand
  # $2: (str) $run_group
  # $3: (str) $wandb_dir
  
  if [ "$1" = "True" ]; then
      wandb_arg="--run_group $2 --wandb_dir $3"
  else
      wandb_arg="-wboff --wandb_dir $3"
  fi
  echo $wandb_arg
}