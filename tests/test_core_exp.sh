#!/bin/bash

# Check that the core_exp script produces no runtime errors for a selection of
# configs found in tests/configs.

declare -a yaml_files=(\
    "lr/co.yml"\
    "lr/pen.yml"\
    "mlp/co_layer.yml"\
    "mlp/co_model.yml"\
    "mlp/pen_layer.yml"\
    "mlp/pen_model.yml"\
    "lenet/co_layer.yml"\
    "lenet/co_model.yml"\
    "lenet/pen_layer.yml"\
    "lenet/pen_model.yml"\
    )

for yaml_file in "${yaml_files[@]}"
do
    yaml_path="tests/configs/$yaml_file"
    printf "\nRunning $yaml_path\n"
    python core_exp.py --yaml_file $yaml_path -wboff "$@" || exit
done
