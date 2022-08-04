#!/bin/bash

# Check that the core_exp script produces no runtime errors for a selection of
# configs involving ResNets.
# This script is separate from test_core_exp.sh as ResNet training may not run
# on resource constrained systems.

declare -a yaml_files=("")

for yaml_file in "${yaml_files[@]}"
do
    yaml_path="tests/configs/$yaml_file"
    printf "\nRunning $yaml_path\n"
    python core_exp.py --yaml_file $yaml_path -wboff "$@" || exit
done

