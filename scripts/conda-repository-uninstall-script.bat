::echo "Uninstalling featuretools-workshop conda environment..."

::# remove conda environment
conda env remove -n featuretools-workshop --quiet --dry-run

::echo "Uninstalling Finished..."