::echo "Installation Started..."
:: create conda environment
conda create -n featuretools-workshop python=3.6 pandas  \
numpy scipy cython matplotlib seaborn scikit-learn joblib --quiet --dry-run
:: activate source
activate featuretools-workshop
:: install additional dependencies
conda install nb_conda jupyter --quiet --dry-run
pip install pandas-profiling featuretools lightgbm

::echo "Installation Finished..."