set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=lingua_250623 

# Create the conda environment

conda create -n $env_prefix python=3.11 -y 
conda activate $env_prefix 

echo "Currently in env $(which python)"

# Install packages
pip install torch==2.7.0 xformers
pip install ninja
pip install --requirement requirements.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"
