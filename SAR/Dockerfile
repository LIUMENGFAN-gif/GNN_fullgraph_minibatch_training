# Use the NVIDIA CUDA image as a parent image
#nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
#nvidia/cuda:12.4.0-devel-ubuntu22.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Install curl
RUN apt-get update && apt-get install -y curl && apt-get install sudo -y && sudo apt install nano && sudo apt install net-tools

# Install Miniconda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh && \
    $HOME/miniconda/bin/conda clean --all && \
    ln -s $HOME/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc 


# Update Python in the base Conda environment to 3.8
RUN /root/miniconda/bin/conda install -y python=3.8

# Install packages based on the existing environment.yml file
RUN /root/miniconda/bin/conda env update -f /workspace/environment_default.yml --prune

# Make RUN commands use the new environment
SHELL ["/root/miniconda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]

# Activate the environment on container startup
CMD ["/bin/bash"]