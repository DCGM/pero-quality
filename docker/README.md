# Docker support

This directory contains basic info about how to run this repository in docker. Dockerfile will be enough for users familiar with docker, and scripts `run.sh` and `build.sh` are here for new users.

## Prerequisites

- [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-engine---community-](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-engine---community-)

- [nvidia docker](https://github.com/NVIDIA/nvidia-docker) support

## Usage

Note: use sudo with docker commands if user is not in `docker` group

1. build image: ./build.sh

2. create and run container: ./run.sh
