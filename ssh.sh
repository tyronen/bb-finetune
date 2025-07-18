#!/usr/bin/env bash
# run like `source ssh.sh` on tmp runpod, after sending your private key and this script

# ensure we have git, clone repo, cd in etc.
apt-get update
apt-get install -y vim rsync git nvtop htop tmux curl ca-certificates git-lfs lsof nano less cargo rustc swig ninja-build build-essential

# start ssh agent, add key, go to /workspace
eval "$(ssh-agent -s)"
ssh-add .ssh/id_ed25519
cd /workspace

mkdir -p bb-finetune
cd bb-finetune
pip install --upgrade pip
