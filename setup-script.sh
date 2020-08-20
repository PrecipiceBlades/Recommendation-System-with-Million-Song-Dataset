#!/bin/bash
set -x -e

sudo yum -y install git htop
cd /home/hadoop
git clone https://github.com/YiRuitao/10605_Group8.git
cd 10605_Group8
sudo bash dependencies.sh
