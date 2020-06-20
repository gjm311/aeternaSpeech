#!/bin/bash
#SBATCH --job-name=getSpecFull
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/%x-%j-on-%N.out
#SBATCH -e /home/%u/%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

export WORKON_HOME==/cluster/ag61iwyb/.python_cache

pip3 install --user -r requirements.txt 

python3 get_spec_full.py "/tedx_spanish_corpus/speech/"

