#!/bin/sh
#$ -cwd
#$ -j y
#$ -N steel
#$ -jc gpu-container_g1.72h
#$ -ac d=nvcr-tensorflow-1810-py3
chmod 664 /home/iygnoh/Severstal/input/severstal-steel-defect-detection/train.csv
export PATH=/home/iygnoh/miniconda3/envs/myenv/bin:$PATH
cd src


for i in 1234 1235 1236 1237 1238
do
	python main.py --sch 1 --loss 2 --output 2 --augment 2 -e 48 --seed $i
done

