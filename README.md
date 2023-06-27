# SAAM
using transformer and segmentation anything for realtime segmentation

# Requirements
- Python 3.8
- CUDA 11.1+

# Installation
```bash
conda env create -f environment.yaml
conda activate saam
python -m spacy download en_core_web_sm

cd ..
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .; cd ../Semantic-Segment-Anything
```
Dowload pth of SAM

```bash
mkdir ckp && cd ckp
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```
# Configuration
all default configurations are stored in default_config.yml and config.py


# RUN SAAM
```bash
python scripts/sam_transfomer.py --data_dir data/room/  --seg_model oneformer --dataset ade20k --convert_to_rle --using_sam --classes 23 30 31
```
you can choose which class  needs to be filtered out after segmenetation by setting '--classes'

using pytorch multiprocessing using multiple GPUs 
```bash
python scripts/sam_transfomer_mp.py --data_dir data/room/  --seg_model oneformer --dataset ade20k --convert_to_rle --using_sam --classes 23 30 31 --num_gpus 3
```