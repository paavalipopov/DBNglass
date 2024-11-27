# 0. If you just want the Mean MLP model source code
Go to `src/models/mlp.py`.
`MeanMLP` and `default_HPs` is what you need.

# 1. Requirements
```bash
conda create -n mlp_nn python=3.9
conda activate mlp_nn
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Optional
- `prefix`: custom prefix for the project
    - default prefix is UTC time
    - appears in the name of logs directory and the name of WandB project
    - `exp` mode runs with custom prefix will use HPs from `tune` mode runs with the same prefix
        - unless model.default_HP is set to `True`
- `permute`: whether TS models should be trained on time-reshuffled data
    - set to `permute=Multiple` to permute
- `HP_path`: path to custom hyperparams to load
- `follow_splits`: path to an experiment which splist you want to replicate


# `scripts/run_experiments.py` options:
## Required:
- `mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `exp` - experiment mode: run experiments with the best hyperparams found in the `tune` mode, or with default hyperparams `default_HPs` is set to `True`

- `model`: model for the experiment. Models' config files can be found at `src/conf/model`, and their sourse code is located at `src/models`
    - `mlp` - our hero, TS model
    - `lstm` - classic LSTM model for classification, TS model (not used in the paper)
    - `mean_lstm` - `lstm` with LSTM output embeddings averaging, TS model
    - `pe_transformer` - BERT-inspired model, uses transformer endocder, TS model (not used in the paper)
    - `mean_pe_transformer` - `pe_transformer` with encoder output averaging, TS model

    - `dice` - TS model, https://www.sciencedirect.com/science/article/pii/S1053811922008588?via%3Dihub
    - `milc` - TS model, https://arxiv.org/abs/2007.16041 

    - `bnt` - FNC model, https://arxiv.org/abs/2210.06681
    - `fbnetgen` - TS+FNC model, https://arxiv.org/abs/2205.12465
    - `brainnetcnn` - FNC model, https://www.sciencedirect.com/science/article/pii/S1053811916305237
    - `lr` - Logistic Regression, FNC model

- `dataset`: dataset for the experiments. Datasets' config files can be found at `src/conf/dataset`, and their loading scripts are located at `src/datasets`.
    - `fbirn` - ICA FBIRN dataset
    - `fbirn_sex` - ICA FBIRN dataset wtih sex labels
    - `cobre` - ICA COBRE dataset
    - `bsnip` - ICA BSNIP dataset
    - `abide` - ICA ABIDE dataset (not used in the paper)
    - `abide_869` - ICA ABIDE extended dataset
    - `oasis` - ICA OASIS dataset
    - `adni` - ICA ADNI dataset
    - `hcp` - ICA HCP dataset
    - `ukb` - ICA UKB dataset with `sex` labels
    - `ukb_age_bins` - ICA UKB dataset with `sex X age bins` labels

    - `fbirn_roi` - Schaefer 200 ROIs FBIRN dataset
    - `abide_roi` - Schaefer 200 ROIs ABIDE dataset
    - `hcp_roi_752` - Schaefer 200 ROIs HCP dataset

    - `hcp_non_mni_2` - Deskian/Killiany ROIs HCP dataset in ORIG space
    - `hcp_mni_3` - Deskian/Killiany ROIs HCP dataset in MNI space
    - `hcp_schaefer` - Noisy Schaefer 200 ROIs HCP dataset
    - `hcp_time` - ICA HCP dataset with normal/inversed time direcion
