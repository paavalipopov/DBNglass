# If you want to look at glassDBN source code
Go to `src/models/DBNglassFIX.py`.
`glassDBN` is what you need.
The rest of the script is for training the models on different datasets, not all of them were used in the work.

# Requirements
```bash
conda create -n glass python=3.12
conda activate glass
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
```
