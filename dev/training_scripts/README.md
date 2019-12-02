### Training Script Arguments

### Training in De Novo - From Scratch Mode

You can either run  `[notebook run]` cells in `3b) traindenovo.ipynb` to do training within the notebook (which eseentially calls training_scripts/traindenovo.py from notebook). This is the suggested way for training since we it is tested but you can 
also take `training_scripts/traindenovo.py` script and run from the terminal.

#### Parameters

Eventhough notebook's `script` part is very self-explanatory here is a list of parameters you can pass for `traindenovo.py`:

`data_name:` Data name for experiment, valid args are `notl_brain_mr`, `notl_brain_ct`, `notl_ventricle_mr`, `notl_ventricle_ct`, `atlas_brain_mr`, `atlas_ventricle_mr`

sample_size:Param("Random samples for training, default None - full", int)=None,

seed:Param("Random seed for sample_size", int)=None,

bs:Param("Batch size for training", int)=4,

model_name:Param("Model architecture config - baseline*", str)="baseline1",

MODEL_NAME:Param("Model name to save the model", str)="NOTL_Brain_MR_Baseline_1",

model_dir:Param("Directory to save model", str)="notl_brain_mr_models",

loss_func:Param("Loss function for training", str)='dice',

eps:Param("Eps value for Adam optimizer", float)=1e-8,

epochs:Param("Number of epochs for training", int)=2,

lr:Param("Learning rate for training", float)=0.1)

