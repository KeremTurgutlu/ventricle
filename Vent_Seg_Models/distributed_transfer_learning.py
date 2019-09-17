######################################################
###### SCRIPT FOR DISTRIBUTED TRANSFER LEARNING ######
######################################################

from fastai.vision import *
from fastai.callbacks import *
from fastai.script import *
from fastai.distributed import *
import data_utils
from data_utils import *
from models import *
from learn_utils import *

def annealing_epochs(n_groups, epochs):
    "Generate number of epochs"
    for i in range(n_groups):
        yield int(max(1, annealing_cos(epochs,0,i/(n_groups))))      
        
class SaveModelCallback(TrackerCallback):
    "SaveModelCallback modified for distributed transfer learning"
    def __init__(self, learn:Learner, monitor:str='val_loss', mode:str='auto', every:str='improvement', name:str='bestmodel', best_init=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name = every,name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        if best_init: self.best = best_init 
      
    def on_train_begin(self, **kwargs:Any)->None:
        "Initializes the best value."
        if not hasattr(self, 'best'):
            self.best = float('inf') if self.operator == np.less else -float('inf')
#         print('best init score:', self.best) 
        
    def jump_to_epoch(self, epoch:int)->None:
        try: 
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save(f'{self.name}')


@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
    MODEL_NAME:Param("Name for saving model", str)='TL_Brain_MR_Baseline_10',
    model_dir:Param("Directory to save model", str)='tl_brain_mr_models',
    data_name:Param("data name", str)='notl_brain_mr',
    tl_model_dict_name:Param("dict name for tl data mapping", str)='tl_brain_mr_model_dict',   
    bs:Param("batch size per GPU", int)=2,
    model_name:Param("model name", str)='baseline1',
    loss_func:Param("loss func", str)='dice',
    lr:Param("learning rate", float)=3e-3,
    epochs:Param("number of epochs", int)=100,
    one_cycle:Param("do one cycle or general sched", int)=1,
    early_stop:Param("do early stopping", int)=1,
    clip:Param("do gradient clipping", float)=0.,
    sample_size:Param("Number of samples in training", int)=None,
    load_dir:Param("directory to load pretrained model", str)='atlas_brain_mr_models',
    eps:Param("Adam eps", float)=1e-8, 
    lsuv:Param("do lsuv init", int)=0):
    
    """Distrubuted training of a given experiment.
    Fastest speed is if you run as follows:
        python ../../fastai/fastai/launch.py ..args.. ./distributed_training.py
       
    """
    # distributed init
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    
    # data
    f = data_dict[data_name]
    train_paths, valid_paths, test1_paths, test2_paths = f()
    if sample_size:
        idxs = np.random.choice(range(len(train_paths[0])), size=sample_size, replace=False)
        train_paths = np.array(train_paths)[:,idxs]
    
    train_ds = MRI_3D_Dataset(*train_paths)
    valid_ds = MRI_3D_Dataset(*valid_paths)
    test1_ds = MRI_3D_Dataset(*test1_paths) if test1_paths else None
    test2_ds = MRI_3D_Dataset(*test2_paths) if test2_paths else None

    data = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds, bs=bs)
    test1_dl = DeviceDataLoader(DataLoader(test1_ds), device=data.device) if test1_ds else None
    test2_dl = DeviceDataLoader(DataLoader(test2_ds), device=data.device) if test2_ds else None

    # model
    f = experiment_model_dict[model_name]; m = f()
    apply_leaf(m, partial(cond_init, init_func= nn.init.kaiming_normal_))
    
    # INFO
    if not int(gpu): print(f"Training: {MODEL_NAME} Model: {m.__class__} Train Size: {len(train_paths[0])}")
        
    # callbacks
    early_stop_cb = partial(EarlyStoppingCallback, monitor='dice_score', mode='max', patience=5)
    save_model_cb = partial(SaveModelCallback, monitor='dice_score', mode='max', every='improvement',  name=f'best_of_{MODEL_NAME}')
    reduce_lr_cb = partial(ReduceLROnPlateauCallback, monitor='dice_score', mode='max', patience=0, factor=0.8)
    csv_logger_cb = partial(CSVLogger, filename=f'logs/{model_dir}/{MODEL_NAME}')

    if early_stop: 
        if not int(gpu): print('early_stop: True')
        callback_fns = [early_stop_cb, save_model_cb, reduce_lr_cb, csv_logger_cb, ActivationStats]
    else: 
        if not int(gpu): print('early_stop: False')
        callback_fns = [save_model_cb, csv_logger_cb, ActivationStats]
    
    callback_fns = [save_model_cb] # problem with finetuning
    callbacks = [TerminateOnNaNCallback()]    
    
    # https://github.com/pytorch/pytorch/issues/8860
    learn = Learner(data=data, model=m, opt_func=partial(optim.Adam, betas=(0.9,0.99), eps=1e-4),
                    callbacks=callbacks, callback_fns=callback_fns, model_dir=model_dir)
    learn.loss_func = {'dice':dice_loss, 'bce':BCEWithLogitsFlat(), 'mixed':MixedLoss(10., 2.)}[loss_func] 
    if not int(gpu): print('Loss func:', learn.loss_func)
    learn.metrics = [dice_score]
    learn.to_distributed(gpu)
    if clip:
        print("clipping gradients with norm:", clip)
        learn.to_fp16(dynamic=True, clip=clip, max_scale=2*16) # numerical instability?
    else:
        learn.to_fp16(dynamic=True)
    
    # load old model
    tl_model_dict = getattr(data_utils, tl_model_dict_name)
    old_model_name = tl_model_dict[MODEL_NAME]
    if not int(gpu): print(f"Loading old: {old_model_name}")
    actual_model_dir = learn.model_dir
    learn.model_dir = load_dir # in order to load the atlas model
    learn.load(old_model_name)
    learn.model_dir = actual_model_dir
    
    # check data
#     if not int(gpu): print(learn.data.valid_ds.img_fnames[0])
    
    # check model split
    f = model_split_dict[model_name]
    learn.split(f)
    n_groups = len(learn.layer_groups)
    if not int(gpu): print("Number of layer groups:", n_groups)
           
            
    # finetuning: decide n_epochs with annealing
    # problem: can't save best model with callback? - pickle error occurs randomly?
    # bad solution: only save and load manually the best model so far
    # problem is with learn.load(purge=False)
    tl_epochs = annealing_epochs(n_groups, epochs)
    
    #Fine tuning with low lr=3e-3 improves
    for i in list(range(1, n_groups)):
        _epochs = next(tl_epochs)
        if not int(gpu): print(f"Finetuning to layer: {-i} epochs: {_epochs}")
        learn.freeze_to(-i)
        learn.fit_one_cycle(_epochs, slice(lr))
        best_init = learn.save_model_callback.best
        # TODO: load with opt
#         learn.load(f'best_of_{MODEL_NAME}')
        learn.callback_fns = [cb_fn for cb_fn in learn.callback_fns if cb_fn.func == Recorder]
        learn.callback_fns.append(partial(save_model_cb, best_init=best_init))

        
    learn.unfreeze()
    _epochs = next(tl_epochs)
    if not int(gpu): print(f"Finetuning all layers epochs: {_epochs}")
    learn.fit_one_cycle(_epochs, slice(lr))
    
   
    # Evaluate on test
    from time import time
    if not gpu:
        # load best model 
        learn.load(f'best_of_{MODEL_NAME}')
        os.makedirs('test_results', exist_ok=True)
        os.makedirs(f"test_results/{model_dir}", exist_ok=True)
        os.makedirs(f"test_results/{model_dir}/{MODEL_NAME}", exist_ok=True)
        
        bs = 1
        model = learn.model.eval().to(torch.device(gpu))
        test1_dl = DeviceDataLoader(DataLoader(test1_ds, batch_size=bs),
                                    tfms=[batch_to_half], device=torch.device(gpu)) if test1_ds else None
        test2_dl = DeviceDataLoader(DataLoader(test2_ds, batch_size=bs),
                                    tfms=[batch_to_half], device=torch.device(gpu)) if test2_ds else None

        # test1
        inputs = []
        targets = []
        for xb, yb in test1_dl:
            zb = model(xb).detach().cpu(); inputs.append(zb.float())
            targets.append(yb.detach().cpu().float())

        inputs, targets = torch.cat(inputs).float(), torch.cat(targets).float()
        test1_res = dice_score(inputs, targets).item()

        # test2
        inputs = []
        targets = []
        for xb, yb in test2_dl:
            zb = model(xb).detach(); inputs.append(zb)
            targets.append(yb.detach())

        inputs, targets = torch.cat(inputs).float(), torch.cat(targets).float()
        test2_res = dice_score(inputs, targets).item()

        print(test1_res, test2_res)
        
        save_fn = f"test_results/{model_dir}/{MODEL_NAME}/{str(int(time()))}.txt"
        print(f"saving results to: {save_fn}")
        with open(save_fn, 'w') as f: f.write(str([test1_res, test2_res]))

    else: pass









        
        
