##########################################################
###### SCRIPT FOR DISTRIBUTED FROM SCRATCH TRAINING ######
##########################################################

from fastai.vision import *
from fastai.callbacks import *
from fastai.script import *
from fastai.distributed import *
from data_utils import *
from models import *
from learn_utils import *

        
def dice_loss(logits, target, smooth=1.):
    if torch.any(torch.isnan(logits)): print("logits contain nan")
    probas = torch.sigmoid(logits)
    if torch.any(torch.isnan(probas)): print("probas contain nan")
    iflat = probas.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def bce(input, target):
    bs = input.shape[0]
    return F.binary_cross_entropy_with_logits(input.view(bs,-1).float(), target.view(bs,-1).float())

@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
    MODEL_NAME:Param("Name for saving model", str)='NOTL_Brain_MR_Baseline_10',
    model_dir:Param("Directory to save model", str)='notl_brain_mr_models',
    data_name:Param("data name", str)='notl_brain_mr',
    bs:Param("batch size per GPU", int)=2,
    model_name:Param("model name", str)='baseline1',
    loss_func:Param("loss func", str)='dice',
    lr:Param("learning rate", float)=0.1,
    epochs:Param("number of epochs", int)=100,
    one_cycle:Param("do one cycle or general sched", int)=1,
    early_stop:Param("do early stopping", int)=1,
    clip:Param("do gradient clipping", float)=0.,
    sample_size:Param("Number of samples in training", int)=None,
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
        
    # learn
    early_stop_cb = partial(EarlyStoppingCallback, monitor='dice_score', mode='max', patience=5)
    save_model_cb = partial(SaveModelCallback, monitor='dice_score', mode='max', every='improvement', name=f'best_of_{MODEL_NAME}')
    reduce_lr_cb = partial(ReduceLROnPlateauCallback, monitor='dice_score', mode='max', patience=0, factor=0.8)
    csv_logger_cb = partial(CSVLogger, filename=f'logs/{model_dir}/{MODEL_NAME}')

    if early_stop: 
        if not int(gpu): print('early stop=True')
        callback_fns = [early_stop_cb, save_model_cb, reduce_lr_cb, csv_logger_cb, CatchNanGrad, CatchNanActs, CatchNanParameters]
    else: 
        if not int(gpu): print('early stop=False')
        callback_fns = [save_model_cb, csv_logger_cb, CatchNanGrad, CatchNanActs, CatchNanParameters]
    callbacks = [TerminateOnNaNCallback()]    
    
    learn = Learner(data=data, model=m, opt_func=partial(optim.Adam, betas=(0.9,0.99), eps=eps),
                    callbacks=callbacks, callback_fns=callback_fns, model_dir=model_dir)
    learn.loss_func = {'dice':dice_loss, 'bce':BCEWithLogitsFlat(), 'mixed':MixedLoss(10., 2.)}[loss_func] 
    if not int(gpu): print('Loss func', learn.loss_func)
    learn.metrics = [dice_score]
    learn.to_distributed(gpu)
    if clip: learn.to_fp16(dynamic=True, clip=clip, max_noskip=500, max_scale=2*32) 
    else: learn.to_fp16(dynamic=True)
    if lsuv: lsuv_init(learn)
    

    learn.fit_one_cycle(epochs, max_lr=lr)
    learn.save(f'final_of_{MODEL_NAME}')


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
            zb = model(xb).detach().cpu(); inputs.append(zb)
            targets.append(yb.detach().cpu())

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
        with open(save_fn, 'w') as f: f.write(str([test1_res, test2_res]))

    else: pass
