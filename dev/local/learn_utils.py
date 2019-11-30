from fastai.vision import *
from fastai.callbacks import *
        
    
__all__ = ['dice_loss', 'bce', 'SaveModelCallback', 'get_img_pred_masks', 'plot_predictions', 'dice_score', 'dice_loss',                    'MixedLoss', 'CatchNanGrad', 'CatchNanActs', 'CatchNanParameters', 'lsuv_init', 'cond_init']
    
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
                
                

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
gnorm_types = (nn.GroupNorm,)
insnorm_types = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
norm_types = bn_types + gnorm_types + insnorm_types

def cond_init(m:nn.Module, init_func:LayerFunc):
    "Initialize the non-batchnorm layers of `m` with `init_func`."
    if (not isinstance(m, norm_types)) and (not isinstance(m, nn.PReLU)) and requires_grad(m): 
        init_default(m, init_func)
        
class BackwardHookCallback(LearnerCallback):
    "Callback that can be used to register hooks on `modules`. Implement the corresponding function in `self.hook`."
    def __init__(self, learn:Learner, modules:Sequence[nn.Module]=None, do_remove:bool=True):
        super().__init__(learn)
        self.modules,self.do_remove = modules,do_remove

    def on_train_begin(self, **kwargs):
        "Register the `Hooks` on `self.modules`."
        if not self.modules:
            self.modules = [m for m in flatten_model(self.learn.model)
                            if hasattr(m, 'weight')]
        # needs to be is_forward=False, detach=False
        self.hooks = Hooks(self.modules, self.hook, is_forward=False, detach=False)

    def on_train_end(self, **kwargs):
        "Remove the `Hooks`."
        if self.do_remove: self.remove()

    def remove(self): 
        if getattr(self, 'hooks', None): self.hooks.remove()
    
    def __del__(self): self.remove()
        
        
class CatchNanGrad(BackwardHookCallback):
    "Catch NaN when first appears in grad"

    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        self.stop = False

    def hook(self, m:nn.Module, i:Tensors, o:Tensors)->Tuple[Rank0Tensor,Rank0Tensor]:
        "Take the mean and std of `o`."
        if (not self.stop) and (torch.any(torch.isnan(o[0]))): 
            self.stop = True
            print(m,o)
            return (m, o)
    
    def on_backward_end(self, train, epoch, num_batch, **kwargs):
        "Called after backprop but before optimizer step."
        if train and self.stop: 
            print (f'Epoch/Batch ({epoch}/{num_batch}): Invalid Grad, terminating training.')
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}
        
class CatchNanActs(HookCallback):
    "Catch NaN when first appears in acts"

    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        self.stop = False

    def hook(self, m:nn.Module, i:Tensors, o:Tensors)->Tuple[Rank0Tensor,Rank0Tensor]:
        "Take the mean and std of `o`."
        if (not self.stop) and (torch.any(torch.isnan(o[0]))): 
            self.stop = True
            print(m,o)
            return (m, o)
            
    def on_loss_begin(self, train, epoch, num_batch, **kwargs):
        "Called after forward pass but before loss has been computed."
        if train and self.stop: 
            print (f'Epoch/Batch ({epoch}/{num_batch}): Invalid Activation, terminating training.')
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}
        
class CatchNanParameters(LearnerCallback):
    "Catch NaN when first appears in parameter weight"
    
    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        self.modules = [m for m in flatten_model(self.learn.model) if hasattr(m, 'weight')]
        
    def on_backward_end(self, train, epoch, num_batch, **kwargs):
        for m in self.modules: 
            if m.weight is not None:
                if torch.any(torch.isnan(m.weight)): 
                    print (f'Epoch/Batch ({epoch}/{num_batch}): Invalid Parameter, terminating training.')
                    print(m,o)
                    return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}
                
class ActStats:
    def __init__(self):
        pass
    def __call__(self, m, i, o):
        d = o.data
        self.mean,self.std = d.mean().item(),d.std().item()

def lsuv_module(m, model, xb):
    stats = ActStats()
    h = Hook(m, stats)

    if hasattr(m, 'bias'): 
        while model(xb) is not None and abs(stats.mean)  > 1e-3: m.bias -= stats.mean
    if hasattr(m, 'weight'):
        while model(xb) is not None and abs(stats.std-1) > 1e-3: m.weight.data /= stats.std

    h.remove()
    return stats.mean, stats.std

def lsuv_init(learn):
    "initialize model parameters with LSUV - https://arxiv.org/abs/1511.06422"
    modules = [m for m in flatten_model(learn.model) if hasattr(m, 'weight') and m.weight is not None]
    mdl = learn.model.cuda()
    xb, yb = learn.data.one_batch()
    for m in modules: print(lsuv_module(m, learn.model, xb.cuda()))
    for m in modules: 
        if hasattr(m, 'weight'): assert not torch.any(torch.isnan(m.weight))
        if hasattr(m, 'bias'): assert not torch.any(torch.isnan(m.bias))
    del modules
    del mdl
    del xb
    del yb
    gc.collect()
            
def get_img_pred_masks(learner, dl, thresh=0.5):
    model = learner.model.eval()
    preds = []
    images = []
    masks = []
    for image, mask in tqdm_notebook(dl):
        out = model(image.half().cuda())
        out = torch.sigmoid(out).cpu().data.numpy()
        out = out.astype(float)
        out = (out > thresh)*1
        image = image.cpu().data.numpy()
        mask = mask.cpu().data.numpy()
        for i in range(out.shape[0]):
            masks.append(mask[i])
            preds.append(out[i][0])
            images.append(image[i][0])
    return images, preds, masks

def plot_predictions(true, pred, mask):
    """draw image, pred, mask side by side"""
    fig, ax = plt.subplots(1,3, figsize=(20,10))
    axes = ax.flatten()
    for ax, im, t in zip(axes, [true, pred, mask], ["image", "pred", "mask"]) :
        ax.imshow(im, cmap="gray")
        ax.set_title(t, fontdict={"fontsize":20})

def dice_score(logits, targets, thresh=0.5):
    hard_preds = torch.sigmoid(logits) > thresh
    m1 = hard_preds.view(-1).float() 
    m2 = targets.view(-1).float()
    intersection = (m1 * m2).sum()
    return (2. * intersection) / (m1.sum() + m2.sum() + 1e-6)

def eval_preds(preds, target, thresh=0.5):
    iflat = np.array(preds).reshape(-1)
    tflat = np.array(target).reshape(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection) / (iflat.sum() + tflat.sum() + 1e-6))

def dice_loss(logits, target, smooth=1.):
    logits = torch.sigmoid(logits)
    iflat = logits.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, logits, target):
        logits = logits.squeeze(1)
        if not (target.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logits.size()))

        max_val = (-logits).clamp(min=0)
        loss = logits - logits * target + max_val + \
            ((-max_val).exp() + (-logits - max_val).exp()).log()

        invprobs = F.logsigmoid(-logits * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()
    
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(1 - dice_loss(input, target))
        return loss.mean()    
    