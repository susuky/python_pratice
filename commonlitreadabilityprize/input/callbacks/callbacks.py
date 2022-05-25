#!/usr/bin/env python
# coding: utf-8

# # utils

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List


def noop(x=None): return x
def to_numpy(x): return x.detach().cpu().numpy()
def is_listy(x): return isinstance(x, (list, tuple))
def to_device(x, device):
    if is_listy(x): return [to_device(o, device) for o in x]
    else: return x.to(device)
def get_dirs(obj): return [d for d in dir(obj) if '__' not in d]
def set_cfg_attr(self, cfg):
    dirs = get_dirs(cfg)
    for key in dirs:
        value = getattr(cfg, key)
        setattr(self, key, value)          
            
def add_cb(self, idx, cb):
    cb.learner = self
    cb.before_fit()
    self.cbs.insert(idx, cb)
def del_cb(self, idx):  del self.cbs[idx] 
def update_cb(self, idx, cb):
    del_cb(self, idx)
    add_cb(self, idx, cb)

def plot_history(self, skip_start=0):
    # get loss
    loss = self.history['trn_loss']
    val_loss = self.history['val_loss']
    
    # get metrics
    metrics, val_metrics, func_names = {}, {}, []
    for metric_func in self.metric_funcs:
        func_name = metric_func.__name__
        func_names.append(func_name)
        metrics[func_name] = self.history[f'trn_{func_name}']
        val_metrics[func_name] = self.history[f'val_{func_name}']
        
    nums_of_fig = 1 + len(metrics)

    # x_value
    xlabel = 'iteration' if len(loss) != len(val_loss) else 'epoch'
    val_interval = len(loss) // len(val_loss)
    xlim = range(1, len(loss)+1)[skip_start * val_interval:]
    xlim_val = range(val_interval, len(loss) + val_interval, val_interval)[skip_start:]

    fig = plt.figure(figsize=(nums_of_fig * 4+2, 4))
    # loss
    plt.subplot(1, nums_of_fig, 1)
    plt.plot(xlim, loss[skip_start * val_interval:], 'C0', label='train')
    plt.plot(xlim_val, val_loss[skip_start:], 'C1', label='val')
    plt.title(f'loss')
    plt.legend()
    plt.xlabel(xlabel)

    # metrics
    for i, func_name in enumerate(func_names, 2):
        plt.subplot(1, nums_of_fig, i)
        plt.plot(xlim, metrics[func_name][skip_start * val_interval:], 'C0', label='train')
        plt.plot(xlim_val, val_metrics[func_name][skip_start:], 'C1', label='val')
        plt.title(f'{func_name}')
        plt.legend()
        plt.xlabel(xlabel)

    plt.tight_layout()
    plt.show()
    
def plot_lr(self):
    lrs = sorted(set([*zip(*self.lr_recorder)]))
    
    # plot
    plt.figure(figsize=(len(lrs)*3+2, 3))
    for i, lr in enumerate(lrs, 1):
        plt.subplot(1, len(lrs), i)
        plt.title(f'group{i}')
        plt.plot(lr)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.show() 


# In[ ]:


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count


# In[ ]:


# preload dataloader wrapper for windows
import threading
import queue


'''
if cfg.worker == 0 and os.name == 'nt':
    trn_loader = MultiThreadWrapper(trn_loader, cfg.max_prefetch)  
'''
class MultiThreadWrapper(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        self.exhausted = False

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __len__(self):
        return len(self.generator)

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    def __iter__(self):
        return self


# # callbacks

# In[ ]:


class Callback:
    _default = 'learner'

    def before_fit(self):
        pass

    def after_fit(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_batch(self):
        pass

    def after_batch(self):
        pass

    def __getattr__(self, x):
        attr = getattr(self, self._default, None)
        if attr is not None:
            return getattr(attr, x)


# In[ ]:


# EarlyStop


class EarlyStop(Callback):
    def __init__(self,
                 moniter='val_loss',
                 patience=4,
                 mode='min',
                 verbose=1):
        '''
        mode: {min, max}
        if moniter not exist, this class would do nothing!
        '''

        self.best_score = np.inf if mode == 'min' else -np.inf
        self.moniter = moniter
        self.patience = patience
        self.mode = mode
        self.verbose = verbose

        self.wait = 0
        
    def after_epoch(self):
        if not self.train:
            value = getattr(self, self.moniter, 0.)
            improve = self._cal_improve(value)

            if improve:
                self.wait = 0
                self.best_score = value
            else:
                self.wait += 1

            if self.wait > self.patience:
                self.learner.stop_learn = True
                if self.verbose: print('EarlyStop!')
            else:
                self.learner.stop_learn = False

    def _cal_improve(self, value) -> bool:
        improve = True if value < self.best_score else False
        improve = not improve if self.mode == 'max' else improve
        return improve


# In[ ]:


# ProgressBar
from fastprogress import master_bar, progress_bar


class ProgressBar(Callback):
    def before_fit(self):
        self.learner.mb = master_bar(range(1, self.epochs + 1))
        lines = ['epoch', 'train loss', 'val loss', 'time']
        for func in self.metric_funcs:
            lines.insert(-1, 'train ' + func.__name__)
            lines.insert(-1, 'val ' + func.__name__)
        self.learner.mb.write(lines, table=True)

    def before_epoch(self):
        self.learner.pb = progress_bar(self.dl,
                                       total=len(self.dl),
                                       parent=self.mb)

    def after_epoch(self):
        if not self.train:
            m, s = self.time_spend

            lines = [
                f'{self.epoch}', 
                f'{self.trn_loss:.6f}',
                f'{self.val_loss:.6f}', 
                f'{m:.0f}:{s:02.0f}'
            ]

            for trn_metric, val_metric in zip(self.trn_metrics,
                                              self.val_metrics):
                lines.insert(-1, f'{trn_metric:.6f}')
                lines.insert(-1, f'{val_metric:.6f}')

            self.mb.write(lines, table=True)

    def after_batch(self):
        # set commet on child progressbar
        mode = 'train' if self.train else 'val'
        lines = f'{mode} loss: {self.running_loss.avg():.4f}, '
        for running_metric, metric_func in zip(self.running_metrics,
                                                   self.metric_funcs):
            lines +=f'{mode} {metric_func.__name__}: {running_metric.avg():.4f}, '
        self.mb.child.comment = lines[:-2]


# In[ ]:


# Recorder


class Recorder(Callback):
    def __init__(self, recorder_type='iteration'):
        self.recorder_type = 'iteration'

    def before_fit(self):
        # create recorder
        self.learner.history = {
            'trn_loss': [],
            'val_loss': [],
        }
        # can be multi-function or don't have function
        for metric_func in self.metric_funcs:
            self.learner.history['trn_' + metric_func.__name__] = []
            self.learner.history['val_' + metric_func.__name__] = []

    def before_epoch(self):
        self.learner.running_loss = AverageMeter()
        self.learner.running_metrics = [AverageMeter() for _ in range(len(self.metric_funcs))]

    def after_epoch(self):
        if self.train and self.recorder_type == 'epoch':
            # training loss per epoch
            self.learner.trn_loss = self.running_loss.avg()
            self.history['trn_loss'].append(self.trn_loss)

            # training metrics per epoch
            self.learner.trn_metrics = []
            for running_metric, metric_func in zip(self.running_metrics,
                                                   self.metric_funcs):
                self.trn_metrics.append(running_metric.avg())
                self.history[f'trn_{metric_func.__name__}'].append(running_metric.avg())

        if not self.train:
            # validation loss
            self.learner.val_loss = self.running_loss.avg()
            self.history['val_loss'].append(self.val_loss)
            
            # validation metrics
            self.learner.val_metrics = []
            for running_metric, metric_func in zip(self.running_metrics,
                                                   self.metric_funcs):
                self.val_metrics.append(running_metric.avg())
                self.history[f'val_{metric_func.__name__}'].append(running_metric.avg())
        
    def after_batch(self):
        self.running_loss.update(self.loss_item, self.batch_size)

        # calculate metrics
        for i, metric_func in enumerate(self.metric_funcs):
            with torch.no_grad():
                metric = metric_func(self.preds, self.yb)
            metric = to_numpy(metric)
            self.running_metrics[i].update(metric, self.batch_size)

        if self.train and self.recorder_type == 'iteration':
            # training loss per iteration
            self.learner.trn_loss = self.running_loss.avg()
            self.history['trn_loss'].append(self.trn_loss)

            # training metrics per iteration
            self.learner.trn_metrics = []
            for running_metric, metric_func in zip(self.running_metrics,
                                                   self.metric_funcs):
                self.trn_metrics.append(running_metric.avg())
                self.history[f'trn_{metric_func.__name__}'].append(running_metric.avg())


# In[ ]:


# Timer
import time


class Timer(Callback):
    def before_epoch(self):
        if self.train: self.start_time = time.time()

    def after_epoch(self):
        if not self.train:
            # (minute, second)
            self.learner.time_spend = divmod(time.time() - self.start_time, 60)


# In[ ]:


# Scheduler
import torch.optim as optim
from typing import Callable


def get_scheduler(optimizer, cfg):
    return optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1)


class Scheduler(Callback):
    def __init__(self, get_scheduler):
        self.get_scheduler = get_scheduler
        
    def before_fit(self):
        self.learner.lr_recorder = []
        self.learner.scheduler = self.get_scheduler(self.optimizer, self.cfg)

    def after_batch(self):
        if self.train:
            self.lr_recorder.append(self.scheduler.get_last_lr())
            self.scheduler.step()

    def after_epoch(self):
        pass


# In[ ]:


# ModelCheckPoint
import os

class ModelCheckPoint(Callback):
    def __init__(self, store_name, skip_start=0, auto_load=True, verbose=1):
        self.store_name = store_name
        self.skip_start = skip_start
        self.auto_load = True
        self.verbose = verbose
        
        self.best_loss = np.inf

    def before_fit(self):        
        self.learner.store_name = self.store_name
        

    def after_fit(self):
        if self.auto_load:
            if os.path.exists(f'{self.store_name}.pth'):  
                self.load(self.store_name, model_only=True)
                if self.verbose in [1, 2]:
                    print('Autoload the best model')
            else:
                print('Autoload model, but do not have stored model')

    def after_epoch(self):
        if not self.train:
            if self.val_loss < self.best_loss and self.epoch > self.skip_start:
                store_name = self.learner.store_name
                if self.verbose in [2]:
                    print(f'val_loss: {self.val_loss:.6f} better than best_loss: {self.best_loss:.6f}')
                    print(f'save checkpoints to {save_name}.pth')
                self.save(store_name)
                self.best_loss = self.val_loss 


# In[ ]:


# GetLogger
def get_logger(filename='log', save=True, verbose=True):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    if verbose:
        handler1 = StreamHandler()
        handler1.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler1)
    
    if save:
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))  
        logger.addHandler(handler2)
    return logger

import torch
from datetime import datetime
# filename = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")


class GetLogger(Callback):
    def __init__(self, logger=None, **args):
        self.logger = logger if logger else get_logger(**args)
    
    def before_fit(self): 
        self._log_setup()
        
        lines = f'|{"time":>8s}|{"lr":>12s}|'                f'{"epoch":>9s}|{"iteration":>10s}|'                f'{"loss":>10s}|{"val loss":>10s}|'
        for func in self.metric_funcs:
            func_name = func.__name__
            lines += f'{"train_" + func_name:>12s}|'
            lines += f'{"val_" + func_name:>12s}|'
            
        self.logger.info(lines)
        
    def after_epoch(self):
        if not self.train:
            m, s = self.time_spend
            iteration = self.glob_iters
            
            # get lr
            lr = self.lr_recorder[-1]
            lr = lr[0] if isinstance(lr, List) else lr
            
            lines = [
                f'|{m:5.0f}:{s:02.0f}|',
                f'{lr:>12.4e}|',
                f'{self.epoch:>9d}|', 
                f'{iteration:>10d}|'
                f'{self.trn_loss:>10.6f}|',
                f'{self.val_loss:>10.6f}|', 
            ]

            for trn_metric, val_metric in zip(self.trn_metrics,
                                              self.val_metrics):
                lines.append(f'{trn_metric:>12.6f}|')
                lines.append(f'{val_metric:>12.6f}|')
                
            lines = ''.join(l for l in lines)
        
            self.logger.info(lines)           
    
    def after_fit(self):
        self.logger.info('** Finish **\n')
        
    def _log_setup(self):
        self.logger.info('\n')
        self.logger.info('** Setup **\n')
        self.logger.info(f'Time: {datetime.today()}\n')
        if torch.cuda.is_available():
            self.logger.info(f'device: {torch.cuda.get_device_name(self.device)}\n')
                
        params = {d: getattr(self.cfg, d) for d in get_dirs(self.cfg)}
        self.logger.info(f'Parameters:')
        for param in params.items():
            self.logger.info(f'\t{param[0]}:{param[1]}')        
        self.logger.info('\n')

