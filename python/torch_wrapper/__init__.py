import time
import torch
import numpy as np


def _get_sorted_parameter(model):
    tmp0 = sorted(list(model.named_parameters()), key=lambda x:x[0]) #sorted by name
    ret = [x[1] for x in tmp0]
    return ret


def get_model_flat_parameter(model):
    tmp0 = _get_sorted_parameter(model)
    ret = np.concatenate([x.detach().cpu().numpy().reshape(-1) for x in tmp0])
    return ret


def set_model_flat_parameter(model, theta, index01=None):
    theta = torch.tensor(theta)
    parameter_sorted = _get_sorted_parameter(model)
    if index01 is None:
        tmp0 = np.cumsum(np.array([0] + [x.numel() for x in parameter_sorted])).tolist()
        index01 = list(zip(tmp0[:-1],tmp0[1:]))
    for ind0,(x,y) in enumerate(index01):
        tmp0 = theta[x:y].reshape(*parameter_sorted[ind0].shape)
        if not parameter_sorted[ind0].is_cuda:
            tmp0 = tmp0.cpu()
        parameter_sorted[ind0].data[:] = tmp0


def hf_model_wrapper(model):
    parameter_sorted = _get_sorted_parameter(model)
    tmp0 = np.cumsum(np.array([0] + [x.numel() for x in parameter_sorted])).tolist()
    index01 = list(zip(tmp0[:-1],tmp0[1:]))
    def hf0(theta, tag_grad=True):
        set_model_flat_parameter(model, theta, index01)
        if tag_grad:
            loss = model()
            for x in parameter_sorted:
                if x.grad is not None:
                    x.grad.zero_()
            if hasattr(model, 'grad_backward'): #designed for custom automatic differentiation
                model.grad_backward(loss)
            else:
                loss.backward() #if no .grad_backward() method, it should be a normal torch.nn.Module
            grad = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1) for x in parameter_sorted])
        else:
            with torch.no_grad():
                loss = model()
            grad = None
        return loss.item(), grad
    return hf0


def hf_callback_wrapper(hf_fval, state:dict=None, print_freq:int=1):
    if state is None:
        state = dict()
    state['step'] = 0
    state['time'] = time.time()
    state['fval'] = []
    state['time_history'] = []
    def hf0(theta):
        step = state['step']
        if (print_freq>0) and (step%print_freq==0):
            t0 = state['time']
            t1 = time.time()
            fval = hf_fval(theta, tag_grad=False)[0]
            print(f'[step={step}][time={t1-t0:.3f} seconds] loss={fval}')
            state['fval'].append(fval)
            state['time'] = t1
            state['time_history'].append(t1-t0)
        state['step'] += 1
    return hf0
