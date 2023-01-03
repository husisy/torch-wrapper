import time
import numpy as np
import scipy.optimize

try:
    import torch
except ImportError:
    torch = None

def _get_sorted_parameter(model):
    tmp0 = sorted([(k,v) for k,v in model.named_parameters() if v.requires_grad], key=lambda x:x[0])
    ret = [x[1] for x in tmp0]
    return ret


def get_model_flat_parameter(model):
    tmp0 = _get_sorted_parameter(model)
    ret = np.concatenate([x.detach().cpu().numpy().reshape(-1) for x in tmp0])
    return ret


def get_model_flat_grad(model):
    tmp0 = _get_sorted_parameter(model)
    ret = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1) for x in tmp0])
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
            # scipy.optimize.LBFGS does not support float32 @20221118
            grad = np.concatenate([x.grad.detach().cpu().numpy().reshape(-1).astype(theta.dtype) for x in parameter_sorted])
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


def check_model_gradient(model, tol=1e-5, zero_eps=1e-4, seed=None):
    np_rng = np.random.default_rng(seed)
    num_parameter = get_model_flat_parameter(model).size
    # TODO range for paramter
    theta0 = np_rng.uniform(0, 2*np.pi, size=num_parameter)

    set_model_flat_parameter(model, theta0)
    loss = model()
    for x in model.parameters():
        if x.grad is not None:
            x.grad.zero_()
    if hasattr(model, 'grad_backward'):
        model.grad_backward(loss)
    else:
        loss.backward()
    ret0 = get_model_flat_grad(model)

    def hf0(theta):
        set_model_flat_parameter(model, theta)
        ret = model()
        if hasattr(ret, 'item'):
            ret = ret.item()
        return ret
    ret_ = np.zeros(num_parameter, dtype=np.float64)
    for ind0 in range(ret_.shape[0]):
        tmp0,tmp1 = [theta0.copy() for _ in range(2)]
        tmp0[ind0] += zero_eps
        tmp1[ind0] -= zero_eps
        ret_[ind0] = (hf0(tmp0)-hf0(tmp1))/(2*zero_eps)
    assert np.abs(ret_-ret0).max()<tol


def minimize(model, rand_kind=None, num_repeat=3, tol=1e-7, print_freq=-1,
            method='L-BFGS-B', print_step_info=True, maxiter=None, seed=None):
    np_rng = np.random.default_rng(seed)
    if rand_kind is None:
        rand_kind = ('uniform', -1, 1)
    if isinstance(rand_kind, str):
        if rand_kind=='uniform':
            rand_kind = ('uniform', -1, 1)
        elif rand_kind=='normal':
            rand_kind = ('normal', 0, 1)
    if rand_kind[0]=='uniform':
        hf_theta = lambda *x: np_rng.uniform(rand_kind[1], rand_kind[2], size=x)
    elif rand_kind[0]=='normal':
        hf_theta = lambda *x: np_rng.normal(rand_kind[1], rand_kind[2], size=x)
    num_parameter = len(get_model_flat_parameter(model))
    hf_model = hf_model_wrapper(model)
    ret = []
    min_fun = None
    options = dict() if maxiter is None else {'maxiter':maxiter}
    for ind0 in range(num_repeat):
        theta0 = hf_theta(num_parameter)
        hf_callback = hf_callback_wrapper(hf_model, print_freq=print_freq)
        theta_optim = scipy.optimize.minimize(hf_model, theta0, jac=True, method=method, tol=tol, callback=hf_callback, options=options)
        ret.append(theta_optim)
        min_fun = theta_optim.fun if min_fun is None else min(min_fun, theta_optim.fun)
        if print_step_info:
            print(f'[round={ind0}] min(f)={min_fun}, current(f)={theta_optim.fun}')
    ret = min(ret, key=lambda x: x.fun)
    set_model_flat_parameter(model, ret.x)
    return ret
