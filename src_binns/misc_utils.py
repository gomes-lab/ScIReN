import collections
import torch
import torch.nn

@torch.no_grad()
def update_bn_custom(loader, model, device=None):
    r"""Method from SWA library, but adapted to use our custom data, 
    where each dataloader batch contains x,z,y,profile_id and x,z
    need to be passed through the model.

    Documentation from SWA:    
    Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input in loader:
        batch_x, batch_y, batch_z, batch_profile_id = input
        if device is not None:
            batch_x = batch_x.to(device)
            batch_z = batch_z.to(device)

        model(batch_x, batch_z, whether_predict=0)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)



def zero_gradients(x):
    """
    This was formerly an old function in "torch.autograd.gradcheck".
    It is now removed so we reproduce the code here, see
    https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462
    """
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def _find_z(new_input, batch_y, batch_z, model, criterion, h):
    '''
    Finding the direction in the regularizer
    '''
    batch_x.requires_grad_()
    outputs, h5, new_input = model.eval().partial(batch_x, batch_z, whether_predict=0, return_input=True)
    loss_z = criterion(outputs, batch_y)                
    loss_z.backward(torch.ones(batch_y.shape).to(batch_y.device))         
    grad = new_input.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)  
    zero_gradients(new_input) 
    model.zero_grad()

    return z, norm_grad, new_input

    
def regularizer(batch_x, batch_y, batch_z, model, criterion, h = 3.):
    '''
    Regularizer term in CURE

    Returns both curvature loss
    '''
    with torch.no_grad():
        outputs, h5, new_input = model.eval()(batch_x, batch_z, whether_predict=0, return_input=True)

    # z is a direction of high curvature (pointing in similar direction to the gradient)
    z, norm_grad, new_input = _find_z(batch_x, batch_y, batch_z, model, criterion, h)
    
    new_input.requires_grad_()

    # Compute 
    outputs_pos, _ = model.eval().partial_forward(new_input + z)
    outputs_orig, _ = model.eval().partial_forward(new_input)
    loss_pos = criterion(outputs_pos, batch_y)
    loss_orig = criterion(outputs_orig, batch_y)
    grad_diff = torch.autograd.grad((loss_pos-loss_orig), new_input,
                                     grad_outputs=torch.ones(batch_y.shape).to(batch_y.device),
                                     create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    model.zero_grad()

    return torch.sum(reg) / float(new_input.size(0)), norm_grad