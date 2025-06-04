import torch


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
            @joshuafan: should be (batch, in_dim)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
            @joshuafan: should be (in_dim, G+2k) where G is number of grid intervals, k is spline order
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        
    '''
    b_splines = B_batch(x_eval, grid, k=k)  # output: (batch, in_dim, G+k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))  # [y_eval]: (batch, in_dim, out_dim)
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, smoothness_lamb=1):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
        lamb : float
            regularized least square lambda
            
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
    '''
    #print('haha', x_eval.shape, y_eval.shape, grid.shape)
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
    #print('mat', mat.shape)
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)  # [in_dim, out_dim, batch, 1]
    #print('y_eval', y_eval.shape)
    device = mat.device

    # # ORIGINAL KAN CODE: find the optimal coefficients (no smoothness penalty)
    # coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
    # try:
    #     coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
    # except:
    #     print('lstsq failed')

    # NOT USED: anual psuedo-inverse
    '''lamb=1e-8
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:,:,:,0]'''

    # NEW OPTION: Add smoothness penalty on the coefficients.
    # Need to solve (B'B + \lambda D'D) \alpha = B' y
    # (Equation 2 in "Splines, Knots, and Penalties" (Eiler & Marx 2010)
    #
    # First, construct the D matrix, which has shape [n_coef-1, n_coef] and has the following form
    # [[ -1,  1,  0,  0, ...],
    #  [  0, -1,  1,  0, ...],
    #  [  0,  0, -1,  1, ...],
    #  [  ...            ...]]
    # For row i, position (i, i) contains -1 and position (i, i+1) contains 1. Other entries in the row are 0.
    # When we multiply this D matrix by the coef vector "a", we get a vector:
    # Da = [ a2-a1, a3-a2, a4-a3, ... ]^T
    D = torch.zeros((n_coef-1, n_coef))
    D[range(0, n_coef-1), range(0, n_coef-1)] = -1  # Fill in the (i, i) diagonal with -1
    D[range(0, n_coef-1), range(1, n_coef)] = 1  # Fill in the (i, i+1) diagonal with 1
    B = mat  # [in_dim, out_dim, batch, n_coef]
    B_T = mat.permute((0, 1, 3, 2))  # [in_dim, out_dim, n_coef, batch]

    # Find the least squares fit
    coef = None
    try:
        coef = torch.linalg.lstsq((B_T @ B) + smoothness_lamb * (D.T @ D), B_T @ y_eval).solution[:, :, :, 0]
    except:  # NOTE @joshuafan may be unncessary now
        print(f"lstsq failed")
        raise

    return coef


def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid