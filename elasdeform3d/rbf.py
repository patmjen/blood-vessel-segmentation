import torch
import torch.nn.functional as F


def xlogy(x, y):
    """
    Compute x * log(y) where x != 0 and 0 where x == 0.
    """
    # From https://github.com/pytorch/pytorch/issues/22656#issuecomment-510147594
    z = torch.zeros_like(x)
    return x * torch.where(x == 0, z, torch.log(y))


class Rbf(object):
    """
    Pure PyTorch implementation of scipy's Rbf, which can (optionally) perform
    calculations on the GPU. Only supports Euclidean p-norms for distances.
    """
    # Available radial basis functions that can be selected as strings;
    # they all start with _h_ (self._init_function relies on that)
    def _h_multiquadric(self, r):
        return torch.sqrt((1.0/self.epsilon*r)**2 + 1)


    def _h_inverse_multiquadric(self, r):
        return 1.0/torch.sqrt((1.0/self.epsilon*r)**2 + 1)


    def _h_gaussian(self, r):
        return torch.exp(-(1.0/self.epsilon*r)**2)


    def _h_linear(self, r):
        return r


    def _h_cubic(self, r):
        return r**3


    def _h_quintic(self, r):
        return r**5


    def _h_thin_plate(self, r):
        return xlogy(r**2, r)


    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                functionlist = [x[3:] for x in dir(self)
                                if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                 ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)
        elif callable(self.function):
            allow_one = False
            if hasattr(self.function, 'func_code') or \
               hasattr(self.function, '__code__'):
                val = self.function
                allow_one = True
            elif hasattr(self.function, "__call__"):
                val = self.function.__call__.__func__
            else:
                raise ValueError("Cannot determine number of arguments to "
                                 "function")

            argcount = val.__code__.co_argcount
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                self._function = self.function.__get__(self, Rbf)
            else:
                raise ValueError("Function argument must take 1 or 2 "
                                 "arguments.")

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of "
                             "the same shape")
        return a0


    def __init__(self, *args, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements
        self.xi = torch.stack(
            [torch.as_tensor(a, dtype=torch.float32).flatten()
             for a in args[:-1]])
        self.N = self.xi.shape[-1]
        self.device = self.xi.device

        self.mode = kwargs.pop('mode', '1-D')

        if self.mode == '1-D':
            self.di = torch.as_tensor(args[-1]).flatten()
            self._target_dim = 1
        elif self.mode == 'N-D':
            self.di = torch.as_tensor(args[-1])
            self._target_dim = self.di.shape[-1]
        else:
            raise ValueError("Mode has to be 1-D or N-D.")

        if self.xi.device != self.di.device:
            raise ValueError("All arrays must be on same device.")

        if not all([x.numel() == self.di.shape[0] for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', 2)
        self.epsilon = kwargs.pop('epsilon', None)
        if self.epsilon is None:
            # default epsilon is the "the average distance between nodes" based
            # on a bounding hypercube
            ximax = torch.amax(self.xi, axis=1)
            ximin = torch.amin(self.xi, axis=1)
            edges = ximax - ximin
            edges = edges[torch.nonzero(edges)]
            self.epsilon = torch.prod(edges)/self.N ** (1.0/edges.numel())

        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', 'multiquadric')

        # attach anything left in kwargs to self for use by any user-callable
        # function or to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)

        # Compute weights
        if self._target_dim > 1:  # If we have more than one target dimension,
            # we first factorize the matrix
            self.nodes = torch.zeros(self.N, self._target_dim,
                                     dtype=self.di.dtype, device=self.device)
            lu_data = torch.lu(self.A)
            for i in range(self._target_dim):
                self.nodes[:, i] = torch.lu_solve(self.di[:, i].unsqueeze(0).T,
                                                  *lu_data).squeeze()
        else:
            self.nodes = torch.solve(self.A, self.di)[0]


    @property
    def A(self):
        # this only exists for backwards compatibility: self.A was available
        # and, at least technically, public.

        # For now, just use cdist since torch does not provide a convenient way
        # to get the distances in squareform.
        r = torch.cdist(self.xi.T, self.xi.T, p=self.norm)  # Pairwise norm
        reg = torch.eye(self.N, device=self.device) * self.smooth
        return self._init_function(r) - reg


    def _call_norm(self, x1, x2):
        return torch.cdist(x1.T, x2.T, p=self.norm)


    def __call__(self, *args):
        args = [torch.as_tensor(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        if not all([x.device == self.device for x in args]):
            raise ValueError(
                "Arrays must be on device: {}".format(self.device))
        if self._target_dim > 1:
            shp = args[0].shape + (self._target_dim,)
        else:
            shp = args[0].shape
        xa = torch.stack([a.flatten() for a in args]).type(torch.float32)
        r = self._call_norm(xa, self.xi)
        return torch.mm(self._function(r), self.nodes).reshape(shp)