from . import ndarray_backend_cpu
from . import ndarray
from ._dtype import default_dtype
from ._random import random_array

class BackendDevice:
    def __init__(self, name, mod):
        self.name = name
        self.mod = mod
    
    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name
    
    def __repr__(self) -> str:
        return self.name + "()"
    
    def __getattr__(self, name):
        return getattr(self.mod, name)
    
    def enableed(self):
        return self.mod is not None
    
    def empty(self, shape, dtype=None):
        dtype = default_dtype() if dtype is None else dtype
        return ndarray.NDArray.make(shape, device=self, dtype=dtype)

    def full(self, shape, fill_value, dtype):
        dtype = default_dtype() if dtype is None else dtype
        arr = self.empty(shape, dtype=dtype)
        arr.fill(fill_value)
        return arr
    
    def randn(self, *shape, dtype):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        dtype = dtype if dtype else default_dtype()
        return ndarray.NDArray(random_array(*shape, 
                                    distribution="normalvariate",
                                    params=(0, 1),
                                    dtype="float"),
                        device=self,
                        dtype=dtype)
    

def cpu():
    "Return cpu device"
    return BackendDevice('cpu', ndarray_backend_cpu)

def default_device():
    return cpu()