from . import ndarray_backend_cpu
# from .ndarray import NDArray
from ._dtype import default_dtype

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
        return NDArray.make(shape, device=self, dtype=dtype)

    def full(self, shape, fill_value, dtype):
        dtype = default_dtype() if dtype is None else dtype
        arr = self.empty(shape, dtype=dtype)
        arr.fill(fill_value)
        return arr
    

def cpu():
    "Return cpu device"
    return BackendDevice('cpu', ndarray_backend_cpu)

def default_device():
    return cpu()