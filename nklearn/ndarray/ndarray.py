from . import ndarray_backend_cpu
import numpy as np
import math

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
    

def cpu():
    "Return cpu device"
    return BackendDevice('cpu', ndarray_backend_cpu)

def default_device():
    return cpu()

class NDArray:
    def __init__(self, other, device=None):
        if isinstance(other, NDArray):
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0) # deep copy

        elif isinstance(other, np.ndarray):
            device = device if device else default_device()
            array = self.make(shape=other.shape, device=device)
            array._device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
            
        elif isinstance(other, list):
            pass
        else:
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """ Utility function to compute compact strides """
        strides = 1
        res = []
        for i in range(1, len(shape)+1):
            res.append(strides)
            strides *= shape[-i]
        return tuple(res[::-1])
    
    @staticmethod
    def make(shape, strides=None, offset=0, device=None, handle=None):
        """Create a new NDArray with the given properties.  
        This will allocation the memory if handle=None, otherwise it will 
        use the handle of an existing array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = strides if strides else NDArray.compact_strides(shape)
        array._offset = offset
        array._device = device if device else default_device()
        if handle:
            array._handle = handle
        else:
            array._handle = array._device.Array(math.prod(shape))
        
        return array
    
    def __repr__(self) -> str:
        return "NDArray(" + self.numpy().__str__() + f", device={self._device})"
    
    def numpy(self):
        return self._device.to_numpy(self._handle, self._shape, self._strides, self._offset)
    
