from . import ndarray_backend_cpu
import numpy as np
import math
from enum import Enum
import numbers

def is_numeric_list(lst):
    for el in lst:
        if isinstance(el, list):
            if not is_numeric_list(el):
                return False
        elif not isinstance(el, numbers.Number):
            return False
    return True


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result



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

dtypes = set(["int8", "int16", "int32", "int64", 
            "uint8", "uint16", "uint32", "uint64",
            "float32", "float64"])


def default_dtype():
    return 'float32'

def cpu():
    "Return cpu device"
    return BackendDevice('cpu', ndarray_backend_cpu)

def default_device():
    return cpu()

class NDArray:
    def __init__(self, other, device=None, dtype=None):
        if dtype not in dtypes:
            raise TypeError('Could not find %s in dtypes' % (dtype))

        if isinstance(other, NDArray):
            if device is None:
                device = other.device
            if dtype is None:
                dtype = other.dtype
            self._init(other.to(device) + 0) # deep copy

        elif isinstance(other, np.ndarray):
            device = device if device else default_device()
            if dtype:
                other = other.astype(dtype)
            dtype = str(other.dtype)
            array = self.make(shape=other.shape, device=device, dtype=dtype)
            array._device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
            
        elif isinstance(other, list):
            """  目前无法应对不均匀数组以及浮点型到整形的存储问题。
            device = device if device else default_device()
            dtype = dtype if dtype else default_dtype()

            if not is_numeric_list(other):
                raise ValueError("Invalid item in list! The item in list must be numeric")
            # calculate the shape of list
            cur_shape = []
            cur_lst = other
            while isinstance(cur_lst, list):
                cur_shape.append(len(cur_lst))
                cur_lst = cur_lst[0]
            cur_shape = tuple(cur_shape)

            array = self.make(shape=cur_shape, device=device, dtype=dtype)
            # Flatten the list
            other = flatten(other)
            array._device.from_pylist(other, array._handle)
            self._init(array)
            """

            try:
                array = NDArray(np.array(other, dtype=dtype), device=device, dtype=dtype)
                self._init(array)
            except ValueError as r:
                raise r
        else:
            array = NDArray(np.array(other), device=device, dtype=dtype)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._dtype = other._dtype
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
    def make(shape, strides=None, offset=0, device=None, dtype=None, handle=None):
        """Create a new NDArray with the given properties.  
        This will allocation the memory if handle=None, otherwise it will 
        use the handle of an existing array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = strides if strides else NDArray.compact_strides(shape)
        array._offset = offset
        array._device = device if device else default_device()
        array._dtype = dtype if dtype else default_dtype()
        if handle:
            array._handle = handle
        else:
            array._handle = getattr(array._device, "Array_" + array._dtype)(math.prod(shape))
        
        return array
    
    def __repr__(self) -> str:
        return "NDArray(" + self.numpy().__str__() + f", device={self._device}, dtype={self._dtype})"
    
    def numpy(self):
        return self._device.to_numpy(self._handle, self._shape, self._strides, self._offset)
    
    def to(self, device):
        if device == self._device:
            return self
        else:
            return NDArray(self.numpy(), dtype=self.dtype, sdevice=device)
    
    @property
    def dtype(self):
        return self._dtype

    def astype(self, dtype):
        return NDArray(self.numpy(), device=self._device, dtype=dtype)
