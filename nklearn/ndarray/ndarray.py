from . import ndarray_backend_cpu
import math
from enum import Enum
import numbers
from typing import List

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

def is_valid_list(lst:List) -> bool:
    """判断该 列表 是否可以用来创建 ndarray 对象
    """
    if isinstance(lst, (list, tuple)):
        if len(lst) == 0:
            return False
        if isinstance(lst[0], (list, tuple)):
            length = len(lst[0])
            for sublst in lst:
                if not isinstance(sublst, (list, tuple)) or len(sublst) != length:
                    return False
                if not is_valid_list(sublst):
                    return False
            return True
        elif isinstance(lst[0], numbers.Number):
            for item in lst:
                if not isinstance(item, numbers.Number):
                    return False
            return True
        else:
            return False
                
    else:
        return False
              

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
        if dtype and dtype not in dtypes:
            raise TypeError('Could not find %s in dtypes' % (dtype))

        if isinstance(other, NDArray):
            if dtype is None:
                dtype = other._dtype
            if device is None:
                device = other.device

            array = self.make(shape=other._shape, 
                                device=device, 
                                dtype=dtype)
            
            array._device.from_handle(other._handle, array._handle)
            self._init(array)

        elif type(other).__module__ == "numpy" and type(other).__name__ == 'ndarray':
            try:
                import numpy as np
                print("import numpy")
            except ModuleNotFoundError as mnf:
                raise mnf
            device = device if device else default_device()
            if dtype:
                other = other.astype(dtype)
            dtype = str(other.dtype)
            array = self.make(shape=other.shape, device=device, dtype=dtype)        
            array._device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
            
        elif isinstance(other, list):
            device = device if device else default_device()
            dtype = dtype if dtype else default_dtype()

            if not is_numeric_list(other):
                raise ValueError("Invalid item in list! The item in list must be numeric")
            if not is_valid_list(other):
                raise ValueError("Invalid list")
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
            if "int" in dtype:
                other = list(map(int, other))
            array._device.from_pylist(other, array._handle)
            self._init(array)

        else:
            try:
                dtype = dtype if dtype else default_dtype()
                device = device if device else default_device()
                import numpy as np
                array = NDArray(np.array(other, dtype=dtype), device=device, dtype=dtype)
                self._init(array)
            except ModuleNotFoundError as mnf:
                raise mnf
            except ValueError as r:
                raise r
        
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
    
    def astype(self, dtype):
        if dtype is None:
            dtype = default_dtype()
        return NDArray(self, dtype=dtype, device=self._device)

    def to(self, device):
        device = device if device else default_device()
        if device == self._device:
            return self
        else:
            return NDArray(self, dtype=self.dtype, device=device)

    ### Properies and string representations

    def __repr__(self) -> str:
        return "NDArray(" + self.tonumpy().__str__() + f", device={self._device}, dtype={self._dtype})"
    
    def __str__(self) -> str:
        return self.tonumpy().__str__()

    def tonumpy(self):
        return self._device.to_numpy(self._handle, self._shape, self._strides, self._offset)
    
    def tolist(self):
        try:
            import numpy as np
            temp_ndarray = self.tonumpy()
            return temp_ndarray.tolist()
        except ModuleNotFoundError:
            return self._device.to_list(self._handle, self._shape, self._strides, self._offset)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape
    
    @property
    def strides(self):
        return self._strides
    
    @property
    def device(self):
        return self._device
    
    @property
    def ndim(self):
        return len(self._shape)
    
    @property
    def size(self):
        return math.prod(self._shape)

    ### Basic array manipulation
    def fill(self, value):
        "Fill with a constant value"
        self._device.fill(self._handle, value)
