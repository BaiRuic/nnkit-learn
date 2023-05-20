import math
import numbers
from typing import List

from ._device import default_device
from ._dtype import dtypes, default_dtype


def is_numeric_list(lst: List) -> bool:
    for el in lst:
        if isinstance(el, list):
            if not is_numeric_list(el):
                return False
        elif not isinstance(el, numbers.Number):
            return False
    return True

def flatten(lst:List) ->bool:
    """将多维List 拉平为一维
    param:
        lst: 多维list
    return:
        result:一为list
    """
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
        if "int" in self.dtype and isinstance(value, float):
            value = int(value)
        self._device.fill(self._handle, value)

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (self._strides == self.compact_strides(self._shape)
                and math.prod(self.shape) == self._handle.size)

    def compact(self):
        """ Convert a matrix to be compact """
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides, offset=0):
        """ Restride the matrix without copying memory. """
        assert len(shape) == len(strides)
        return NDArray.make(shape, strides=strides, 
                            device=self.device, 
                            handle=self._handle, 
                            offset=offset,)
    
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.
        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.
        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray : reshaped array; this will point to the same memory as the original NDArray.
        """
        # only modify the shape and stride
        if math.prod(self.shape) != math.prod(new_shape) or not self.is_compact():
            raise ValueError
        
        new_strides = NDArray.compact_strides(new_shape)
        return self.as_strided(shape=new_shape, 
                               strides=new_strides)

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permutation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.
        Args:
            new_axes (tuple): permutation order of the dimensions
        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """
        new_shape = list(range(self.ndim))
        for i in range(len(new_axes)):
            new_shape[i] = self._shape[new_axes[i]]
        

        new_strides = list(range(self.ndim))
        for i in range(len(new_axes)):
            new_strides[i] = self._strides[new_axes[i]]
        
        return self.as_strided(shape=tuple(new_shape), 
                               strides=tuple(new_strides))

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape. new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.
        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1
        Args:
            new_shape (tuple): shape to broadcast to
        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        # not add dims or add dims
        added_axes = list(range(len(new_shape) - len(self._shape)))
        for i in range(-1, -len(self._shape)-1, -1):
            if new_shape[i] != self._shape[i]:
                if self._shape[i] != 1:
                    raise ValueError
                else:
                    added_axes.append(i)

        new_strides = list(range(len(new_shape) - len(self._shape))) + list(self._strides)
        
        for ax in added_axes:
            new_strides[ax] = 0

        return self.as_strided(shape=tuple(new_shape), 
                               strides=tuple(new_strides))

    ### Get and set elements

    def process_slice(self, sl, dim):
        """ Convert a slice to an explicit start/stop/step """
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.
        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory
        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.
        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            corresponding to the subset of the matrix to get
        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = [0] * self.ndim
        new_strides = [0] * self.ndim
        new_offset = 0

        # calculate new shape
        for i in range(0, self.ndim):
            new_shape[i] = math.ceil((idxs[i].stop - idxs[i].start) / idxs[i].step)

        # calculate new strides
        for i in range(0, self.ndim):
            new_strides[i] = self._strides[i] * idxs[i].step
        
        # calculate offset
        for i in range(0, self.ndim):
            new_offset += self._strides[i] * idxs[i].start

        return self.as_strided(shape=tuple(new_shape), 
                                strides = tuple(new_strides),
                                offset=new_offset)

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert math.prod(view.shape) == math.prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                math.prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc
    
    @staticmethod
    def _type_inference(lhs, rhs) -> str:
        """类型推断，根据 运算符的左右操作数来推断运算结果的类型
        """
        if isinstance(rhs, NDArray):
            # 两者全为 intx 或者 floatx 或者 uintx时
            if rhs.dtype[:3] == lhs.dtype[:3]:
                if len(lhs.dtype) > len(lhs.other):
                    dtype = lhs.dtype
                else:
                    dtype = lhs.dtype if lhs.dtype > rhs.dtype else rhs.dtype
            else:
                # 有一个 float,那就是 float
                if "float" in lhs.dtype or "float" in rhs.dtype:
                    dtype = lhs.dtype if "float" in lhs.dtype else rhs.dtype
                else:
                    dtype = rhs.dtype if "uint" in lhs.dtype else lhs.dtype
        else:
            # 两者全为 intx(uintx) 或者 floatx 时
            if ((isinstance(rhs, float) and "float" in lhs.dtype) or
                (isinstance(rhs, int) and lhs.dtype[:3] == "int")):
                dtype = lhs.dtype
            else:
                if isinstance(rhs, float):
                    dtype = default_dtype()
                else:
                    if "float" in lhs.dtype:
                        dtype = lhs.dtype
                    else:
                        num_str = []
                        for c in lhs.dtype:
                            if c.isdigit():
                                num_str.append(c)
                        dtype = "int" + "".join(num_str)
        return dtype

    def ewise_or_scalar(self, other, ewise_func, scalar_func, dtype=None):
        """Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        dtype_ = NDArray._type_inference(self, other)
        out = NDArray.make(self.shape, device=self.device, dtype=dtype_)

        if isinstance(other, NDArray):                
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device, dtype=self.dtype)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq, dtype="uint8")

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge, dtype="uint8")

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    def log(self):
        # 如果当前数据类型为浮点型，那么需要和当前数据类型一直，否则为默认类型
        dtype_ = self.dtype if "float" in self.dtype else default_dtype()
        out = NDArray.make(self.shape, device=self.device, dtype=dtype_)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        dtype_ = self.dtype
        out = NDArray.make(self.shape, device=self.device, dtype=dtype_)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        # 数据类型应该同 log
        dtype_ = self.dtype if "float" in self.dtype else default_dtype()
        out = NDArray.make(self.shape, device=self.device, dtype=dtype_)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multiplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.
        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will re-stride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size
        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]
        dtype_ = NDArray._type_inference(self, other)
        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, self.shape[1], 1),
                )
            # Firstly, make the matrix with shape [a.shape[0] // tile, a.shape[1] // tile, tile, tile]
            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device, dtype=dtype_)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            # TODO :数据类型待推导
            out = NDArray.make((m, p), device=self.device, dtype=dtype_)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis):
        """ Return a view to the array set up for reduction functions and output array. """
        if axis is None:
            view = self.reshape((1,) * (self.ndim - 1) + (math.prod(self.shape),))
            out = NDArray.make((1,) * self.ndim, device=self.device, dtype=self.dtype)
        else:
            # 把要操作的 axis 放到最后！因为放到最后经过 compact() 之后，内存中是连续的
            # 然后按照 self.shape[axis] 一组来执行操作，该参数即为传给cpp接口的第三个参数
            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)]),
                device=self.device,
                dtype=self.dtype,
            )
        return view, out

    def sum(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


def array(a, dtype=None, device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = default_dtype() if dtype is None else dtype
    device = default_device() if device is None else device

    return NDArray(a, device=device, dtype=dtype)

def empty(shape, dtype=None, device=None):
    device = device if device is not None else default_device()
    dtype = dtype if dtype is not None else default_dtype()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype=None, device=None):
    device = device if device is not None else default_device()
    dtype = dtype if dtype is None else default_dtype()
    return device.full(shape, fill_value, dtype)

def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)