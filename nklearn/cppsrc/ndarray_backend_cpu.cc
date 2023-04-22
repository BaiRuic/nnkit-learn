#include <pybind11/numpy.h>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>

namespace nklearn{
    namespace cpu{

#define ALIGNMENT 256
#define TILE 8

template<typename T>
class AlignedArray{
public:
    size_t size;
    T * ptr;

    AlignedArray(const size_t size){
        int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * sizeof(T));
        if (ret == 1)
            throw std::bad_alloc();
        this->size = size; 
    }

    ~AlignedArray(){
        free(ptr);
    }

    size_t ptr_as_int(){
        return (size_t)ptr;
    }
};

template<typename T>
void Fill(AlignedArray<T> *out, T val){
    /*
    * Fill the values of an aligned value with val
    */
   for (size_t i=0; i < out->size; i++ ){
    out->ptr[i] = val;
   }
}

template<typename T>
void Compact(const AlignedArray<T>& a, AlignedArray<T>* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  size_t max_loop = strides.size();
  size_t cnt = 0;
  std::vector<uint16_t> stack;

  // 将原始数组中的元素复制到压缩后数组中的指定位置
  auto func = [&](std::vector<uint16_t>& stack) {
    size_t idx = offset;
    for(size_t i=0; i < stack.size(); i++){
      idx += strides[i] * stack[i];
    }
    (*out).ptr[cnt++] = a.ptr[idx];
  };
  
  // 多重循环
  std::function<void(size_t, size_t)> nested_for_loop = [&](size_t n, size_t max_n) {
  // void nested_for_loop(size_t n, size_t max_n) {
    if (n == max_n){
      func(stack);
      return;
    }
    for (size_t i=0; i < shape[n]; i++){
      stack.push_back(i);
      nested_for_loop(n+1, max_n);
      stack.pop_back();
    }
  };

  nested_for_loop(0, max_loop); 
}

template<typename T>
void EwiseSetitem(const AlignedArray<T>& a, AlignedArray<T>* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  size_t max_loop = strides.size();
  size_t cnt = 0;
  std::vector<uint16_t> stack;

  // 将原始数组中的元素复制到压缩后数组中的指定位置
  auto func = [&](std::vector<uint16_t>& stack) {
    size_t idx = offset;
    for(size_t i=0; i < stack.size(); i++){
      idx += strides[i] * stack[i];
    }
    out->ptr[idx] = a.ptr[cnt++];
  };
  
  // 多重循环
  std::function<void(size_t, size_t)> nested_for_loop = [&](size_t n, size_t max_n) {
  // void nested_for_loop(size_t n, size_t max_n) {
    if (n == max_n){
      func(stack);
      return;
    }
    for (size_t i=0; i < shape[n]; i++){
      stack.push_back(i);
      nested_for_loop(n+1, max_n);
      stack.pop_back();
    }
  };

  nested_for_loop(0, max_loop); 
}

template<typename T>
void ScalarSetitem(const size_t size, T val, AlignedArray<T>* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  size_t max_loop = strides.size();
  std::vector<uint16_t> stack;

  // 将原始数组中的元素复制到压缩后数组中的指定位置
  auto func = [&](std::vector<uint16_t>& stack) {
    size_t idx = offset;
    for(size_t i=0; i < stack.size(); i++){
      idx += strides[i] * stack[i];
    }
    out->ptr[idx] = val;
  };
  
  // 多重循环
  std::function<void(size_t, size_t)> nested_for_loop = [&](size_t n, size_t max_n) {
  // void nested_for_loop(size_t n, size_t max_n) {
    if (n == max_n){
      func(stack);
      return;
    }
    for (size_t i=0; i < shape[n]; i++){
      stack.push_back(i);
      nested_for_loop(n+1, max_n);
      stack.pop_back();
    }
  };

  nested_for_loop(0, max_loop); 
}

template<typename T1, typename T2, typename T3>
void EwiseAdd(const AlignedArray<T1>& a, const AlignedArray<T2>& b, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (T3)(a.ptr[i] + b.ptr[i]);
  }
}

template<typename T1, typename T2, typename T3>
void ScalarAdd(const AlignedArray<T1>& a, T2 val, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<T3>(a.ptr[i] + val);
  }
}


template<typename T1, typename T2, typename T3>
void EwiseMul(const AlignedArray<T1>& a, const AlignedArray<T2>& b, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<T3>(a.ptr[i] * b.ptr[i]);
  }
}

template<typename T1, typename T2, typename T3>
void ScalarMul(const AlignedArray<T1>& a, T2 val, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a multiply the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<T3>(a.ptr[i] * val);
  }
}

template<typename T1, typename T2, typename T3>
void EwiseDiv(const AlignedArray<T1>& a, const AlignedArray<T2>& b, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<T3>(a.ptr[i] / b.ptr[i]);
  }
}

template<typename T1, typename T2, typename T3>
void ScalarDiv(const AlignedArray<T1>& a, T2 val, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a multiply the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<T3>(a.ptr[i] / val);
  }
}

template<typename T>
void ScalarPower(const AlignedArray<T>& a, int64_t val, AlignedArray<T>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a multiply the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

// TODO:此处T3应该是T1或者T2的一种
template<typename T1, typename T2, typename T3>
void EwiseMaximum(const AlignedArray<T1>& a, const AlignedArray<T2>& b, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(static_cast<T3>(a.ptr[i]), static_cast<T3>(b.ptr[i]));
  }
}

template<typename T1, typename T2, typename T3>
void ScalarMaximum(const AlignedArray<T1>& a, T2 val, AlignedArray<T3>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a multiply the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(static_cast<T3>(a.ptr[i]), static_cast<T3>(val));
  }
}

template<typename T>
void EwiseEq(const AlignedArray<T>& a, const AlignedArray<T>& b, AlignedArray<uint8_t>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == b.ptr[i];
  }
}

template<typename T>
void ScalarEq(const AlignedArray<T>& a, T val, AlignedArray<uint8_t>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a multiply the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == val;
  }
}

template<typename T1, typename T2>
void EwiseGe(const AlignedArray<T1>& a, const AlignedArray<T2>& b, AlignedArray<uint8_t>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= b.ptr[i];
  }
}

template<typename T1, typename T2>
void ScalarGe(const AlignedArray<T1>& a, T2 val, AlignedArray<uint8_t>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a multiply the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= val;
  }
}


template<typename T1, typename T2>
void EwiseLog(const AlignedArray<T1>& a, AlignedArray<T2>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

template<typename T>
void EwiseExp(const AlignedArray<T>& a, AlignedArray<T>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

template<typename T1, typename T2>
void EwiseTanh(const AlignedArray<T1>& a, AlignedArray<T2>* out) {
  /**
   * Set entries in out to be the mul of corresponding entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}


template<typename T1, typename T2, typename T3>
void Matmul(const AlignedArray<T1>& a, const AlignedArray<T2>& b, AlignedArray<T3>* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */
  for (size_t i=0; i < m; i++){
    for (size_t j=0; j < p; j++){
      T3 temp_val = 0;
      for (size_t k=0; k <n; k++)
        temp_val += a.ptr[i * n + k] * b.ptr[k * p + j];
      out->ptr[i * p + j] = temp_val;
    }
  }
}

template<typename T1, typename T2, typename T3>
inline void AlignedDot(const T1* __restrict__ a,
                       const T2* __restrict__ b,
                       T3* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const T1*)__builtin_assume_aligned(a, TILE * sizeof(T1));
  b = (const T2*)__builtin_assume_aligned(b, TILE * sizeof(T2));
  out = (T3*)__builtin_assume_aligned(out, TILE * sizeof(T3));

  for(size_t i=0; i < TILE; i++)
    for(size_t j=0; j<TILE; j++){
      T3 temp_val = 0;
      for(size_t k=0; k<TILE; k++)
        temp_val += a[i * TILE + k] * b[k * TILE + j];
      out[i * TILE + j] += temp_val;
    }
}

template<typename T1, typename T2, typename T3>
void MatmulTiled(const AlignedArray<T1>& a, const AlignedArray<T2>& b, AlignedArray<T3>* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  size_t totalItems = m * p;
  for(size_t i = 0; i < totalItems; i++)
    (*out).ptr[i] = 0;
  for (size_t i=0; i<m/TILE; i++)
    for(size_t j=0; j<p/TILE; j++)
      for(size_t k=0; k<n/TILE; k++)
        AlignedDot(&a.ptr[i * n * TILE + k * TILE * TILE], 
                   &b.ptr[k * p * TILE + j * TILE * TILE], 
                   &out->ptr[i * p * TILE + j * TILE * TILE]);
}

template<typename T>
void ReduceMax(const AlignedArray<T>& a, AlignedArray<T>* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  size_t cnt = 0;
  for(size_t i=0; i < a.size; i+=reduce_size){
    T temp_val = a.ptr[i];
    for(size_t j=1; j<reduce_size; j++)
      temp_val = std::max(temp_val, a.ptr[i+j]);
    out->ptr[cnt++] = temp_val;
  }
}

template<typename T>
void ReduceSum(const AlignedArray<T>& a, AlignedArray<T>* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  size_t cnt = 0;
  for(size_t i=0; i < a.size; i+=reduce_size){
    T temp_val = a.ptr[i];
    for(size_t j=1; j<reduce_size; j++)
      temp_val += a.ptr[i+j];
    out->ptr[cnt++] = temp_val;
  }
}


namespace py = pybind11;

template<typename T>
void from_numpy(py::array_t<T> a, AlignedArray<T> * out){
    std::memcpy(out->ptr, a.request().ptr, out->size * sizeof(T));
}

template<typename T>
void from_pylist(std::vector<T> a, AlignedArray<T> * out){
    std::memcpy(out->ptr, a.data(), out->size * sizeof(T));
}

template<typename T1, typename T2>
void from_handle(AlignedArray<T1> * a, AlignedArray<T2> * out){
    for(size_t i = 0; i < out->size; i++)
        out->ptr[i] = (T2)a->ptr[i];
}


template<typename T>
void declare_array(py::module &m, const std::string &typestr) {
    using Class = AlignedArray<T>;
    std::string pyclass_name = std::string("Array_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str())
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("ptr", &Class::ptr_as_int)
        .def_readonly("size", &Class::size);
}

template<typename T>
py::array_t<T> to_numpy(const AlignedArray<T>& a,
                        std::vector<size_t> shape, 
                        std::vector<size_t> strides, size_t offset){
    std::vector<size_t> real_strides = strides;
    std::transform(real_strides.begin(), real_strides.end(), real_strides.begin(),
        [](size_t& c) { return c * sizeof(T); }); 
 
    return py::array_t<T>(shape, real_strides, a.ptr+offset); 
}


template<typename T>
py::list arrayToList(py::array_t<T> array) {
    
    auto buf = array.request();
    T * ptr = (T*) buf.ptr;
    size_t itemsize = buf.itemsize;
    if (buf.ndim == 1){
        py::list lst;
        for (size_t i=0; i<buf.shape[0]; i++)
            lst.append(ptr[i * buf.strides[0] / itemsize]);
        return lst;
    }else{
        py::list inner;
        // size_t stride = buf.strides[0] / sizeof(T);
        for (size_t i=0; i<buf.shape[0]; i++){
            py::array_t<T> sub_arr = py::array_t<T>(
                    std::vector<size_t>(buf.shape.begin() + 1, buf.shape.end()),
                    std::vector<size_t>(buf.strides.begin() + 1, buf.strides.end()),
                    ptr + i * buf.strides[0] / itemsize
                );
            inner.append(arrayToList<T>(sub_arr));
        }
        return inner;
    }
}

template<typename T>
py::list to_list(const AlignedArray<T>& a,
                    std::vector<size_t> shape, 
                    std::vector<size_t> strides, size_t offset){
        
    py::array_t<T> arr = to_numpy<T>(a, shape, strides, offset);
    return arrayToList<T>(arr);
}

} //namespace cpu
} // namespace nklearn

PYBIND11_MODULE(ndarray_backend_cpu, m){
    namespace py = pybind11;
    using namespace nklearn;
    using namespace cpu;
    
    m.attr("__device_name__") = "cpu";
    m.attr("__tile_size__") = TILE;
    
    // 暴露多种类型Array, 类名为 Array_float32
    //declare_array<_Float16>(m, "float16");
    declare_array<_Float32>(m, "float32");
    declare_array<_Float64>(m, "float64");
    declare_array<int8_t>(m, "int8");
    declare_array<int16_t>(m, "int16");
    declare_array<int32_t>(m, "int32");
    declare_array<int64_t>(m, "int64");
    declare_array<uint8_t>(m, "uint8");
    declare_array<uint16_t>(m, "uint16");
    declare_array<uint32_t>(m, "uint32");
    declare_array<uint64_t>(m, "uint64");
    
    //m.def("from_numpy", from_numpy<_Float16>);
    //m.def("from_pylist", from_pylist<_Float16>);
    //m.def("from_handle", from_handle<T1, T2>);
    // m.def("to_numpy", to_numpy<_Float16>);
    // m.def("to_list", to_list<_Float16>);
    // m.def("fill", Fill<_Float64>);
    // m.def("compact", Compact<_Float32>);


    // m.def("ewise_setitem", EwiseSetitem<_Float32>);
    // m.def("scalar_setitem", ScalarSetitem<_Float32>);
    // m.def("ewise_add", EwiseAdd<_Float32>);
    // m.def("scalar_add", ScalarAdd<_Float32>);
    // m.def("ewise_mul", EwiseMul<_Float32>);
    // m.def("scalar_mul", ScalarMul<_Float32>);
    // m.def("ewise_div", EwiseDiv<_Float32>);
    // m.def("scalar_div", ScalarDiv<_Float32>);
    // m.def("scalar_power", ScalarPower<_Float32>);

    // m.def("ewise_maximum", EwiseMaximum<_Float32>);
    // m.def("scalar_maximum", ScalarMaximum<_Float32>);
    
    // m.def("ewise_eq", EwiseEq<_Float32>);
    // m.def("scalar_eq", ScalarEq<_Float32>);
    // m.def("ewise_ge", EwiseGe<_Float32>);
    // m.def("scalar_ge", ScalarGe<_Float32>);

    // m.def("ewise_log", EwiseLog<_Float32>);
    // m.def("ewise_exp", EwiseExp<_Float32>);
    // m.def("ewise_tanh", EwiseTanh<_Float32>);

    // m.def("matmul", Matmul<_Float32>);
    // m.def("matmul_tiled", MatmulTiled<_Float32>);

    // m.def("reduce_max", ReduceMax<_Float32>);
    // m.def("reduce_sum", ReduceSum<_Float32>);

    #include"templatedCode.cpp"
}