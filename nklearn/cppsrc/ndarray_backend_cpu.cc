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
  /// BEGIN YOUR SOLUTION
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
  /// END YOUR SOLUTION
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
    m.def("from_numpy", from_numpy<_Float32>);
    m.def("from_numpy", from_numpy<_Float64>);
    m.def("from_numpy", from_numpy<int8_t>);
    m.def("from_numpy", from_numpy<int16_t>);
    m.def("from_numpy", from_numpy<int32_t>);
    m.def("from_numpy", from_numpy<int64_t>);
    m.def("from_numpy", from_numpy<uint8_t>);
    m.def("from_numpy", from_numpy<uint16_t>);
    m.def("from_numpy", from_numpy<uint32_t>);
    m.def("from_numpy", from_numpy<uint64_t>);

    //m.def("from_pylist", from_pylist<_Float16>);
    m.def("from_pylist", from_pylist<_Float32>);
    m.def("from_pylist", from_pylist<_Float64>);
    m.def("from_pylist", from_pylist<int8_t>);
    m.def("from_pylist", from_pylist<int16_t>);
    m.def("from_pylist", from_pylist<int32_t>);
    m.def("from_pylist", from_pylist<int64_t>);
    m.def("from_pylist", from_pylist<uint8_t>);
    m.def("from_pylist", from_pylist<uint16_t>);
    m.def("from_pylist", from_pylist<uint32_t>);
    m.def("from_pylist", from_pylist<uint64_t>);

        //m.def("from_handle", from_handle<T1, T2>);
    m.def("from_handle", from_handle<_Float32, _Float32>);
    m.def("from_handle", from_handle<_Float32, _Float64>);
    m.def("from_handle", from_handle<_Float32, int8_t>);
    m.def("from_handle", from_handle<_Float32, int16_t>);
    m.def("from_handle", from_handle<_Float32, int32_t>);
    m.def("from_handle", from_handle<_Float32, int64_t>);
    m.def("from_handle", from_handle<_Float32, uint8_t>);
    m.def("from_handle", from_handle<_Float32, uint16_t>);
    m.def("from_handle", from_handle<_Float32, uint32_t>);
    m.def("from_handle", from_handle<_Float32, uint64_t>);
    m.def("from_handle", from_handle<_Float64, _Float32>);
    m.def("from_handle", from_handle<_Float64, _Float64>);
    m.def("from_handle", from_handle<_Float64, int8_t>);
    m.def("from_handle", from_handle<_Float64, int16_t>);
    m.def("from_handle", from_handle<_Float64, int32_t>);
    m.def("from_handle", from_handle<_Float64, int64_t>);
    m.def("from_handle", from_handle<_Float64, uint8_t>);
    m.def("from_handle", from_handle<_Float64, uint16_t>);
    m.def("from_handle", from_handle<_Float64, uint32_t>);
    m.def("from_handle", from_handle<_Float64, uint64_t>);
    m.def("from_handle", from_handle<_Float64, _Float32>);
    m.def("from_handle", from_handle<_Float64, _Float64>);
    m.def("from_handle", from_handle<_Float64, int8_t>);
    m.def("from_handle", from_handle<_Float64, int16_t>);
    m.def("from_handle", from_handle<_Float64, int32_t>);
    m.def("from_handle", from_handle<_Float64, int64_t>);
    m.def("from_handle", from_handle<_Float64, uint8_t>);
    m.def("from_handle", from_handle<_Float64, uint16_t>);
    m.def("from_handle", from_handle<_Float64, uint32_t>);
    m.def("from_handle", from_handle<_Float64, uint64_t>);
    m.def("from_handle", from_handle<int8_t, _Float32>);
    m.def("from_handle", from_handle<int8_t, _Float64>);
    m.def("from_handle", from_handle<int8_t, int8_t>);
    m.def("from_handle", from_handle<int8_t, int16_t>);
    m.def("from_handle", from_handle<int8_t, int32_t>);
    m.def("from_handle", from_handle<int8_t, int64_t>);
    m.def("from_handle", from_handle<int8_t, uint8_t>);
    m.def("from_handle", from_handle<int8_t, uint16_t>);
    m.def("from_handle", from_handle<int8_t, uint32_t>);
    m.def("from_handle", from_handle<int8_t, uint64_t>);
    m.def("from_handle", from_handle<int16_t, _Float32>);
    m.def("from_handle", from_handle<int16_t, _Float64>);
    m.def("from_handle", from_handle<int16_t, int8_t>);
    m.def("from_handle", from_handle<int16_t, int16_t>);
    m.def("from_handle", from_handle<int16_t, int32_t>);
    m.def("from_handle", from_handle<int16_t, int64_t>);
    m.def("from_handle", from_handle<int16_t, uint8_t>);
    m.def("from_handle", from_handle<int16_t, uint16_t>);
    m.def("from_handle", from_handle<int16_t, uint32_t>);
    m.def("from_handle", from_handle<int16_t, uint64_t>);
    m.def("from_handle", from_handle<int32_t, _Float32>);
    m.def("from_handle", from_handle<int32_t, _Float64>);
    m.def("from_handle", from_handle<int32_t, int8_t>);
    m.def("from_handle", from_handle<int32_t, int16_t>);
    m.def("from_handle", from_handle<int32_t, int32_t>);
    m.def("from_handle", from_handle<int32_t, int64_t>);
    m.def("from_handle", from_handle<int32_t, uint8_t>);
    m.def("from_handle", from_handle<int32_t, uint16_t>);
    m.def("from_handle", from_handle<int32_t, uint32_t>);
    m.def("from_handle", from_handle<int32_t, uint64_t>);
    m.def("from_handle", from_handle<int64_t, _Float32>);
    m.def("from_handle", from_handle<int64_t, _Float64>);
    m.def("from_handle", from_handle<int64_t, int8_t>);
    m.def("from_handle", from_handle<int64_t, int16_t>);
    m.def("from_handle", from_handle<int64_t, int32_t>);
    m.def("from_handle", from_handle<int64_t, int64_t>);
    m.def("from_handle", from_handle<int64_t, uint8_t>);
    m.def("from_handle", from_handle<int64_t, uint16_t>);
    m.def("from_handle", from_handle<int64_t, uint32_t>);
    m.def("from_handle", from_handle<int64_t, uint64_t>);
    m.def("from_handle", from_handle<uint8_t, _Float32>);
    m.def("from_handle", from_handle<uint8_t, _Float64>);
    m.def("from_handle", from_handle<uint8_t, int8_t>);
    m.def("from_handle", from_handle<uint8_t, int16_t>);
    m.def("from_handle", from_handle<uint8_t, int32_t>);
    m.def("from_handle", from_handle<uint8_t, int64_t>);
    m.def("from_handle", from_handle<uint8_t, uint8_t>);
    m.def("from_handle", from_handle<uint8_t, uint16_t>);
    m.def("from_handle", from_handle<uint8_t, uint32_t>);
    m.def("from_handle", from_handle<uint8_t, uint64_t>);
    m.def("from_handle", from_handle<uint16_t, _Float32>);
    m.def("from_handle", from_handle<uint16_t, _Float64>);
    m.def("from_handle", from_handle<uint16_t, int8_t>);
    m.def("from_handle", from_handle<uint16_t, int16_t>);
    m.def("from_handle", from_handle<uint16_t, int32_t>);
    m.def("from_handle", from_handle<uint16_t, int64_t>);
    m.def("from_handle", from_handle<uint16_t, uint8_t>);
    m.def("from_handle", from_handle<uint16_t, uint16_t>);
    m.def("from_handle", from_handle<uint16_t, uint32_t>);
    m.def("from_handle", from_handle<uint16_t, uint64_t>);
    m.def("from_handle", from_handle<uint32_t, _Float32>);
    m.def("from_handle", from_handle<uint32_t, _Float64>);
    m.def("from_handle", from_handle<uint32_t, int8_t>);
    m.def("from_handle", from_handle<uint32_t, int16_t>);
    m.def("from_handle", from_handle<uint32_t, int32_t>);
    m.def("from_handle", from_handle<uint32_t, int64_t>);
    m.def("from_handle", from_handle<uint32_t, uint8_t>);
    m.def("from_handle", from_handle<uint32_t, uint16_t>);
    m.def("from_handle", from_handle<uint32_t, uint32_t>);
    m.def("from_handle", from_handle<uint32_t, uint64_t>);
    m.def("from_handle", from_handle<uint64_t, _Float32>);
    m.def("from_handle", from_handle<uint64_t, _Float64>);
    m.def("from_handle", from_handle<uint64_t, int8_t>);
    m.def("from_handle", from_handle<uint64_t, int16_t>);
    m.def("from_handle", from_handle<uint64_t, int32_t>);
    m.def("from_handle", from_handle<uint64_t, int64_t>);
    m.def("from_handle", from_handle<uint64_t, uint8_t>);
    m.def("from_handle", from_handle<uint64_t, uint16_t>);
    m.def("from_handle", from_handle<uint64_t, uint32_t>);
    m.def("from_handle", from_handle<uint64_t, uint64_t>);

    //m.def("to_numpy", to_numpy<_Float16>);
    m.def("to_numpy", to_numpy<_Float32>);
    m.def("to_numpy", to_numpy<_Float64>);
    m.def("to_numpy", to_numpy<int8_t>);
    m.def("to_numpy", to_numpy<int16_t>);
    m.def("to_numpy", to_numpy<int32_t>);
    m.def("to_numpy", to_numpy<int64_t>);
    m.def("to_numpy", to_numpy<uint8_t>);
    m.def("to_numpy", to_numpy<uint16_t>);
    m.def("to_numpy", to_numpy<uint32_t>);
    m.def("to_numpy", to_numpy<uint64_t>);

    //m.def("to_list", to_list<_Float16>);
    m.def("to_list", to_list<_Float32>);
    m.def("to_list", to_list<_Float64>);
    m.def("to_list", to_list<int8_t>);
    m.def("to_list", to_list<int16_t>);
    m.def("to_list", to_list<int32_t>);
    m.def("to_list", to_list<int64_t>);
    m.def("to_list", to_list<uint8_t>);
    m.def("to_list", to_list<uint16_t>);
    m.def("to_list", to_list<uint32_t>);
    m.def("to_list", to_list<uint64_t>);

    m.def("fill", Fill<_Float32>);
    m.def("fill", Fill<_Float64>);
    m.def("fill", Fill<int8_t>);
    m.def("fill", Fill<int16_t>);
    m.def("fill", Fill<int32_t>);
    m.def("fill", Fill<int64_t>);
    m.def("fill", Fill<uint8_t>);
    m.def("fill", Fill<uint16_t>);
    m.def("fill", Fill<uint32_t>);
    m.def("fill", Fill<uint64_t>);

    m.def("compact", Compact<_Float32>);
    m.def("compact", Compact<_Float64>);
    m.def("compact", Compact<int8_t>);
    m.def("compact", Compact<int16_t>);
    m.def("compact", Compact<int32_t>);
    m.def("compact", Compact<int64_t>);
    m.def("compact", Compact<uint8_t>);
    m.def("compact", Compact<uint16_t>);
    m.def("compact", Compact<uint32_t>);
    m.def("compact", Compact<uint64_t>);
}