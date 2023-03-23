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




namespace py = pybind11;

template<typename T>
void from_numpy(py::array_t<T> a, AlignedArray<T> * out){
    std::memcpy(out->ptr, a.request().ptr, out->size * sizeof(T));
}

template<typename T>
void from_pylist(std::vector<T> a, AlignedArray<T> * out){
    std::memcpy(out->ptr, a.data(), out->size * sizeof(T));
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
        return lst
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
    m.def("fill", Fill<int32_t>);
}