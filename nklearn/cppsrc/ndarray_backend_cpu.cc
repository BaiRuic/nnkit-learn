#include <pybind11/numpy.h>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>

namespace nklearn{
    namespace cpu{

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


class AlignedArray{
public:
    size_t size;
    scalar_t * ptr;

    AlignedArray(const size_t size){
        int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
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


} //namespace cpu
} // namespace nklearn

PYBIND11_MODULE(ndarray_backend_cpu, m){
    namespace py = pybind11;
    using namespace nklearn;
    using namespace cpu;
    
    m.attr("__device_name__") = "cpu";
    m.attr("__tile_size__") = TILE;
    
    py::class_<AlignedArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("ptr", &AlignedArray::ptr_as_int)
        .def_readonly("size", &AlignedArray::size);
    

    m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray * out){
        std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
    });


    m.def("from_pylist", [](py::list a, AlignedArray *out){
        std::memcpy(out->ptr, a.cast<std::vector<scalar_t>>().data(), out->size);
    });

    m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape, 
                        std::vector<size_t> strides, size_t offset){
                            std::vector<size_t> real_strides = strides;
                            std::transform(real_strides.begin(), real_strides.end(), real_strides.begin(),
                                            [](size_t& c) { return c * ELEM_SIZE; }); 
                                            
                            return py::array_t<scalar_t>(shape, real_strides, a.ptr+offset);
                        });
    
}