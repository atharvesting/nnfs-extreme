#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <NN.hpp>

namespace py = pybind11;

PYBIND11_MODULE(nnfs_extreme, m) {
    m.doc() = "Python Bindings for NNFS_Extreme";
    py::class_<Network>(m, "Network")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        
        .def("feedforward", [](const Network& net, py::array_t<float> input) {
            auto buf = input.request();
            std::vector<float> data((float*)buf.ptr, (float*)buf.ptr + buf.size);
            Matrix<float> mat(data.size(), 1, data);
            Matrix<float> result = net.feedforward(mat);

            std::vector<float> out(result.rix.data(), result.rix.data() + result.size());
            return out;
        }, py::arg("input"));
}