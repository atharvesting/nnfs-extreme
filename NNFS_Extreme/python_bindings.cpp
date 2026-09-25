#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "NN.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nnfs_extreme, m) {
    m.doc() = "Python bindings for NNFS_Extreme";

    py::enum_<Threading>(m, "Threading")
        .value("Single", Threading::Single)
        .value("Multi", Threading::Multi);

    py::class_<Config>(m, "Config")
        .def(py::init([](std::vector<int> topology, Threading threading, int threads) {
            return Config(30, 128, 30.0F, std::move(topology), threading, threads);
        }), py::arg("topology"), py::arg("thread_state"), py::arg("thread_count"));

    py::class_<Network>(m, "Network")
        .def(py::init<const std::string&, Config&>(), py::arg("model_path"), py::arg("config"))
        .def("feedforward", [](const Network& net, py::array_t<float, py::array::c_style | py::array::forcecast> input) {
            auto buf = input.request();
            if (buf.ndim != 1)
                throw py::value_error("input must be a one-dimensional float array");
            std::vector<float> data(static_cast<float*>(buf.ptr),
                                    static_cast<float*>(buf.ptr) + buf.size);
            Matrix<float> result = net.feedforward(Matrix<float>(data.size(), 1, data));
            return std::vector<float>(result.rix.data(), result.rix.data() + result.size());
        }, py::arg("input"));
}
