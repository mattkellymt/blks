#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include "lib_cuda.cuh"

static PyObject* py_parse_u32(PyObject* self, PyObject* args) {
    const char* str;
    if (!PyArg_ParseTuple(args, "s", &str)) {
        return nullptr;
    }
    pair res = parse_u32(str);
    if (res.ec != 0) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse input as uint32_t");
        return nullptr;
    }
    return PyLong_FromUnsignedLong(res.u32);
}

static PyObject* py_parse_i32(PyObject* self, PyObject* args) {
    const char* str;
    if (!PyArg_ParseTuple(args, "s", &str)) {
        return nullptr;
    }
    pair res = parse_i32(str);
    if (res.ec != 0) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse input as int32_t");
        return nullptr;
    }
    return PyLong_FromLong(res.i32);
}

static PyObject* py_parse_f32(PyObject* self, PyObject* args) {
    const char* str;
    if (!PyArg_ParseTuple(args, "s", &str)) {
        return nullptr;
    }
    pair res = parse_f32(str);
    if (res.ec != 0) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse input as float");
        return nullptr;
    }
    return PyFloat_FromDouble(res.f32);
}

static PyObject* py_launch_threads(PyObject* self, PyObject* args) {
    unsigned int grid_size;
    unsigned int block_size;
    if (!PyArg_ParseTuple(args, "II", &grid_size, &block_size)) {
        return nullptr;
    }
    return PyLong_FromLong(launch_threads(grid_size, block_size));
}

static PyMethodDef LibCudaMethods[] = {
    {"parse_u32", py_parse_u32, METH_VARARGS, ""},
    {"parse_i32",  py_parse_i32,  METH_VARARGS, ""},
    {"parse_f32",  py_parse_f32,  METH_VARARGS, ""},
    {"launch_threads", py_launch_threads, METH_VARARGS, ""},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef libcudamodule = {
    PyModuleDef_HEAD_INIT,
    "mod_cuda",
    "",
    -1,
    LibCudaMethods
};

PyMODINIT_FUNC PyInit_mod_cuda(void) {
    return PyModule_Create(&libcudamodule);
}
