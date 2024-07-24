#include "cublas_common.h"
#include "utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cublasLt.h>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(error) << std::endl;                     \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": "   \
                << cublasGetStatusString(status) << std::endl;                 \
      throw std::runtime_error("cuBLAS error");                                \
    }                                                                          \
  } while (0)

// init and clean up cublas
static int init_cublas() {
  if (cublasLtCreate(&cublaslt_handle) != CUBLAS_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create cuBLASLt handle");
    return -1;
  }
  if (cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size) != cudaSuccess) {
    cublasLtDestroy(cublaslt_handle);
    cublaslt_handle = nullptr;
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to allocate cuBLASLt workspace");
    return -1;
  }
  return 0;
}

static void cleanup_cublas() {
  if (cublaslt_workspace) {
    cudaFree(cublaslt_workspace);
    cublaslt_workspace = nullptr;
  }
  if (cublaslt_handle) {
    cublasLtDestroy(cublaslt_handle);
    cublaslt_handle = nullptr;
  }
}

// Load an array pointer for a cupy array
void *get_array_pointer(PyObject *arr, const char *name) {
  LOG("Entering get_array_pointer for " << name);

  if (!arr) {
    std::cerr << name << " is NULL" << std::endl;
    PyErr_SetString(PyExc_TypeError, "Array object is NULL");
    return nullptr;
  }

  LOG("Array object type: " << Py_TYPE(arr)->tp_name);

  if (!PyObject_HasAttrString(arr, "data")) {
    std::cerr << name << " has no 'data' attribute" << std::endl;
    PyErr_SetString(PyExc_AttributeError,
                    "Array object has no attribute 'data'");
    return nullptr;
  }

  PyObject *data_attr = PyObject_GetAttrString(arr, "data");
  if (!data_attr) {
    std::cerr << "Failed to get 'data' attribute for " << name << std::endl;
    PyErr_Print();
    return nullptr;
  }

  std::string data_type_name = Py_TYPE(data_attr)->tp_name;
  LOG("Type of data attribute for " << name << ": " << data_type_name);

  // Handle MemoryPointer object
  if (data_type_name.find("MemoryPointer") != std::string::npos) {
    PyObject *ptr_attr = PyObject_GetAttrString(data_attr, "ptr");
    if (!ptr_attr) {
      std::cerr << "Failed to get 'ptr' attribute from MemoryPointer for "
                << name << std::endl;
      Py_DECREF(data_attr);
      PyErr_Print();
      return nullptr;
    }

    void *ptr = PyLong_AsVoidPtr(ptr_attr);
    Py_DECREF(ptr_attr);
    Py_DECREF(data_attr);

    if (PyErr_Occurred()) {
      std::cerr << "Error occurred while getting pointer for " << name
                << std::endl;
      PyErr_Print();
      return nullptr;
    }

    if (!ptr) {
      std::cerr << "Got NULL pointer for " << name << std::endl;
      PyErr_SetString(PyExc_ValueError, "Got NULL pointer from array data");
      return nullptr;
    }

    LOG("Successfully got pointer for " << name << ": " << ptr);
    return ptr;
  } else {
    std::cerr << "Unexpected data attribute type for " << name << ": "
              << data_type_name << std::endl;
    Py_DECREF(data_attr);
    PyErr_SetString(PyExc_TypeError, "Unexpected data attribute type");
    return nullptr;
  }
}
