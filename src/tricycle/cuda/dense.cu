/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits> // std::bool_constant
// llmc internal imports
#include "cublas_common.h"
#include "cuda_common.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cublasLt.h>
#include <string>

// Global variables

// Wrapper around cublasLtMatmul that is meant to support everything we need in
// llm.c https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
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

#define SAFE_GET_ARRAY_POINTER(arr, ptr_var)                                   \
  data_attr = PyObject_GetAttrString(arr, "data");                             \
  if (!data_attr) {                                                            \
    PyErr_SetString(PyExc_AttributeError,                                      \
                    "Array object has no attribute 'data'");                   \
    return NULL;                                                               \
  }                                                                            \
  ptr = PyLong_AsVoidPtr(data_attr);                                           \
  Py_DECREF(data_attr);                                                        \
  if (PyErr_Occurred()) {                                                      \
    return NULL;                                                               \
  }                                                                            \
  ptr_var = static_cast<float *>(ptr);

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
void *get_array_pointer(PyObject *arr, const char *name) {
  std::cout << "Entering get_array_pointer for " << name << std::endl;

  if (!arr) {
    std::cerr << name << " is NULL" << std::endl;
    PyErr_SetString(PyExc_TypeError, "Array object is NULL");
    return nullptr;
  }

  std::cout << "Array object type: " << Py_TYPE(arr)->tp_name << std::endl;

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
  std::cout << "Type of data attribute for " << name << ": " << data_type_name
            << std::endl;

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

    std::cout << "Successfully got pointer for " << name << ": " << ptr
              << std::endl;
    return ptr;
  } else {
    std::cerr << "Unexpected data attribute type for " << name << ": "
              << data_type_name << std::endl;
    Py_DECREF(data_attr);
    PyErr_SetString(PyExc_TypeError, "Unexpected data attribute type");
    return nullptr;
  }
}

void matmul_cublaslt(floatX *d, const floatX *a, const floatX *b,
                     const floatX *bias, int m, int n, int k,
                     cudaStream_t stream = 0, bool transA = true,
                     bool transB = false, int batch_count = 0,
                     size_t strideA = 0, size_t strideB = 0,
                     size_t strideOut = 0, bool accumulate = false,
                     bool backward = false) {
  std::cout << "Entering matmul_cublaslt" << std::endl;
  std::cout << "Dimensions: m=" << m << ", n=" << n << ", k=" << k << std::endl;
  std::cout << "Pointers: d=" << d << ", a=" << a << ", b=" << b
            << ", bias=" << bias << std::endl;
  std::cout << "Other params: transA=" << transA << ", transB=" << transB
            << ", batch_count=" << batch_count << ", accumulate=" << accumulate
            << ", backward=" << backward << std::endl;

  NVTX_RANGE_FN();
  bool has_bias = (bias != NULL);
  std::cout << "has_bias: " << has_bias << std::endl;

  // check alignment
  if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 ||
      ((uintptr_t)d % 16) != 0 || (has_bias && ((uintptr_t)bias % 16) != 0)) {
    throw std::runtime_error("All cuBLASLt pointers must be aligned!");
  }
  std::cout << "Alignment check passed" << std::endl;

  // create the operation descriptor
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t ALayout = nullptr, BLayout = nullptr,
                         CLayout = nullptr, DLayout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  try {
    std::cout << "Creating matmul descriptor" << std::endl;
    CUBLAS_CHECK(
        cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));
    std::cout << "Matmul descriptor created" << std::endl;

    // Set matrix operation attributes
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    std::cout << "Setting matrix operation attributes" << std::endl;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
        (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
        (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));
    std::cout << "Matrix operation attributes set" << std::endl;

    // define matrix layouts
    std::cout << "Creating matrix layouts" << std::endl;
    if (transA) {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if (transB) {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));
    std::cout << "Matrix layouts created" << std::endl;

    // create a preference handle
    std::cout << "Creating preference handle" << std::endl;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));
    std::cout << "Preference handle created" << std::endl;

    // find a suitable algorithm
    std::cout << "Finding suitable algorithm" << std::endl;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
        preference, 1, &heuristic, &returnedResults));

    if (returnedResults == 0) {
      throw std::runtime_error("No suitable cuBLASLt algorithm found");
    }
    std::cout << "Suitable algorithm found" << std::endl;

    // set whether to accumulate or not
    std::cout << "Setting alpha and beta" << std::endl;
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    std::cout << "Alpha and beta set: alpha=" << alpha << ", beta=" << beta
              << std::endl;

    // call the matmul
    std::cout << "Calling cublasLtMatmul" << std::endl;
    CUBLAS_CHECK(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, a,
                                ALayout, b, BLayout, &beta, d, CLayout, d,
                                DLayout, &heuristic.algo, cublaslt_workspace,
                                cublaslt_workspace_size, stream));

    std::cout << "cublasLtMatmul completed successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error in matmul_cublaslt: " << e.what() << std::endl;
    // Clean up resources
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (operationDesc)
      cublasLtMatmulDescDestroy(operationDesc);
    if (ALayout)
      cublasLtMatrixLayoutDestroy(ALayout);
    if (BLayout)
      cublasLtMatrixLayoutDestroy(BLayout);
    if (CLayout)
      cublasLtMatrixLayoutDestroy(CLayout);
    if (DLayout)
      cublasLtMatrixLayoutDestroy(DLayout);
    throw; // re-throw the exception
  }

  // Clean up resources
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatmulDescDestroy(operationDesc);
  cublasLtMatrixLayoutDestroy(ALayout);
  cublasLtMatrixLayoutDestroy(BLayout);
  cublasLtMatrixLayoutDestroy(CLayout);
  cublasLtMatrixLayoutDestroy(DLayout);

  CUDA_CHECK(cudaGetLastError());
  std::cout << "Exiting matmul_cublaslt" << std::endl;
}
extern "C" void matmul_forward_cublaslt(floatX *out, floatX *inp,
                                        floatX *weight, floatX *bias, int B,
                                        int T, int C, int OC,
                                        cudaStream_t stream = 0) {
  std::cout << "Entering matmul_forward_cublaslt" << std::endl;
  std::cout << "Dimensions: B=" << B << ", T=" << T << ", C=" << C
            << ", OC=" << OC << std::endl;
  std::cout << "Pointers: out=" << out << ", inp=" << inp
            << ", weight=" << weight << ", bias=" << bias << std::endl;

  try {
    if (cublaslt_handle == nullptr) {
      throw std::runtime_error("cublaslt_handle is not initialized");
    }
    if (cublaslt_workspace == nullptr) {
      throw std::runtime_error("cublaslt_workspace is not initialized");
    }
    matmul_cublaslt(out, weight, inp, bias, OC, B * T, C, stream, true, false,
                    0, 0, 0, 0, false);
  } catch (const std::exception &e) {
    std::cerr << "Exception in matmul_forward_cublaslt: " << e.what()
              << std::endl;
    throw;
  }

  std::cout << "Exiting matmul_forward_cublaslt" << std::endl;
}

static PyObject *py_matmul_forward_cublaslt(PyObject *self, PyObject *args) {
  PyObject *output, *input, *weights, *bias;
  long B, T, C, out_shape;
  PyObject *stream_obj;

  if (!PyArg_ParseTuple(args, "OOOOllllO", &output, &input, &weights, &bias, &B,
                        &T, &C, &out_shape, &stream_obj)) {
    PyErr_Print();
    return NULL;
  }

  std::cout << "Parsed arguments in py_matmul_forward_cublaslt" << std::endl;
  std::cout << "B: " << B << ", T: " << T << ", C: " << C
            << ", out_shape: " << out_shape << std::endl;

  float *output_ptr, *input_ptr, *weights_ptr, *bias_ptr;

  output_ptr = static_cast<float *>(get_array_pointer(output, "output"));
  if (!output_ptr)
    return NULL;

  input_ptr = static_cast<float *>(get_array_pointer(input, "input"));
  if (!input_ptr)
    return NULL;

  weights_ptr = static_cast<float *>(get_array_pointer(weights, "weights"));
  if (!weights_ptr)
    return NULL;

  bias_ptr = static_cast<float *>(get_array_pointer(bias, "bias"));
  if (!bias_ptr)
    return NULL; // Handle CUDA stream

  cudaStream_t cuda_stream = nullptr;
  if (PyObject_HasAttrString(stream_obj, "ptr")) {
    PyObject *ptr_attr = PyObject_GetAttrString(stream_obj, "ptr");
    if (ptr_attr) {
      cuda_stream = reinterpret_cast<cudaStream_t>(PyLong_AsVoidPtr(ptr_attr));
      Py_DECREF(ptr_attr);
    } else {
      PyErr_Clear(); // Clear the attribute error
    }
  }

  if (cuda_stream == nullptr) {
    std::cout << "Using default stream (nullptr)" << std::endl;
  } else {
    std::cout << "Using provided stream: " << cuda_stream << std::endl;
  }
  std::cout << "Extracted CUDA stream" << std::endl;

  try {
    matmul_forward_cublaslt(output_ptr, input_ptr, weights_ptr, bias_ptr, B, T,
                            C, out_shape, cuda_stream);
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }

  std::cout << "Completed py_matmul_forward_cublaslt" << std::endl;

  Py_RETURN_NONE;
}

// Module method definitions
static PyMethodDef LlmcMethods[] = {
    {"matmul_forward_cublaslt", py_matmul_forward_cublaslt, METH_VARARGS,
     "Perform matrix multiplication using cuBLAS LT"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Module definition
static struct PyModuleDef llmcmodule = {PyModuleDef_HEAD_INIT, "llmc", NULL, -1,
                                        LlmcMethods};

static void llmc_free(void *unused) { cleanup_cublas(); }

PyMODINIT_FUNC PyInit_llmc(void) {
  PyObject *m;

  m = PyModule_Create(&llmcmodule);
  if (m == NULL)
    return NULL;

  if (init_cublas() < 0) {
    Py_DECREF(m);
    return NULL;
  }

  if (PyModule_AddFunctions(m, LlmcMethods) < 0) {
    Py_DECREF(m);
    return NULL;
  }

  if (PyModule_AddObject(
          m, "__cleanup__",
          PyCapsule_New((void *)llmc_free, "__cleanup__", NULL)) < 0) {
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
