#pragma once
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <string>

#ifndef _CUDA_ERROR_
#define _CUDA_ERROR_
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif

#ifndef _CUSPARSE_ERROR_
#define _CUSPARSE_ERROR_
static inline void cusparseHandleError(cusparseStatus_t status,
                                       const char*      file,
                                       const int        line)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("\n%s in %s at line %d\n", cusparseGetErrorString(status), file,
               line);
        exit(EXIT_FAILURE);
    }
    return;
}
#define CUSPARSE_ERROR(err) (cusparseHandleError(err, __FILE__, __LINE__))
#endif

#ifndef _CUSOLVER_ERROR_
#define _CUSOLVER_ERROR_
static inline void cusolverHandleError(cusolverStatus_t status,
                                       const char*      file,
                                       const int        line)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        auto cusolverGetErrorString = [](cusolverStatus_t status) {
            switch (status) {
                case CUSOLVER_STATUS_SUCCESS:
                    return "CUSOLVER_STATUS_SUCCESS";
                case CUSOLVER_STATUS_NOT_INITIALIZED:
                    return "CUSOLVER_STATUS_NOT_INITIALIZED";
                case CUSOLVER_STATUS_ALLOC_FAILED:
                    return "CUSOLVER_STATUS_ALLOC_FAILED";
                case CUSOLVER_STATUS_INVALID_VALUE:
                    return "CUSOLVER_STATUS_INVALID_VALUE";
                case CUSOLVER_STATUS_ARCH_MISMATCH:
                    return "CUSOLVER_STATUS_ARCH_MISMATCH";
                case CUSOLVER_STATUS_EXECUTION_FAILED:
                    return "CUSOLVER_STATUS_EXECUTION_FAILED";
                case CUSOLVER_STATUS_INTERNAL_ERROR:
                    return "CUSOLVER_STATUS_INTERNAL_ERROR";
                case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                default:
                    return "UNKNOWN_ERROR";
            }
        };

        printf("\n%s in %s at line %d\n", cusolverGetErrorString(status), file,
               line);
        exit(EXIT_FAILURE);
    }
    return;
}
#define CUSOLVER_ERROR(err) (cusolverHandleError(err, __FILE__, __LINE__))
#endif

class CUDATimer
{
   public:
    CUDATimer()
    {
        CUDA_ERROR(cudaEventCreate(&m_start));
        CUDA_ERROR(cudaEventCreate(&m_stop));
    }
    ~CUDATimer()
    {
        CUDA_ERROR(cudaEventDestroy(m_start));
        CUDA_ERROR(cudaEventDestroy(m_stop));
    }
    void start(cudaStream_t stream = 0)
    {
        m_stream = stream;
        CUDA_ERROR(cudaEventRecord(m_start, m_stream));
    }
    void stop()
    {
        CUDA_ERROR(cudaEventRecord(m_stop, m_stream));
        CUDA_ERROR(cudaEventSynchronize(m_stop));
    }
    float elapsed_millis()
    {
        float elapsed = 0;
        CUDA_ERROR(cudaEventElapsedTime(&elapsed, m_start, m_stop));
        return elapsed;
    }

   private:
    cudaEvent_t  m_start, m_stop;
    cudaStream_t m_stream;
};

void cusolver_solver(const std::string&           name,
                     Eigen::SparseMatrix<double>& Q,
                     const Eigen::MatrixXd&       rhs,
                     Eigen::MatrixXd&             U)
{
    U.resize(rhs.rows(), rhs.cols());

    if (Q.IsRowMajor()) {
        printf("Error: Q is row major. No need to transpose it...\n");
        exit(EXIT_FAILURE);
    }
    auto Q_trans = Q.transpose();

    printf("| %30s | ", name.c_str());

    // create stream
    cudaStream_t stream = NULL;
    CUDA_ERROR(cudaStreamCreate(&stream));

    // create cusolver handle
    cusolverSpHandle_t cusolver_handle = NULL;
    CUSOLVER_ERROR(cusolverSpCreate(&cusolver_handle));

    // create cusparse handle (used in residual evaluation)
    cusparseHandle_t cusparse_handle = NULL;
    CUSPARSE_ERROR(cusparseCreate(&cusparse_handle));

    // set cusparse and cusolver stream
    CUSOLVER_ERROR(cusolverSpSetStream(cusolver_handle, stream));
    CUSPARSE_ERROR(cusparseSetStream(cusparse_handle, stream));

    // configure matrix descriptor
    cusparseMatDescr_t matQ_desc = NULL;
    CUSPARSE_ERROR(cusparseCreateMatDescr(&matQ_desc));
    CUSPARSE_ERROR(cusparseSetMatType(matQ_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (Q_trans.outerIndexPtr()[0] == 0) {
        // that should be always the case
        CUSPARSE_ERROR(
            cusparseSetMatIndexBase(matQ_desc, CUSPARSE_INDEX_BASE_ZERO));
    } else {
        CUSPARSE_ERROR(
            cusparseSetMatIndexBase(matQ_desc, CUSPARSE_INDEX_BASE_ONE));
    }

    int issym = 0;
    CUSOLVER_ERROR(cusolverSpXcsrissymHost(
        cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(), matQ_desc,
        Q_trans.outerIndexPtr(), Q_trans.outerIndexPtr() + 1,
        Q_trans.innerIndexPtr(), &issym));

    if (!issym) {
        printf("Error: Q is not symmetric...\n");
        exit(EXIT_FAILURE);
    }
    double t_factor = 0;
    printf("%6.2g secs | ", t_factor);


    double *d_Q_val, *d_rhs, *d_U;
    int *   d_Q_rowPtr, *d_Q_ColInd;
    double  tol = 1.0e-8;
    int     singularity;

    CUDA_ERROR(
        cudaMalloc((void**)&d_Q_val, Q_trans.nonZeros() * sizeof(double)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_Q_rowPtr, (Q_trans.rows() + 1) * sizeof(int)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_Q_ColInd, Q_trans.nonZeros() * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_rhs, Q_trans.rows() * sizeof(double)));
    CUDA_ERROR(cudaMalloc((void**)&d_U, Q_trans.rows() * sizeof(double)));

    CUDA_ERROR(cudaMemcpy(d_Q_val, Q_trans.valuePtr(),
                          Q_trans.nonZeros() * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy(d_Q_rowPtr, Q_trans.outerIndexPtr(),
                          (Q_trans.rows() + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy(d_Q_ColInd, Q_trans.innerIndexPtr(),
                          Q_trans.nonZeros() * sizeof(int),
                          cudaMemcpyHostToDevice));
    double total_time_ms = 0;
    for (int d = 0; d < rhs.cols(); ++d) {

        CUDA_ERROR(cudaMemcpy(d_rhs, rhs.col(d).data(),
                              Q_trans.rows() * sizeof(double),
                              cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMemset(d_U, 0, Q_trans.rows() * sizeof(double)));

        CUDATimer timer;
        timer.start(stream);
        CUSOLVER_ERROR(cusolverSpDcsrlsvchol(
            cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(), matQ_desc,
            d_Q_val, d_Q_rowPtr, d_Q_ColInd, d_rhs, tol, 0, d_U, &singularity));
        CUDA_ERROR(cudaStreamSynchronize(stream));
        timer.stop();

        total_time_ms += timer.elapsed_millis();

        CUDA_ERROR(cudaMemcpy(U.col(d).data(), d_U,
                              Q_trans.rows() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    printf("%6.2g secs | ", total_time_ms * 100);

    printf("%6.6g |\n", (rhs - Q * U).array().abs().maxCoeff());

    CUDA_ERROR(cudaFree(d_Q_val));
    CUDA_ERROR(cudaFree(d_Q_rowPtr));
    CUDA_ERROR(cudaFree(d_Q_ColInd));
    CUDA_ERROR(cudaFree(d_rhs));
    CUDA_ERROR(cudaFree(d_U));
}
