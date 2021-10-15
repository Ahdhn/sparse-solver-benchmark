#pragma once
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>
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
static inline void cusparseHandleError(cusparseStatus_t status, const char* file, const int line)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("\n%s in %s at line %d\n", cusparseGetErrorString(status), file, line);
        exit(EXIT_FAILURE);
    }
    return;
}
#define CUSPARSE_ERROR(err) (cusparseHandleError(err, __FILE__, __LINE__))
#endif

#ifndef _CUSOLVER_ERROR_
#define _CUSOLVER_ERROR_
static inline void cusolverHandleError(cusolverStatus_t status, const char* file, const int line)
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

        printf("\n%s in %s at line %d\n", cusolverGetErrorString(status), file, line);
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

struct CPUTimer
{
    CPUTimer()
    {
    }
    ~CPUTimer()
    {
    }
    void start()
    {
        m_start = std::chrono::high_resolution_clock::now();
    }
    void stop()
    {
        m_stop = std::chrono::high_resolution_clock::now();
    }
    float elapsed_millis()
    {
        return std::chrono::duration<float, std::milli>(m_stop - m_start).count();
    }

   private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_stop;
};

void cusolver_solver_high_level(const std::string&           name,
                                Eigen::SparseMatrix<double>& Q,
                                const Eigen::MatrixXd&       rhs,
                                Eigen::MatrixXd&             U)
{
    if (Q.IsRowMajor()) {
        printf("Error: Q is row major. No need to transpose it...\n");
        exit(EXIT_FAILURE);
    }
    auto Q_trans = Q.transpose();

    U.resize(rhs.rows(), rhs.cols());

    printf("| %30s | ", name.c_str());

    // create stream
    cudaStream_t stream = NULL;
    CUDA_ERROR(cudaStreamCreate(&stream));

    // create cusolver handle
    cusolverSpHandle_t cusolver_handle = NULL;
    CUSOLVER_ERROR(cusolverSpCreate(&cusolver_handle));

    // set cusolver stream
    CUSOLVER_ERROR(cusolverSpSetStream(cusolver_handle, stream));


    // configure matrix descriptor
    cusparseMatDescr_t matQ_desc = NULL;
    CUSPARSE_ERROR(cusparseCreateMatDescr(&matQ_desc));
    CUSPARSE_ERROR(cusparseSetMatType(matQ_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (Q_trans.outerIndexPtr()[0] == 0) {
        // that should be always the case
        CUSPARSE_ERROR(cusparseSetMatIndexBase(matQ_desc, CUSPARSE_INDEX_BASE_ZERO));
    } else {
        CUSPARSE_ERROR(cusparseSetMatIndexBase(matQ_desc, CUSPARSE_INDEX_BASE_ONE));
    }

    // sanity check to make sure Q is symmetric
    int issym = 0;
    CUSOLVER_ERROR(cusolverSpXcsrissymHost(
        cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(), matQ_desc, Q_trans.outerIndexPtr(),
        Q_trans.outerIndexPtr() + 1, Q_trans.innerIndexPtr(), &issym));

    if (!issym) {
        printf("Error: Q is not symmetric...\n");
        exit(EXIT_FAILURE);
    }
    double t_factor = 0;
    printf("%6.2g secs | ", t_factor);

    // 0 = no reorder
    // 1 = symrcm
    // 2= symamd
    // 3 =  csrmetisnd
    int reorder = 3;

    // Allocate and move CSR matrix to the GPU
    double *d_Q_val, *d_rhs, *d_U;
    int *   d_Q_rowPtr, *d_Q_ColInd;
    double  tol = 1.0e-8;
    int     singularity;

    CUDA_ERROR(cudaMalloc((void**)&d_Q_val, Q_trans.nonZeros() * sizeof(double)));
    CUDA_ERROR(cudaMalloc((void**)&d_Q_rowPtr, (Q_trans.rows() + 1) * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_Q_ColInd, Q_trans.nonZeros() * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_rhs, Q_trans.rows() * sizeof(double)));
    CUDA_ERROR(cudaMalloc((void**)&d_U, Q_trans.rows() * sizeof(double)));

    CUDA_ERROR(cudaMemcpy(d_Q_val, Q_trans.valuePtr(), Q_trans.nonZeros() * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy(d_Q_rowPtr, Q_trans.outerIndexPtr(), (Q_trans.rows() + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy(d_Q_ColInd, Q_trans.innerIndexPtr(), Q_trans.nonZeros() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Solve -- treating each column in rhs as a different rhs
    double total_time_ms = 0;
    for (int d = 0; d < rhs.cols(); ++d) {

        // Copy rhs to the device
        CUDA_ERROR(cudaMemcpy(d_rhs, rhs.col(d).data(), Q_trans.rows() * sizeof(double),
                              cudaMemcpyHostToDevice));

        // init the solution to zero
        CUDA_ERROR(cudaMemset(d_U, 0, Q_trans.rows() * sizeof(double)));

        CUDATimer timer;
        timer.start(stream);

        // Solve using cholesky factorization
        CUSOLVER_ERROR(cusolverSpDcsrlsvchol(cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(),
                                             matQ_desc, d_Q_val, d_Q_rowPtr, d_Q_ColInd, d_rhs, tol,
                                             reorder, d_U, &singularity));
        timer.stop();
        CUDA_ERROR(cudaStreamSynchronize(stream));


        if (0 <= singularity) {
            printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
        }

        total_time_ms += timer.elapsed_millis();

        // copy solution back to the host
        CUDA_ERROR(cudaMemcpy(U.col(d).data(), d_U, Q_trans.rows() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    printf("%6.2g secs | ", (total_time_ms * 0.001));

    printf("%6.6g |\n", (rhs - Q * U).array().abs().maxCoeff());

    // free device memory
    CUDA_ERROR(cudaFree(d_Q_val));
    CUDA_ERROR(cudaFree(d_Q_rowPtr));
    CUDA_ERROR(cudaFree(d_Q_ColInd));
    CUDA_ERROR(cudaFree(d_rhs));
    CUDA_ERROR(cudaFree(d_U));

    CUSOLVER_ERROR(cusolverSpDestroy(cusolver_handle));
    CUDA_ERROR(cudaStreamDestroy(stream));
    CUSPARSE_ERROR(cusparseDestroyMatDescr(matQ_desc));
}


void cusolver_solver_low_level(const std::string&           name,
                               Eigen::SparseMatrix<double>& Q,
                               const Eigen::MatrixXd&       rhs,
                               Eigen::MatrixXd&             U)
{
    /*
     * solves Q*U = rhs
     * step 1: B = Q(A,A) = A*Q*A'
     *   A is the ordering to minimize zero fill-in
     * step 2: solve B*z = A*rhs for z
     * step 3: U = inv(A)*z
     * The three steps are
     *  (A*Q*A')*(A*U) = (A*rhs)
     */
    if (Q.IsRowMajor()) {
        printf("Error: Q is row major. No need to transpose it...\n");
        exit(EXIT_FAILURE);
    }
    auto Q_trans = Q.transpose();

    U.resize(rhs.rows(), rhs.cols());

    printf("| %30s | ", name.c_str());

    // create stream
    cudaStream_t stream = NULL;
    CUDA_ERROR(cudaStreamCreate(&stream));

    // create cusparse handle
    cusparseHandle_t cusparse_handle = NULL;
    CUSPARSE_ERROR(cusparseCreate(&cusparse_handle));

    // create cusolver handle
    cusolverSpHandle_t cusolver_handle = NULL;
    CUSOLVER_ERROR(cusolverSpCreate(&cusolver_handle));

    // set cusolver and cusparse stream
    CUSOLVER_ERROR(cusolverSpSetStream(cusolver_handle, stream));
    CUSPARSE_ERROR(cusparseSetStream(cusparse_handle, stream));

    // configure matrix descriptor
    cusparseMatDescr_t matQ_desc = NULL;
    CUSPARSE_ERROR(cusparseCreateMatDescr(&matQ_desc));
    CUSPARSE_ERROR(cusparseSetMatType(matQ_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (Q_trans.outerIndexPtr()[0] == 0) {
        // that should be always the case
        CUSPARSE_ERROR(cusparseSetMatIndexBase(matQ_desc, CUSPARSE_INDEX_BASE_ZERO));
    } else {
        CUSPARSE_ERROR(cusparseSetMatIndexBase(matQ_desc, CUSPARSE_INDEX_BASE_ONE));
    }

    // sanity check to make sure Q is symmetric
    int issym = 0;
    CUSOLVER_ERROR(cusolverSpXcsrissymHost(
        cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(), matQ_desc, Q_trans.outerIndexPtr(),
        Q_trans.outerIndexPtr() + 1, Q_trans.innerIndexPtr(), &issym));

    if (!issym) {
        printf("Error: Q is not symmetric...\n");
        exit(EXIT_FAILURE);
    }


    // A*rhs
    double* h_Arhs = (double*)malloc(sizeof(double) * Q_trans.rows());
    double* d_Arhs = NULL;
    CUDA_ERROR(cudaMalloc((void**)&d_Arhs, sizeof(double) * Q.rows()));

    // reorder mat to reduce zero fill-in
    int* h_A = (int*)malloc(sizeof(int) * Q.rows());
    int* d_A = NULL;
    CUDA_ERROR(cudaMalloc((void**)&d_A, sizeof(int) * Q.rows()));

    // B: used in B*z = A*rhs
    int*    h_B_rowPtr = (int*)malloc(sizeof(int) * (Q_trans.rows() + 1));
    int*    h_B_colInd = (int*)malloc(sizeof(int) * Q_trans.nonZeros());
    double* h_B_val = (double*)malloc(sizeof(double) * Q_trans.nonZeros());
    int *   d_B_rowPtr, *d_B_colInd;
    double* d_B_val;
    CUDA_ERROR(cudaMalloc((void**)&d_B_rowPtr, sizeof(int) * (Q_trans.rows() + 1)));
    CUDA_ERROR(cudaMalloc((void**)&d_B_colInd, sizeof(int) * Q_trans.nonZeros()));
    CUDA_ERROR(cudaMalloc((void**)&d_B_val, sizeof(double) * Q_trans.nonZeros()));


    // working space for permutation: B = A*Q*A^T
    size_t size_perm = 0;
    void*  perm_buffer_cpu = NULL;

    // Map from Q values to B
    int* h_mapBfromQ = (int*)malloc(sizeof(int) * Q_trans.nonZeros());
    for (int j = 0; j < Q_trans.nonZeros(); j++) {
        h_mapBfromQ[j] = j;
    }

    // Allocate and move Q, and U to the device
    double *d_Q_val, *d_U, *d_z;  // z = B \ A*rhs
    int *   d_Q_rowPtr, *d_Q_ColInd;
    CUDA_ERROR(cudaMalloc((void**)&d_Q_val, Q_trans.nonZeros() * sizeof(double)));
    CUDA_ERROR(cudaMalloc((void**)&d_Q_rowPtr, (Q_trans.rows() + 1) * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_Q_ColInd, Q_trans.nonZeros() * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_U, Q_trans.rows() * sizeof(double)));
    CUDA_ERROR(cudaMalloc((void**)&d_z, Q_trans.rows() * sizeof(double)));

    CUDA_ERROR(cudaMemcpyAsync(d_Q_val, Q_trans.valuePtr(), Q_trans.nonZeros() * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(d_Q_rowPtr, Q_trans.outerIndexPtr(),
                               (Q_trans.rows() + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(d_Q_ColInd, Q_trans.innerIndexPtr(),
                               Q_trans.nonZeros() * sizeof(int), cudaMemcpyHostToDevice, stream));


    // Step 1: reorder the matrix Q to minimize zero fill-in
    //
    // 0 = no reorder -> Q = 0:n-1
    // 1 = symrcm ->  A = symrcm(Q)
    // 2 = symamd ->  A = symamd(Q)
    // 3 =  metisnd ->  A = metis(Q)
    double factor_time_ms = 0;

    CPUTimer cpu_timer;
    cpu_timer.start();
    int reorder = 3;
    if (reorder == 1) {
        CUSOLVER_ERROR(cusolverSpXcsrsymrcmHost(cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(),
                                                matQ_desc, Q_trans.outerIndexPtr(),
                                                Q_trans.innerIndexPtr(), h_A));
    } else if (reorder == 2) {
        CUSOLVER_ERROR(cusolverSpXcsrsymamdHost(cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(),
                                                matQ_desc, Q_trans.outerIndexPtr(),
                                                Q_trans.innerIndexPtr(), h_A));
    } else if (reorder == 3) {
        CUSOLVER_ERROR(cusolverSpXcsrmetisndHost(
            cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(), matQ_desc, Q_trans.outerIndexPtr(),
            Q_trans.innerIndexPtr(), NULL, h_A));
    } else {
#pragma omp parallel for
        for (int j = 0; j < Q_trans.rows(); ++j) {
            h_A[j] = j;
        }
    }

// perm mat B -- we have to copy Q into B since permutation happens in-place
#pragma omp parallel for
    for (int j = 0; j < Q_trans.rows() + 1; ++j) {
        h_B_rowPtr[j] = Q_trans.outerIndexPtr()[j];
    }
#pragma omp parallel for
    for (int j = 0; j < Q_trans.nonZeros(); ++j) {
        h_B_colInd[j] = Q_trans.innerIndexPtr()[j];
    }
    cpu_timer.stop();
    factor_time_ms += cpu_timer.elapsed_millis();

    // get the size of cpu buffer needed for permutation
    CUSOLVER_ERROR(cusolverSpXcsrperm_bufferSizeHost(cusolver_handle, Q_trans.rows(),
                                                     Q_trans.cols(), Q_trans.nonZeros(), matQ_desc,
                                                     h_B_rowPtr, h_B_colInd, h_A, h_A, &size_perm));

    perm_buffer_cpu = (void*)malloc(sizeof(char) * size_perm);
    assert(NULL != perm_buffer_cpu);

    cpu_timer.start();
    // do the permutation which works only on the col and row indices
    CUSOLVER_ERROR(cusolverSpXcsrpermHost(cusolver_handle, Q_trans.rows(), Q_trans.cols(),
                                          Q_trans.nonZeros(), matQ_desc, h_B_rowPtr, h_B_colInd,
                                          h_A, h_A, h_mapBfromQ, perm_buffer_cpu));

// update permutation matrix value by using the mapping from Q such that
// B = Q( mapBfromQ )
#pragma omp parallel for
    for (int j = 0; j < Q_trans.nonZeros(); j++) {
        h_B_val[j] = Q_trans.valuePtr()[h_mapBfromQ[j]];
    }
    cpu_timer.stop();
    factor_time_ms += cpu_timer.elapsed_millis();

    printf("%6.2g secs | ", (factor_time_ms * 0.001));

    // Move B, A*rhs, and A to device
    CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, sizeof(int) * Q.rows(), cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(d_B_rowPtr, h_B_rowPtr, sizeof(int) * (Q_trans.rows() + 1),
                               cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(d_B_colInd, h_B_colInd, sizeof(int) * Q_trans.nonZeros(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(d_B_val, h_B_val, sizeof(double) * Q_trans.nonZeros(),
                               cudaMemcpyHostToDevice, stream));

    CUDA_ERROR(cudaDeviceSynchronize());


    double tol = 1.0e-8;
    int    singularity;

    // Solve -- treating each column in rhs as a different rhs
    double solve_time_ms = 0;
    for (int col = 0; col < rhs.cols(); ++col) {

        // h_Arhs = rhs(A)
#pragma omp parallel for
        for (int row = 0; row < Q_trans.rows(); row++) {
            assert(h_A[row] < Q_trans.cols());
            h_Arhs[row] = rhs.col(col)[h_A[row]];
        }
        CUDA_ERROR(cudaMemcpyAsync(d_Arhs, h_Arhs, sizeof(double) * Q.rows(),
                                   cudaMemcpyHostToDevice, stream));

        // init the solution to zero
        CUDA_ERROR(cudaMemset(d_z, 0, Q_trans.rows() * sizeof(double)));

        CUDATimer timer;
        timer.start(stream);

        // solve B*z = A*rhs
        CUSOLVER_ERROR(cusolverSpDcsrlsvchol(cusolver_handle, Q_trans.rows(), Q_trans.nonZeros(),
                                             matQ_desc, d_B_val, d_B_rowPtr, d_B_colInd, d_Arhs,
                                             tol, 0, d_z, &singularity));

        // solve A*u = z
        CUSPARSE_ERROR(cusparseDsctr(cusparse_handle, Q_trans.rows(), d_z, d_A, d_U,
                                     CUSPARSE_INDEX_BASE_ZERO));

        timer.stop();
        CUDA_ERROR(cudaStreamSynchronize(stream));

        solve_time_ms += timer.elapsed_millis();

        if (0 <= singularity) {
            printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
        }

        // copy solution back to the host
        CUDA_ERROR(cudaMemcpy(U.col(col).data(), d_U, Q_trans.rows() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    printf("%6.2g secs | ", (solve_time_ms * 0.001));

    printf("%6.6g |\n", (rhs - Q * U).array().abs().maxCoeff());

    // free device memory
    CUDA_ERROR(cudaFree(d_Q_val));
    CUDA_ERROR(cudaFree(d_Q_rowPtr));
    CUDA_ERROR(cudaFree(d_Q_ColInd));
    CUDA_ERROR(cudaFree(d_B_rowPtr));
    CUDA_ERROR(cudaFree(d_B_colInd));
    CUDA_ERROR(cudaFree(d_B_val));
    CUDA_ERROR(cudaFree(d_U));
    CUDA_ERROR(cudaFree(d_Arhs));
    CUDA_ERROR(cudaFree(d_A));
    CUDA_ERROR(cudaFree(d_z));
    free(h_Arhs);
    free(h_A);
    free(h_B_rowPtr);
    free(h_B_colInd);
    free(h_B_val);
    free(h_mapBfromQ);
    free(perm_buffer_cpu);

    CUSOLVER_ERROR(cusolverSpDestroy(cusolver_handle));
    CUSPARSE_ERROR(cusparseDestroy(cusparse_handle));
    CUDA_ERROR(cudaStreamDestroy(stream));
    CUSPARSE_ERROR(cusparseDestroyMatDescr(matQ_desc));
}
