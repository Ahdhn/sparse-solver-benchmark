#pragma once 
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string>
#include <Eigen/Core>
#include <Eigen/Sparse>

template <int T>
void cusolver_solver(
	const std::string& name,
	const Eigen::SparseMatrix<double>& Q,
	const Eigen::MatrixXd& rhs,
	Eigen::MatrixXd& U) {

}




