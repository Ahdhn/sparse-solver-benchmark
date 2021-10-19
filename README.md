# 🚀 Sparse Positive Definite Linear System Solver Benchmark 📏

## Ahmed's notes on cuSOLVER: ##
Please read the rest of the README to learn more about the problem setup and how to run the code!

In order to test GPU performance, we have added [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html). cuSOLVER provides two different APIs for Cholesky factorization — a high-level and low-level API. The high-level API provides an easy to use API for linear solver on the GPU. It reorders and factors the input matrix and allocates the required memory under the hood. So, it is _not_ possible to separate the factorization time from the solve time. The low-level API provides finer grain control over the different operations. While the factorization and solve operations take place on the GPU, there is only a host API for the reordering of the matrix. In the table below, `cusolver (Preview/reorder)` is cuSOLVER implementation using `metisnd` reorder (which gave the best performance) and Cholesky factorization from `cusolverSp_LOWLEVEL_PREVIEW.h`. `cusolver (High)` uses the high-level API and thus Factor time is zero. Finally, GPU time does _not_ include the time it takes to move the data back and forth between host and device, allocating memory on the device, or Cholesky analysis part.

The results below are from a Intel Xeon Gold 5218 CPU @ 2.30GHz (2 processors) and NVIDIA RTX A6000 GPU. 

> # harmonic
>
> matrix rows= 360757, cols= 360757, nnz= 2525287
>
> |                         Method |      Factor |       Solve |     L∞ norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |    3.4 secs |   0.06 secs | 1.05662e-10 |
> |                cusolver (High) |      0 secs |     13 secs | 1.05662e-10 |
> |    Eigen::CholmodSupernodalLLT |    1.8 secs |   0.24 secs | 1.13266e-10 |
> |           Eigen::SimplicialLLT |    1.8 secs |   0.15 secs | 1.19565e-10 |
> |          Eigen::SimplicialLDLT |    1.8 secs |   0.14 secs | 1.48007e-10 |
> |            catamari::SparseLDL |    2.4 secs |   0.16 secs | 1.27168e-10 |
> |              Eigen::PardisoLLT |    2.4 secs |    1.4 secs | 8.80447e-11 |
> |                Eigen::SparseLU |    8.5 secs |   0.36 secs | 6.85971e-11 |
> | Eigen::BiCGSTAB<IncompleteLUT> |    2.1 secs |      2 secs | 6.89608e-11 |
> |       Eigen::CG<IncompleteLUT> |    2.1 secs |    4.8 secs | 6.89608e-11 |
 
> # biharmonic
> matrix rows= 360757, cols= 360757, nnz= 7356259
>
> |                         Method |      Factor |       Solve |     L∞ norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |      7 secs |   0.14 secs | 0.000116914 |
> |                cusolver (High) |      0 secs |     29 secs | 0.000116914 |
> |    Eigen::CholmodSupernodalLLT |    4.3 secs |   0.33 secs | 0.000174864 |
> |           Eigen::SimplicialLLT |     13 secs |   0.49 secs | 7.11373e-05 |
> |          Eigen::SimplicialLDLT |     13 secs |   0.49 secs | 2.53609e-05 |
> |            catamari::SparseLDL |     21 secs |   0.62 secs | 0.000144347 |
> |              Eigen::PardisoLLT |    4.6 secs |    1.4 secs | 6.61918e-05 |
> |                Eigen::SparseLU |     58 secs |      1 secs | 2.22766e-05 |
> | Eigen::BiCGSTAB<IncompleteLUT> |     15 secs |    6.3 secs | 5.58785e-05 |
> |       Eigen::CG<IncompleteLUT> |     15 secs |    9.4 secs | 8.63961e-05 |

> # triharmonic
>
>matrix rows= 360757, cols= 360757, nnz= 15169333
>
>|                         Method |      Factor |       Solve |     L∞ norm |
>|-------------------------------:|------------:|------------:|------------:|
>|     cusolver (Preview/reorder) |     14 secs |   0.27 secs | 63.6828     |
>|                cusolver (High) |      0 secs |     58 secs | 63.6828     |
>|    Eigen::CholmodSupernodalLLT |    7.7 secs |   0.46 secs | 39.3699     |
>|           Eigen::SimplicialLLT |     48 secs |      1 secs | 16.7099     |
>|          Eigen::SimplicialLDLT |     49 secs |      1 secs | 24.1019     |
>|            catamari::SparseLDL |     90 secs |    1.5 secs | 102.97      |
>|              Eigen::PardisoLLT |    8.1 secs |    1.8 secs | 42.375      |
>|                Eigen::SparseLU | 2.6e+02 secs|    2.3 secs | 26.3567     |
-----------------------------------------------------------------------------


This is an informal benchmark for _k_-harmonic diffusion problems on triangle
meshes found commonly in geometry processing.

> Current upshot: 
> 
> - CholMod is **very good** 🐐
> - Pardiso **holds up alright** 🏆
> - Eigen LLT/LDLT are **OK** but suffer for less sparse systems 🏅
> - catamari is **OK** for medium but inaccurate for big systems (bonus: it's MPL2 🆓)
> - Eigen LU¹ is significantly **slower** 🐌
>
> ¹These systems are
> [SPD](https://en.wikipedia.org/wiki/Definite_symmetric_matrix) so LU is not a
> good choice, but provides a reference.

## Clone

    git clone --recursive https://github.com/alecjacobson/sparse-solver-benchmark

## Build

    mkdir build
    cd build
    cmake ../ -DCMAKE_BUILD_TYPE=Release
    make

## Run

    ./sparse_solver_benchmark [path to triangle mesh]

## Example

Running

    ./sparse_solver_benchmark ../xyzrgb_dragon-720K.ply 

on my MacBook 2.3 GHz Quad-Core Intel Core i7 with 32 GB Ram will produce:

> # harmonic
> 
> |                         Method |      Factor |       Solve |     L∞ norm |
> |-------------------------------:|------------:|------------:|------------:|
> |    Eigen::CholmodSupernodalLLT |   0.86 secs |    0.1 secs | 1.13266e-10 |
> |           Eigen::SimplicialLLT |    1.5 secs |   0.12 secs | 1.19565e-10 |
> |          Eigen::SimplicialLDLT |    1.5 secs |   0.14 secs | 1.48007e-10 |
> |            catamari::SparseLDL |    1.7 secs |   0.11 secs | 1.27168e-10 |
> |              Eigen::PardisoLLT |    1.9 secs |   0.59 secs | 1.05662e-10 |
> |                Eigen::SparseLU |    4.8 secs |   0.18 secs | 6.85971e-11 |
> 
> # biharmonic
> 
> |                         Method |      Factor |       Solve |     L∞ norm |
> |-------------------------------:|------------:|------------:|------------:|
> |    Eigen::CholmodSupernodalLLT |    1.7 secs |   0.13 secs | 8.33117e-05 |
> |           Eigen::SimplicialLLT |     12 secs |   0.37 secs | 5.58785e-05 |
> |          Eigen::SimplicialLDLT |     12 secs |   0.41 secs | 6.92762e-05 |
> |            catamari::SparseLDL |     13 secs |   0.33 secs | 0.0002359 |
> |              Eigen::PardisoLLT |      4 secs |   0.69 secs | 6.34374e-05 |
> |                Eigen::SparseLU |     31 secs |    0.6 secs | 4.10639e-05 |
> 
> # triharmonic
> 
> |                         Method |      Factor |       Solve |     L∞ norm |
> |-------------------------------:|------------:|------------:|------------:|
> |    Eigen::CholmodSupernodalLLT |    3.3 secs |   0.19 secs | 42.2496 |
> |           Eigen::SimplicialLLT |     41 secs |    0.8 secs | 36.0525 |
> |          Eigen::SimplicialLDLT |     41 secs |   0.89 secs | 32.1019 |
> |            catamari::SparseLDL |     50 secs |   0.78 secs | 150.97 |
> |              Eigen::PardisoLLT |    6.9 secs |   0.81 secs | 96.7205 |
> |                Eigen::SparseLU | 1.3e+02 secs |    1.2 secs | 22.0579 |

Obviously [YMMV](https://www.google.com/search?q=YMMV), if you find something
interesting [let me know!](https://github.com/alecjacobson/sparse-solver-benchmark/issues).

## What about this other solver XYZ?

Please [submit a pull
request](https://github.com/alecjacobson/sparse-solver-benchmark/pulls) with a
wrapper for solver XYZ. The more the merrier. 


## What are the systems being solved?

This code will build a discretization of the ∆ᵏ operator and solve a system of
the form:

    ∆ᵏ u + u = x

where x is the surface's embedding. In matrix form this is:


    (Wᵏ + M) u = M x

where Wᵏ is defined recursively as:

    W¹ = L
    Wᵏ⁺¹ = Wᵏ M⁻¹ L

and `L` is the discrete Laplacian and `M` is the discrete mass matrix.

This is a form of smoothing (k=1 is implicit mean curvature flow "Implicit
Fairing of Irregular Meshes" Desbrun et al. 1999, k≥2 is higher order, e.g., "Mixed
Finite Elements for Variational Surface Modeling" Jacobson et al. 2010)

For k=1, the system is generally OK w.r.t. conditioning and the sparsity for a
minifold mesh will be 7 non-zeros per row (on average).

For k=3, the system can get really badly scaled and starts to become more dense
(~40 non-zeros per row).
