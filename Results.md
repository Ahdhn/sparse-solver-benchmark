# **Big Matrix** ([link to input](https://ucdavis365-my.sharepoint.com/:u:/g/personal/ahmahmoud_ucdavis_edu/ESAbEmxoe4RJqrl__bdsfX8BraXH4Bh9E1Ptr7Pz9conPg?e=2AFBkd))
## *RTX A6000* 

> # Harmonic
>
> matrix rows= 1009118, cols= 1009118, nnz= 7063814
>
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |    9.4 secs |   0.15 secs | 1.66622e-12 |
> |                cusolver (High) |      0 secs |     36 secs | 1.66622e-12 |
> |    Eigen::CholmodSupernodalLLT |      5 secs |   0.59 secs | 1.81899e-12 |
> |           Eigen::SimplicialLLT |     14 secs |   0.49 secs | 6.13909e-12 |
> |          Eigen::SimplicialLDLT |     14 secs |   0.48 secs | 5.68434e-12 |
> |            catamari::SparseLDL |     22 secs |   0.62 secs | 5.00222e-12 |
> |              Eigen::PardisoLLT |    6.6 secs |    3.4 secs | 7.35412e-13 |
> |                Eigen::SparseLU |     52 secs |    1.2 secs | 3.52429e-12 |

> # Biharmonic
> matrix rows= 1009118, cols= 1009118, nnz= 20693752
> 
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |     20 secs |   0.34 secs | 2.17649e-08 |
> |                cusolver (High) |      0 secs |    ALLOC_FAILED
> |    Eigen::CholmodSupernodalLLT |     13 secs |      1 secs | 4.67912e-09 |
> |           Eigen::SimplicialLLT | 1.7e+02 secs |    1.9 secs | 6.61521e-09 |
> |          Eigen::SimplicialLDLT | 1.6e+02 secs |    1.8 secs | 1.11711e-08 |
> |            catamari::SparseLDL | 7.1e+02 secs |    2.9 secs | 6.76262e-09 |
> |              Eigen::PardisoLLT |     13 secs |      3 secs | 1.36159e-08 |
> |                Eigen::SparseLU | 4.1e+02 secs |    3.7 secs | 7.33891e-09 |

> # Triharmonic
> matrix rows= 1009118, cols= 1009118, nnz= 42921650
> 
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |     45 secs |   0.59 secs | 0.000220632 |
> |                cusolver (High) |      0 secs |    ALLOC_FAILED
> |    Eigen::CholmodSupernodalLLT |     25 secs |    1.5 secs | 0.000287713 |
> |           Eigen::SimplicialLLT | 5.7e+02 secs |    4.1 secs | 0.000467 |
> |          Eigen::SimplicialLDLT | 5.8e+02 secs |    3.8 secs | 0.000220108 |
> |            catamari::SparseLDL | 3.1e+03 secs |    7.2 secs | 0.00034493 |
> |              Eigen::PardisoLLT |     26 secs |    4.4 secs | 0.000222996 |
> |                Eigen::SparseLU | 1.5e+03 secs |    7.9 secs | 0.00228721 |


## *A100*

> # Harmonic
>
> matrix rows= 1009118, cols= 1009118, nnz= 7063814
>
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |     10 secs |   0.16 secs | 2.04636e-12 |
> |                cusolver (High) |      0 secs | ALLOC_FAILED |

> # Biharmonic
> matrix rows= 1009118, cols= 1009118, nnz= 20693752
> 
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |     23 secs |   0.49 secs | 1.93812e-08 |
> |                cusolver (High) |      0 secs | ALLOC_FAILED

> # Triharmonic
> matrix rows= 1009118, cols= 1009118, nnz= 42921650
> 
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |     49 secs |    0.8 secs | 0.000243265 |
> |                cusolver (High) |      0 secs | ALLOC_FAILED





# **Medium  Matrix** ([link to input](https://github.com/Ahdhn/sparse-solver-benchmark/blob/main/xyzrgb_dragon-720K.ply))
## *RTX A6000* 

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
>
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




## *A100*

> # Harmonic
> 
> matrix rows= 360757, cols= 360757, nnz= 2525287
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |    3.6 secs |  0.067 secs | 5.50583e-11 |
> |                cusolver (High) |      0 secs |     13 secs | 5.50583e-11 |

> # Biharmonic
> 
> matrix rows= 360757, cols= 360757, nnz= 7356259
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |    7.4 secs |   0.18 secs | 5.58785e-05 |
> |                cusolver (High) |      0 secs |     28 secs | 5.58785e-05 |

> # Triharmonic
> matrix rows= 360757, cols= 360757, nnz= 15169333
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |     14 secs |   0.32 secs | 9.55742 |
> |                cusolver (High) |      0 secs | ALLOC_FAILED |

# **Small Matrix** ([link to input](https://ucdavis365-my.sharepoint.com/:u:/g/personal/ahmahmoud_ucdavis_edu/ETjTF3jomO5OkvPECOE07J0BoDHHw2XWBRh-9SAXaPhG9w?e=xfWLUi))
## *RTX A6000* 

> # Harmonic
>
> matrix rows= 29921, cols= 29921, nnz= 209435
>
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |    0.2 secs |  0.018 secs | 4.3928e-15 |
> |                cusolver (High) |      0 secs |   0.88 secs | 4.3928e-15 |
> |    Eigen::CholmodSupernodalLLT |   0.17 secs |  0.021 secs | 1.88096e-15 |
> |           Eigen::SimplicialLLT |   0.11 secs |  0.012 secs | 6.70192e-15 |
> |          Eigen::SimplicialLDLT |  0.089 secs |  0.009 secs | 5.89695e-15 |
> |            catamari::SparseLDL |  0.094 secs |   0.01 secs | 3.97726e-15 |
> |              Eigen::PardisoLLT |   0.27 secs |   0.14 secs | 1.62757e-15 |
> |                Eigen::SparseLU |   0.32 secs |  0.021 secs | 2.98143e-15 |

> # Biharmonic
>
> matrix rows= 29921, cols= 29921, nnz= 595397
>
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |   0.36 secs |  0.043 secs | 2.54958e-09 |
> |                cusolver (High) |      0 secs |    1.7 secs | 2.54958e-09 |
> |    Eigen::CholmodSupernodalLLT |   0.29 secs |  0.027 secs | 2.55985e-09 |
> |           Eigen::SimplicialLLT |   0.51 secs |  0.034 secs | 2.55985e-09 |
> |          Eigen::SimplicialLDLT |    0.4 secs |  0.038 secs | 1.59547e-09 |
> |            catamari::SparseLDL |   0.59 secs |  0.039 secs | 2.55985e-09 |
> |              Eigen::PardisoLLT |   0.29 secs |   0.12 secs | 1.64138e-09 |
> |                Eigen::SparseLU |    1.5 secs |  0.065 secs | 1.17246e-09 |

> # Triharmonic
>
> matrix rows= 29921, cols= 29921, nnz= 1210911
>
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |   0.63 secs |  0.075 secs | 0.00630138 |
> |                cusolver (High) |      0 secs |      3 secs | 0.00630138 |
> |    Eigen::CholmodSupernodalLLT |   0.49 secs |  0.037 secs | 0.00510706 |
> |           Eigen::SimplicialLLT |    1.3 secs |  0.059 secs | 0.00314806 |
> |          Eigen::SimplicialLDLT |    1.2 secs |  0.063 secs | 0.00624473 |
> |            catamari::SparseLDL |    2.2 secs |  0.075 secs | 0.00383347 |
> |              Eigen::PardisoLLT |   0.43 secs |   0.14 secs | 0.00507063 |
> |                Eigen::SparseLU |    7.2 secs |   0.15 secs | 0.00426015 |
> -----------------------------------------------------------------------------


## *A100*

> # Harmonic
>
> matrix rows= 29921, cols= 29921, nnz= 209435
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |   0.28 secs |  0.022 secs | 2.48129e-15 |
> |                cusolver (High) |      0 secs |   0.86 secs | 2.48129e-15 |

> # Biharmonic
>
> matrix rows= 29921, cols= 29921, nnz= 595397
>
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |   0.44 secs |  0.058 secs | 3.02551e-09 |
> |                cusolver (High) |      0 secs |    1.7 secs | 3.02551e-09 |

> # Triharmonic
>
> matrix rows= 29921, cols= 29921, nnz= 1210911
>
> |                         Method |      Factor |       Solve |  L_inf norm |
> |-------------------------------:|------------:|------------:|------------:|
> |     cusolver (Preview/reorder) |   0.79 secs |   0.09 secs | 0.00468188 |
> |                cusolver (High) |      0 secs |    3.1 secs | 0.00468188 |


