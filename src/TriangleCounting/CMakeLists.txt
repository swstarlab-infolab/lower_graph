find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER g++)
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

list(APPEND CUDA_NVCC_FLAGS
	--expt-relaxed-constexpr
    -gencode arch=compute_70,code=sm_70
	#-gencode arch=compute_52,code=sm_52
	#-O0 -std=c++14 -g -lineinfo)
	-O3 -std=c++14)

add_subdirectory(LowerTriangular)
