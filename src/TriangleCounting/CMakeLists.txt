find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER g++)
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

list(APPEND CUDA_NVCC_FLAGS
    -gencode arch=compute_70,code=sm_70
	#-O0 -std=c++14 -g -lineinfo)
	-O3 -std=c++14)

add_subdirectory(DegreeBased)
add_subdirectory(LowerTriangular)