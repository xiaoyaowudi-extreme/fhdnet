#include <cuda.hpp>

namespace fhdnet
{
    cudaDeviceProp cuda_device_property;
    bool is_cuda_initted = false;
    void cuda_init()
    {
        if(is_cuda_initted) return;
        CHECK( cudaGetDeviceProperties(&cuda_device_property, 0) );
        CHECK( cudaSetDevice(0) );
        is_cuda_initted = true;
    }
}