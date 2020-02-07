# cuda_mt19937
> Author: Ziyao Xiao

> Date: 2/7/2020

> File: cuda_mt19937.hpp
#
### ___Discription___:

This file allows programms use mt19937 random int algorithm on gpu.

### ___Classes___:

> cuda_mt19937

> __Functions__:


>> ```cpp
>> cuda_mt19937::cuda_mt19937(uint32_t __seed = 0);
>> ```

>> &nbsp;

>> _Discription_:

>> It's the initialize function of the class.

>> ___It can't be called from gpu.___

>> &nbsp;

>> _Parameters_:

>> + \_\_seed The random seed.

> &nbsp;

>> ```cpp
>> __device__ uint32_t cuda_mt19937::operator()();
>> ```

>> &nbsp;

>> _Discription_:

>> It generates a unsigned 32 bits integer.

>> ___It can only be used on gpu.___

>> &nbsp;

>> _Parameters_:

>> + None

> &nbsp;

>> ```cpp
>> cuda_mt19937::~cuda_mt19937();
>> ```

>> &nbsp;

>> _Discription_:

>> It's the destructor of the function and it frees the memory alloced on gpu.

>> ___It can't be called from gpu.___

>> &nbsp;

>> _Parameters_:

>> + None

> &nbsp;

>> ```cpp
>> cuda_mt19937* cuda_mt19937::create_device_ptr();
>> ```

> &nbsp;

>> _Discription_:

>> It creates the class ptr so that the program can use it on gpu.

>> ___The data between the class on gpu ans the class on cpu are not synced.___

>> &nbsp;

>> _Parameters_:

>> + None

> &nbsp;

>> ```cpp
>> void cuda_mt19937::free_device_ptr(cuda_mt19937 *ptr);
>> ```

>> &nbsp;

>> _Discription_:

>> It free the device ptr which is create by create_device_ptr.

>> ___It can't be called from gpu.___

>> &nbsp;

>> _Parameters_:

>> + ptr The class ptr on gpu.

> __Variables__

>> __None__