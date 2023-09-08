
#include <iostream>
#include <vector_types.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"

//#include "cutil_math.h"
//#include "cutil_math.h"
#include <stdio.h>

#define M_PI 3.1415926
#define width 512
#define height 384
#define smps 1024 //samples

//__device__ executed on the device and callable only from the device
struct Ray
{
    float3 orig;     // ray origin
    float3 dir;  //ray direction
    __device__ Ray(float3 o, float3 d) :orig(o), dir(d){}
};

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//__global__ :executed on the device(GPU), callable only form host(CPU）
// this kernel run in parallel on all the CUDA threads
__global__ void render_kernel(float3 output_d)
{
    // assign a CUDA thread to every pixel(x,y)
    // blockIdx, threadIdx and blockDim are CUDA specific keywords
    // replace rested outer loops in CPU code looping over image rows and image columns
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int i = (height - y - 1) * width + x; //整行*宽度 + x the index of current thread (calculate using threadIdx)

    unsigned int s1 = x; //seed for random number generator
    unsigned int s2 = y;

    //generate ray directed at left corner of the screen
    //calculate the directions for all other rays by adding cx and cy increments in x and y directions
    Ray cam(make_float3(50, 52, 259.6), normalize(make_float3(0, -0.042, -1)));// first hardcoded camera ray(origin, direction) 
    float3 cx = make_float3(width * .5135 / height, 0.0, 0.0);//ray direction offset in x direction.
    float3 cy = normalize(cross(cx, cam.dir) ) * 0.5315f;// ray direction offset in y direction (.5135 is field of view angle)




}

int main()
{
    float3* output_h = new float3[width * height]; //pointer to memory for image on the host(system RAM)
    float3* output_d; //pointer to memory for image on the device(GPU VRAM)

    //allocate memory on the device (GPU VRAM)
    cudaMalloc(&output_d, width * height * sizeof(float3));

    //dim3 is CUDA specific type, block and grid are required to schedule CUDA thread over streaming multiprocessors
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    printf("cuda initialized. \nStart rending...");





    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
