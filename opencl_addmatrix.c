#define KERNEL_FUNC "matrix_add"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main() {
    // Kernel code
    const char* kernelSource =
        "__kernel void matrix_add(__global const float* matrix_1, __global const float* matrix_2, __global float* result, int rows, int cols) {"
        "    int x = get_global_id(0);                                                                                                         "
        "    int y = get_global_id(1);                                                                                                         "
        "    if (x < cols && y < rows) {                                                                                                       "
        "       int index = y * cols + x;                                                                                                      "
        "       result[index] = matrix_1[index] + matrix_2[index];                                                                             "
        "    }                                                                                                                                 "
        "}                                                                                                                                     "
        ;




	// The matrix dimensions
	int rows = 100;
	int columns = 100;
	int matrix_size = rows * columns;

    // Get the OpenCl variables 
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer_matrix_1;
    cl_mem buffer_matrix_2;
    cl_mem buffer_result;
    FILE* program_handle;
    char* program_buffer, * program_log;
    size_t program_size, log_size;

    float* matrix_1 = (float*)malloc(matrix_size * sizeof(float));
    float* matrix_2 = (float*)malloc(matrix_size * sizeof(float));
    float* result = (float*)malloc(matrix_size * sizeof(float));
    if (matrix_1 == NULL || matrix_2 == NULL) {
        printf("ERROR: Allocating memory for matrices failed");
        return 1;
    }
    for (int i = 0; i < matrix_size; i++) {
        matrix_1[i] = 1.0f;
        matrix_2[i] = 2.0f;
    }

    err = clGetPlatformIDs(1, &platform, NULL);
    assert(err == CL_SUCCESS);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(err == CL_SUCCESS);
    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS);


    buffer_matrix_1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * matrix_size, matrix_1, &err);
    assert(err == CL_SUCCESS);
    buffer_matrix_2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * matrix_size, matrix_2, &err);
    assert(err == CL_SUCCESS);
    buffer_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * matrix_size, NULL, &err);
    assert(err == CL_SUCCESS);


    program = clCreateProgramWithSource(ctx, 1,
        (const char**)&kernelSource, NULL, &err);
    if (err < 0) {
        perror("Error creating program");
        exit(1);
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Tää on vissii joku kernelin sisäsen login dumppaus?
    if (err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (err < 0) {
        perror("Couldn't create the kernel");
        exit(1);
    }


    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_matrix_1);
    if (err < 0) {
        perror("Error setting the kernel argument");
        exit(1);
    }
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_matrix_2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &columns);

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err < 0) {
        perror("Error creating the command queue");
        exit(1);
    }

    size_t globalSize[2] = {columns, rows};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);

    err = clEnqueueReadBuffer(queue, buffer_result, CL_TRUE, 0, sizeof(float) * matrix_size, result, 0, NULL, NULL);

    printf("Result matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%.2f ", result[i * columns + j]);
        }
        printf("\n");
    }

    clReleaseMemObject(buffer_matrix_1);
    clReleaseMemObject(buffer_matrix_2);
    clReleaseMemObject(buffer_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    free(matrix_1);
    free(matrix_2);
    free(result);
    return 0;
}

