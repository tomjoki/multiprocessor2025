#define RESIZE_KERNEL_FUNC "resizeImage"
#define GRAYSCALE_KERNEL_FUNC "convertToGrayscale"
#define FILTER_KERNEL_FUNC "gaussianBlurFilter"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <windows.h>
#include <CL/opencl.h>
#include "lodepng.h"

double get_time();

int main() {

    const char* resizeKernel = 
        "__kernel void resizeImage(__global const uchar4* image, __global uchar4* result_image, int inwidth, int inheight, int outwidth, int outheight) {"
        "    int x = get_global_id(0);                                                                                                                   "
        "    int y = get_global_id(1);                                                                                                                   "
        "    if (x >= outwidth || y >= outheight) return;                                                                                                "
        "    int in_x = x * 4;                                                                                                                           "
        "    int in_y = y * 4;                                                                                                                           "
        "    uint4 sum = (uint4)(0, 0, 0, 0);                                                                                                            "
        "    for (int dy = 0; dy < 4; dy++) {                                                                                                            "
        "       for (int dx = 0; dx < 4; dx++) {                                                                                                         "
        "           uchar4 pixel = image[(in_y + dy) * inwidth + (in_x + dx)];                                                                           "
        "           sum.x += pixel.x;                                                                                                                    "
        "           sum.y += pixel.y;                                                                                                                    "
        "           sum.z += pixel.z;                                                                                                                    "
        "           sum.w += pixel.w;                                                                                                                    "
        "       }                                                                                                                                        "
        "    }                                                                                                                                           "
        "    uchar4 result;                                                                                                                              "
        "    result.x = sum.x / 16;                                                                                                                      "
        "    result.y = sum.y / 16;                                                                                                                      "
        "    result.z = sum.z / 16;                                                                                                                      "
        "    result.w = sum.w / 16;                                                                                                                      "
        "    result_image[y * outwidth + x] = result;                                                                                                    "
        "}                                                                                                                                               "
        ;

    const char* grayscaleKernel =
        "__kernel void convertToGrayscale(__global const uchar4 * image, __global uchar* grayscale, int width, int height) {"
        "    int x = get_global_id(0);                                                                                      "
        "    int y = get_global_id(1);                                                                                      "
        "    if (x >= width || y >= height) return;                                                                         "
        "    int single_dim_index = y * width + x;                                                                          "
        "    uchar4 pixel = image[single_dim_index];                                                                        "
        "    uchar gray_conversion = (uchar)(0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z);                    "
        "    grayscale[single_dim_index] = gray_conversion;                                                                 "
        "}                                                                                                                  "
        ;

    const char* filterKernel =
        "__kernel void gaussianBlurFilter(__global const uchar4* image, __global uchar4* filtered, int width, int height) {"
        "    int x = get_global_id(0);                                                                                    "
        "    int y = get_global_id(1);                                                                                    "
        "    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) {                                                   "
        "        filtered[y * width + x] = image[y * width + x];                                                          "
        "        return;                                                                                                  "
        "    }                                                                                                            "
        "    const float filter[5][5] = {                                                                                 "
        "        {1.0f / 273, 4.0f / 273, 7.0f / 273, 4.0f / 273, 1.0f / 273},                                            "
        "        {4.0f / 273, 16.0f / 273, 26.0f / 273, 16.0f / 273, 4.0f / 273},                                         "
        "        {7.0f / 273, 26.0f / 273, 41.0f / 273, 26.0f / 273, 7.0f / 273},                                         "
        "        {4.0f / 273, 16.0f / 273, 26.0f / 273, 16.0f / 273, 4.0f / 273},                                         "
        "        {1.0f / 273, 4.0f / 273, 7.0f / 273, 4.0f / 273, 1.0f / 273}                                             "
        "    };                                                                                                           "
        "    float4 sum = (float4)(0.0f);                                                                                 "
        "    for (int ky = -2; ky <= 2; ky++) {                                                                           "
        "        int yi = y + ky;                                                                                         "
        "        for (int kx = -2; kx <= 2; kx++) {                                                                       "
        "            int xi = x + kx;                                                                                     "
        "            float weight = filter[ky + 2][kx + 2];                                                               "
        "            sum += convert_float4(image[yi * width + xi]) * weight;                                              "
        "        }                                                                                                        "
        "    }                                                                                                            "
        "    filtered[y * width + x] = convert_uchar4(sum + (float4)(0.5f));                                              "
        "}                                                                                                                "
        ;


    double curr_time = get_time();

    const char* filename = "C:/Users/Tommi/Desktop/MultiProcessor/Project1/Project1/im0.png";
    unsigned char* image = NULL;
    unsigned width = 0, height = 0;

    unsigned error;

    error = lodepng_decode32_file(&image, &width, &height, filename);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return;
    }
    printf("width is: %u, height is: %u\n", width, height);

    unsigned resized_width = width / 4, resized_height = height / 4;

    printf("resized width will be: %u, resized height will be: %u\n", resized_width, resized_height);

    unsigned char* resized_image = (unsigned char*)malloc(resized_width * resized_height * 4);
    unsigned char* gray_image = (unsigned char*)malloc(width * height);
    unsigned char* filtered_image = (unsigned char*)malloc(width * height * 4);

    // Platform statistics -----------------------------------------------------------------------------------
    cl_int status;
    cl_platform_id platforms[64];
    unsigned int platform_count;
    cl_int platform_result = clGetPlatformIDs(64, platforms, &platform_count);
    assert(platform_result == CL_SUCCESS);
    printf("%d\n", platform_count);
    char platform_name[256];
    size_t platform_name_length;

    size_t  str_info_size;
    char* str_info;
    cl_uint uint_info;
    cl_uint       num_devices;
    cl_device_id* devices;
    for (int i = 0; i < platform_count; ++i) {
        printf("  Platform:\n");

        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &str_info_size);
        str_info = (char*)malloc(str_info_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, str_info_size, str_info, NULL);
        printf("%s\n", str_info);
        free(str_info);

        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 0, NULL, &str_info_size);
        str_info = (char*)malloc(str_info_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, str_info_size, str_info, NULL);
        printf("%s\n", str_info);
        free(str_info);

        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 0, NULL, &str_info_size);
        str_info = (char*)malloc(str_info_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, str_info_size, str_info, NULL);
        printf("%s\n", str_info);
        free(str_info);

        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        printf("Number of devices: %d\n", num_devices);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (int j = 0; j < num_devices; ++j) {
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &str_info_size);
            str_info = (char*)malloc(str_info_size);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, str_info_size, str_info, NULL);
            printf("Device name: %s\n", str_info);
            free(str_info);

            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint_info), &uint_info, NULL);
            printf("Device maximum compute units: %d\n", uint_info);

            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint_info), &uint_info, NULL);
            printf("Device maximum clock frq: %d\n", uint_info);

        }
    }
    //--------------------------------------------------------------------------------------------------------

    // OpenCL variables for resize kernel, duplicates for other programs and kernels made from these
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer_input;
    cl_mem buffer_result;
    FILE* program_handle;
    char* program_buffer, * program_log;
    size_t program_size, log_size;

    err = clGetPlatformIDs(1, &platform, NULL);
    assert(err == CL_SUCCESS);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(err == CL_SUCCESS);
    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS);


    buffer_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, image, &err);
    assert(err == CL_SUCCESS);
    cl_mem grayscale_buffer_input;
    grayscale_buffer_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, image, &err);
    assert(err == CL_SUCCESS);
    cl_mem filter_buffer_input;
    filter_buffer_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, image, &err);
    assert(err == CL_SUCCESS);

    buffer_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, resized_width * resized_height * 4, NULL, &err);
    assert(err == CL_SUCCESS);
    cl_mem grayscale_buffer_result;
    grayscale_buffer_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, width * height, NULL, &err);
    assert(err == CL_SUCCESS);
    cl_mem filter_buffer_result;
    filter_buffer_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, width * height * 4, NULL, &err);
    assert(err == CL_SUCCESS);

    // Resize program
    program = clCreateProgramWithSource(ctx, 1,
        (const char**)&resizeKernel, NULL, &err);
    if (err < 0) {
        perror("Error creating program");
        exit(1);
    }

    // Grayscale program
    cl_program grayscale_program;
    grayscale_program = clCreateProgramWithSource(ctx, 1,
        (const char**)&grayscaleKernel, NULL, &err);
    if (err < 0) {
        perror("Error creating program");
        exit(1);
    }

    // Filter program
    cl_program filter_program;
    filter_program = clCreateProgramWithSource(ctx, 1,
        (const char**)&filterKernel, NULL, &err);
    if (err < 0) {
        perror("Error creating program");
        exit(1);
    }

    // Build all programs
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err < 0) {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    err = clBuildProgram(grayscale_program, 1, &device, NULL, NULL, NULL);

    if (err < 0) {
        clGetProgramBuildInfo(grayscale_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(grayscale_program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    err = clBuildProgram(filter_program, 1, &device, NULL, NULL, NULL);

    if (err < 0) {
        clGetProgramBuildInfo(filter_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(filter_program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    // Create kernels for all programs
    kernel = clCreateKernel(program, RESIZE_KERNEL_FUNC, &err);
    if (err < 0) {
        perror("Couldn't create the kernel");
        exit(1);
    }
    cl_kernel grayscale_kernel;
    grayscale_kernel = clCreateKernel(grayscale_program, GRAYSCALE_KERNEL_FUNC, &err);
    if (err < 0) {
        perror("Couldn't create the kernel");
        exit(1);
    }
    cl_kernel filter_kernel;
    filter_kernel = clCreateKernel(filter_program, FILTER_KERNEL_FUNC, &err);
    if (err < 0) {
        perror("Couldn't create the kernel");
        exit(1);
    }


    // Set arguments for resize kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_input);
    if (err < 0) {
        perror("Error setting the kernel argument");
        exit(1);
    }
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_result);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &height);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &resized_width);
    clSetKernelArg(kernel, 5, sizeof(unsigned int), &resized_height);


    // Create command queue
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err < 0) {
        perror("Error creating the command queue");
        exit(1);
    }

    // Execute the resize kernel
    size_t globalSize[2] = {resized_width, resized_height};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);

    // Read the result buffer
    err = clEnqueueReadBuffer(queue, buffer_result, CL_TRUE, 0, resized_width * resized_height * 4, resized_image, 0, NULL, NULL);

    // Save the resized image result as PNG
    error = lodepng_encode32_file("C:/Users/Tommi/Desktop/MultiProcessor/Project1/Project1/im0_resize.png", resized_image, resized_width, resized_height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }



    // Set arguments for grayscale kernel
    clSetKernelArg(grayscale_kernel, 0, sizeof(cl_mem), &grayscale_buffer_input);
    clSetKernelArg(grayscale_kernel, 1, sizeof(cl_mem), &grayscale_buffer_result);
    int w = (int)width, h = (int)height;
    clSetKernelArg(grayscale_kernel, 2, sizeof(unsigned int), &w);
    clSetKernelArg(grayscale_kernel, 3, sizeof(unsigned int), &h);

    // Execute grayscale kernel
    size_t grayscaleGlobalSize[2] = { width, height };
    err = clEnqueueNDRangeKernel(queue, grayscale_kernel, 2, NULL, grayscaleGlobalSize, NULL, 0, NULL, NULL);

    // Read the result buffer for grayscale image
    err = clEnqueueReadBuffer(queue, grayscale_buffer_result, CL_TRUE, 0, width * height, gray_image, 0, NULL, NULL);


    // Convert grayscale data to rgba and save to png
    unsigned char* output_rgba = (unsigned char*)malloc(width * height * 4);
    for (unsigned i = 0; i < width * height; i++) {
        output_rgba[i * 4 + 0] = gray_image[i]; // R
        output_rgba[i * 4 + 1] = gray_image[i]; // G
        output_rgba[i * 4 + 2] = gray_image[i]; // B
        output_rgba[i * 4 + 3] = 255;          // A
    }
    error = lodepng_encode32_file("C:/Users/Tommi/Desktop/MultiProcessor/Project1/Project1/im0_bw.png", output_rgba, width, height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }

    // Set arguments for filter kernel
    clSetKernelArg(filter_kernel, 0, sizeof(cl_mem), &filter_buffer_input);
    clSetKernelArg(filter_kernel, 1, sizeof(cl_mem), &filter_buffer_result);
    clSetKernelArg(filter_kernel, 2, sizeof(unsigned int), &width);
    clSetKernelArg(filter_kernel, 3, sizeof(unsigned int), &height);

    // Execute filter kernel
    size_t filterGlobalSize[2] = { width, height };
    err = clEnqueueNDRangeKernel(queue, filter_kernel, 2, NULL, filterGlobalSize, NULL, 0, NULL, NULL);

    // Read the result buffer for filtered image
    err = clEnqueueReadBuffer(queue, filter_buffer_result, CL_TRUE, 0, width * height * 4, filtered_image, 0, NULL, NULL);

    error = lodepng_encode32_file("C:/Users/Tommi/Desktop/MultiProcessor/Project1/Project1/im0_filtered.png", filtered_image, width, height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }


    free(image);
    free(resized_image);
    free(gray_image);
    free(output_rgba);
    free(filtered_image);
    clReleaseMemObject(buffer_input);
    clReleaseMemObject(grayscale_buffer_input);
    clReleaseMemObject(filter_buffer_input);
    clReleaseMemObject(buffer_result);
    clReleaseMemObject(grayscale_buffer_result);
    clReleaseMemObject(filter_buffer_result);
    clReleaseKernel(kernel);
    clReleaseKernel(grayscale_kernel);
    clReleaseKernel(filter_kernel);
    clReleaseProgram(program);
    clReleaseProgram(grayscale_program);
    clReleaseProgram(filter_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);



    double time_after_exec = get_time();
    // printf("time b4 exec: %f\n", curr_time);
    // printf("time after exec: %f\n", time_after_exec);

    double execution_time = time_after_exec - curr_time;
    printf("Execution time: %f (seconds)\n", execution_time);

    return 0;
}

double get_time() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}