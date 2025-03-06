#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <windows.h>
#include <CL/opencl.h>

float** allocate_memory_to_matrix(int rows, int cols);
float** allocate_values_to_matrix(float** matrix, int rows, int cols, float value);
float** add_matrix(float** matrix_1, float** matrix_2, int rows, int cols);
void print_matrix_values(float** matrix, int rows, int cols);
void free_matrix_memory(float** matrix, int rows, int cols);
double get_time();


int main() {

    cl_int status;

    double curr_time = get_time();

    int rows = 100;
    int cols = 100;



    float** matrix_1 = allocate_memory_to_matrix(rows, cols);
    float** matrix_2 = allocate_memory_to_matrix(rows, cols);
    if (matrix_1 == NULL || matrix_2 == NULL) {
        printf("ERROR");
        return 1;
    }
    matrix_1 = allocate_values_to_matrix(matrix_1, rows, cols, 1);
    matrix_2 = allocate_values_to_matrix(matrix_2, rows, cols, 2);

    float** matrix_3 = add_matrix(matrix_1, matrix_2, rows, cols);
    free_matrix_memory(matrix_1, rows, cols);
    free_matrix_memory(matrix_2, rows, cols);
    free_matrix_memory(matrix_3, rows, cols);
    matrix_1 = NULL;
    matrix_2 = NULL;
    matrix_3 = NULL;

    double time_after_exec = get_time();
    // printf("time b4 exec: %f\n", curr_time);
    // printf("time after exec: %f\n", time_after_exec);

    double execution_time = time_after_exec - curr_time;
    printf("Execution time: %f (seconds)\n", execution_time);

    cl_platform_id platforms[64];
    unsigned int platform_count;
    cl_int platform_result = clGetPlatformIDs(64, platforms, &platform_count);
    assert (platform_result == CL_SUCCESS);
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


    

    return 0;
}

float** allocate_memory_to_matrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }
    }
    return matrix;
}

float** allocate_values_to_matrix(float** matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = value;
        }
    }
    return matrix;
}

float** add_matrix(float** matrix_1, float** matrix_2, int rows, int cols) {
    float** new_matrix = allocate_memory_to_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix[i][j] = matrix_1[i][j] + matrix_2[i][j];
        }
    }
    //print_matrix_values(new_matrix, rows, cols);
    return new_matrix;
}

void print_matrix_values(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printf("Row %d\n", i + 1);
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void free_matrix_memory(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        free((void*)matrix[i]);
    }
    free((void*)matrix);
}

double get_time() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}