#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <windows.h>
#include <CL/opencl.h>
#include "lodepng.h"

float** allocate_memory_to_matrix(int rows, int cols);
float** allocate_values_to_matrix(float** matrix, int rows, int cols, float value);
float** add_matrix(float** matrix_1, float** matrix_2, int rows, int cols);
void print_matrix_values(float** matrix, int rows, int cols);
void free_matrix_memory(float** matrix, int rows, int cols);
double get_time();
void ReadImage(const char* filename, unsigned char** image, unsigned *width, unsigned *height);
void resize_image(unsigned char** image, unsigned* width, unsigned* height, unsigned char** resized_image);
void GrayScaleImage(unsigned char** image, unsigned* width, unsigned* height, unsigned char** gray_image);
void ApplyFilter(unsigned char** image, unsigned* width, unsigned* height, unsigned char** blurred_image);
void WriteImage(const char* filename, const unsigned char** image, unsigned* width, unsigned* height);


int main() {

    cl_int status;

    double curr_time = get_time();

    int rows = 100;
    int cols = 100;
    const char* filename = "<path_to_file>/im0.png";
    unsigned char* resized_image = NULL;
    unsigned char* gray_image = NULL;
    unsigned char* image = NULL;
    unsigned char* blurred_image = NULL;
    unsigned width = 0, height = 0;
    unsigned resized_width = 735, resized_height = 504;
    ReadImage(filename, &image, &width, &height);
    printf("width is: %u, height is: %u\n", width, height);

    resize_image(&image, &width, &height, &resized_image);
    GrayScaleImage(&image, &width, &height, &gray_image);
    ApplyFilter(&image, &width, &height, &blurred_image);
    WriteImage("<path_to_file>/im0_resize.png", &resized_image, &resized_width, &resized_height);
    WriteImage("<path_to_file>/im0_bw.png", &gray_image, &width, &height);
    WriteImage("<path_to_file>/im0_blurred.png", &blurred_image, &width, &height);
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
    return 0;
}

void ApplyFilter(unsigned char** image, unsigned* width, unsigned* height, unsigned char** blurred_image) {
    // source: gaussian blur discrete approximation www.researchgate.net/figure/Discrete-approximation-of-the-Gaussian-kernels-3x3-5x5-7x7_fig2_325768087
    const float filter[5][5] = {
        {1.0f / 273, 4.0f / 273, 7.0f / 273, 4.0f / 273, 1.0f / 273},
        {4.0f / 273, 16.0f / 273, 26.0f / 273, 16.0f / 273, 4.0f / 273},
        {7.0f / 273, 26.0f / 273, 41.0f / 273, 26.0f / 273, 7.0f / 273},
        {4.0f / 273, 16.0f / 273, 26.0f / 273, 16.0f / 273, 4.0f / 273},
        {1.0f / 273, 4.0f / 273, 7.0f / 273, 4.0f / 273, 1.0f / 273}
    };
    *blurred_image = (unsigned char*)malloc((* width) * (* height) * 4 * sizeof(unsigned char));
    if (!blurred_image) {
        printf("Error: Memory allocation failed.\n");
        return;
    }

    for (unsigned j = 2; j < (*height) - 2; j++) {
        for (unsigned i = 2; i < (*width) - 2; i++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            unsigned center_index = 4 * (j * (* width) + i);

            for (int fj = -2; fj <= 2; fj++) {
                for (int fi = -2; fi <= 2; fi++) {
                    unsigned neighbor_index = 4 * ((j + fj) * (* width) + (i + fi));
                    float weight = filter[fj + 2][fi + 2];

                    r += (*image)[neighbor_index] * weight;
                    g += (*image)[neighbor_index + 1] * weight;
                    b += (*image)[neighbor_index + 2] * weight;
                }
            }

            (* blurred_image)[center_index] = (unsigned char)(r < 0 ? 0 : (r > 255 ? 255 : r));
            (*blurred_image)[center_index + 1] = (unsigned char)(g < 0 ? 0 : (g > 255 ? 255 : g));
            (*blurred_image)[center_index + 2] = (unsigned char)(b < 0 ? 0 : (b > 255 ? 255 : b));
            (*blurred_image)[center_index + 3] = (*image)[center_index + 3];
        }
    }
}

void WriteImage(const char* filename, const unsigned char** image, unsigned* width, unsigned* height){
    /*Encode the image*/
    unsigned error = lodepng_encode32_file(filename, *image, *width, *height);

    /*if there's an error, display it*/
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
    }
}

void GrayScaleImage(unsigned char** image, unsigned* width, unsigned* height, unsigned char** gray_image) {
    *gray_image = (unsigned char*)malloc((* width) * (* height) * 4 * sizeof(unsigned char));
    for (unsigned int i = 0; i < *width; i++) {
        for (unsigned int j = 0; j < *height; j++) {
            unsigned int index = 4 * (j * (*width) + i);
            // r * 0.2126 + g * 0.7152 + b * 0.0722 --> gray
            unsigned char gray_value = (unsigned char)(((*image)[index] * 0.2126) + ((*image)[index + 1] * 0.7152) + ((*image)[index + 2] * 0.0722));
            (*gray_image)[index] = gray_value;
            (*gray_image)[index + 1] = gray_value;
            (*gray_image)[index + 2] = gray_value;
            (*gray_image)[index + 3] = (*image)[index + 3];
        }
    }
}

void resize_image(unsigned char** image, unsigned* width, unsigned* height, unsigned char** resized_image) {
    // reduces the size of the picture by 4 ==> 1/16th of the original image
    unsigned int new_width = *width / 4;
    unsigned int new_height = *height / 4;
    printf("\nnew width is %u, new height is: %u\n", new_width, new_height);
    *resized_image = (unsigned char*)malloc(new_width * new_height * 4 * sizeof(unsigned char));

    for (unsigned int i = 0; i < new_width; i++) {
        for (unsigned int j = 0; j < new_height; j++) {
            unsigned int index = 4 * (j * new_width + i);
            unsigned int original_index = 4 * ((j * 4) * (*width) + (i * 4));
            (*resized_image)[index] = (*image)[original_index];
            (*resized_image)[index + 1] = (*image)[original_index + 1];
            (*resized_image)[index + 2] = (*image)[original_index + 2];
            (*resized_image)[index + 3] = (*image)[original_index + 3];
        }
    }
}

void ReadImage(const char* filename, unsigned char** image, unsigned* width, unsigned* height) {
    unsigned error;

    error = lodepng_decode32_file(image, width, height, filename);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return;
    }

    //free(*image);
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