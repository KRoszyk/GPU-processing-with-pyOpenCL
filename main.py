# Author of the code: Kamil Roszyk

import pyopencl as cl
import numpy as np
import cv2
import time

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
context = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(context)  # Create a command queue with your context

# Operation of loading the cat's image as a grayscale image and its parameters to the variables
img = cv2.imread("cute_cat.jpg", 0).astype(np.float32)
img_height = img.shape[0]
img_width = img.shape[1]

##############################################
# Processing for CPU
##############################################
out_cpu = np.zeros(img.shape).astype(np.float32)
out_gpu = np.zeros(img.shape).astype(np.float32)

# Start of counting time for the CPU
start_cpu = time.time()

# Operation of the convolution with a Prewitt mask for an angle of 0 degrees
for row in range(0, img_height-1):
    for col in range(0, img_width-1):
        if col == 0 or row == 0 or row == (img_height-1) or row == (img_width-1):
            out_cpu[row, col] = 0
        else:
            pixel = img[row-1, col+1] + img[row, col+1] + img[row+1, col+1] - img[row-1, col-1] - img[row, col-1] - img[row+1, col-1]
            if pixel > 255:
                pixel = 255
            if pixel < 0:
                pixel = 0
            out_cpu[row, col] = pixel

# End of counting time for the CPU
end_cpu = time.time()

print("Time of executing operations for CPU: ", end_cpu - start_cpu)
cv2.imwrite("cpu_conv.jpg", out_cpu)

##############################################
# Processing for GPU
##############################################

# Creating kernel for GPU in C++ code
program = cl.Program(context, """
__kernel void calculate_conv(__global float *input_gpu, __global float *out_gpu, __global int *width, __global int *height)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    int img_width = *width;
    int img_height = *height;
    int index = row * img_width + col;
    float pixel = 0;
    if (row == 0 || col == 0  || col == img_width - 1 || row == img_height-1)
    {
        out_gpu[index] = 0;
    }
    else
    {
        pixel = input_gpu[index + 1 - img_width] + input_gpu[index + 1] + input_gpu[index + 1 + img_width] - input_gpu[index - 1 - img_width] - input_gpu[index - 1] - input_gpu[index - 1 + img_width];
        if (pixel > 255) pixel = 255;
        if (pixel < 0) pixel = 0;
        out_gpu[index] = pixel;
    }
}""").build()   

# Declaration of buffers for which GPU operations will be performed and passing host variables to them
buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=img)
buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, out_gpu.nbytes)
buffer_width = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int32(img_width))
buffer_height = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int32(img_height))

# Start of counting time for the GPU
start_gpu = time.time()

# Calling a function that performs operations on the graphics card and writing the value of the output buffer to a host variable
program.calculate_conv(queue, img.shape, None, buffer_in, buffer_out, buffer_width, buffer_height)
cl._enqueue_read_buffer(queue, buffer_out, out_gpu).wait()

# End of counting time for the GPU
end_gpu = time.time()

print("Time of executing operations for GPU : ", end_gpu - start_gpu)
cv2.imwrite('gpu_conv.jpg', out_gpu)
print("The image results for GPU and CPU were generated as cpu_conv.jpg and gpu_conv.jpg. The results should be the same.")







