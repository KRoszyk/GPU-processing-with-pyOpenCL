# GPU-processing-with-pyOpenCL

I decided to show the GPU performance against CPU for parallel processing using pyOpenCL.
The operation I chose for this was to execute the convolution of the image shown below with the Prewitt's mask. It is a form of a Prewitt's mask that detects vertical edges.

**Sample image on which processing is performed:**
![Screenshot](https://github.com/KRoszyk/GPU-processing-with-pyOpenCL/blob/main/cute_cat.jpg)

**The form of the Prewitt's mask:**

![Screenshot](https://github.com/KRoszyk/GPU-processing-with-pyOpenCL/blob/main/prewitt.PNG)


# GPU processing

The operations for CPU are basic and only require knowledge of the numpy package. 

For GPU it is more difficult because not only do we need to create a kernel written in **C++**, but we also need to create and handle buffers. We need to specify flags for access to buffers and remember to write the value of the result buffer to the host variable. 
Below you can find kernel code written in C++.

```cpp
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
```
# Results and conclusions
**The processing result for CPU and GPU should be the same and look like the image below:**
![Screenshot](https://github.com/KRoszyk/GPU-processing-with-pyOpenCL/blob/main/gpu_conv.jpg)

We can clearly see that especially the vertical edges like a cat's whiskers have been exposed. This is the result of using this kind of Prewitt's mask. 
Despite the fact that we got the same images for GPU and CPU, the image processing times differ significantly. We can see the results below:
```
Time of executing operations for CPU:  6.437806129455566
Time of executing operations for GPU :  0.025449037551879883
```
Basing on the results we can see that the processing time for CPU is almost 253 times longer than for GPU. This result confirms that for operations that can be paralleled, such as training convolutional neural networks, it is worth using GPU, because it can significantly speed up simple calculations.

# Important information
If you want to use pyOpenCL and test this code, you can find a library file by yourself or download this file from my repository and  enter the command in the project terminal:

```
pip install pyopencl-2020.1cl21-cp37-cp37m-win_amd64.whl
```

