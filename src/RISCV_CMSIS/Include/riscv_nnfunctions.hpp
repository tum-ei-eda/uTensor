/**
   \mainpage CMSIS NN Software Library
   *
   * Introduction
   * ------------
   *
   * This user manual describes the CMSIS NN software library,
   * a collection of efficient neural network kernels developed to maximize the
   * performance and minimize the memory footprint of neural networks on Cortex-M processor cores.
   *
   * The library is divided into a number of functions each covering a specific category:
   * - Neural Network Convolution Functions
   * - Neural Network Activation Functions
   * - Fully-connected Layer Functions
   * - Neural Network Pooling Functions
   * - Softmax Functions
   * - Neural Network Support Functions
   */

/**
 * @defgroup groupNN Neural Network Functions
 * These functions perform basic operations for neural network layers.
 */

#ifndef _RISCV_NNFUNCTIONS_HPP
#define _RISCV_NNFUNCTIONS_HPP

#include <stdint.h>
#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <unistd.h>
#if defined(USE_VEXT)
#include "riscv_vext.hpp"
#endif

#define NN_ROUND(out_shift) ( (0x1u << out_shift) >> 1  )

static inline int32_t __SSAT(int32_t val, uint32_t sat) 
  {
    if ((sat >= 1U) && (sat <= 32U))
    {    
      const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U); 
      const int32_t min = -1 - max ;
      if (val > max) 
      {    
        return max; 
      }    
      else if (val < min) 
      {    
        return min; 
      }    
    }    
    return val; 
  }

static inline uint32_t __USAT(int32_t val, uint32_t sat) 
  {
    if (sat <= 31U) 
    {    
      const uint32_t max = ((1U << sat) - 1U);
      if (val > (int32_t)max)
      {
        return max;
      }
      else if (val < 0)
      {
        return 0U;
      }
    }
    return (uint32_t)val;
  }

/*
template<typename T>
static inline void __USATX(int32_t * val, uint32_t sat)
{
    if (sat <= 31U) 
    {    
      T * tmp = (T *)val;
      const uint8_t max = ((1U << sat) - 1U);
      for(int i = 0; i < sizeof(int)/sizeof(T); i++)
      {
        if (tmp[i] > (T)max)
        {
          tmp[i] = max;
        }
        else if (tmp[i] < 0)
        {
          tmp[i] = 0;
        }
      }
    }
}
*/
static inline void __USAT16(int16_t * val, uint32_t sat) 
{
  if (sat <= 31U) 
  {    
    const uint16_t max = ((1U << sat) - 1U);
    for(int i = 0; i < sizeof(int)/sizeof(int16_t); i++)
    {
      if (val[i] > (int16_t)max)
      {
        val[i] = max;
      }
      else if (val[i] < 0)
      {
        val[i] = 0;
      }
    }
  }
}

static inline void __USAT8(int8_t * val, uint32_t sat) 
{
  if (sat <= 31U) 
  {    
    const uint8_t max = ((1U << sat) - 1U);
    for(int i = 0; i < sizeof(int); i++)
    {
      if (val[i] > (int8_t)max)
      {
        val[i] = max;
      }
      else if (val[i] < 0)
      {
        val[i] = 0;
      }
    }
  }
}


static inline void __SSAT16(int16_t * val, uint32_t sat) 
{
  if ((sat >= 1U) && (sat <= 32U)) 
  {    
    const int16_t max = (int16_t)((1U << (sat - 1U)) - 1U);
    const int16_t min = -1 - max;
    for(int i = 0; i < sizeof(int)/sizeof(int16_t); i++)
    {
      if (val[i] > max)
      {
        val[i] = max;
      }
      else if (val[i] < min)
      {
        val[i] = min;
      }
    }
  }
}

static inline void __SSAT8(int8_t * val, uint32_t sat) 
{
  if ((sat >= 1U) && (sat <= 32U)) 
  {    
    const int8_t max = (int8_t)((1U << (sat - 1U)) - 1U);
    const int8_t min = -1 - max;
    for(int i = 0; i < sizeof(int); i++)
    {
      if (val[i] > max)
      {
        val[i] = max;
      }
      else if (val[i] < min)
      {
        val[i] = min;
      }
    }
  }
}

/**
 * @defgroup NNConv Neural Network Convolution Functions
 *
 * Perform convolution layer
 *
 * The convolution is implemented in 2 steps: im2col and GEMM
 *
 * im2col is a process of converting each patch of image data into
 * a column. After im2col, the convolution is computed as matrix-matrix
 * multiplication.
 *
 * To reduce the memory footprint, the im2col is performed partially.
 * Each iteration, only a few column (i.e., patches) are generated and
 * computed with GEMM kernels similar to CMSIS-DSP riscv_mat_mult functions.
 *
 */

  /**
   * @brief Basic int8 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns <code>riscv_MATH_SUCCESS</code>
   *
   */
    void riscv_convolve_HWC_int8_basic(const int8_t * Im_in,
                                         const uint16_t dim_im_in,
                                         const uint16_t ch_im_in,
                                         const int8_t * wt,
                                         const uint16_t ch_im_out,
                                         const uint16_t dim_kernel,
                                         const uint16_t padding,
                                         const uint16_t stride,
                                         const int8_t * bias,
                                         const uint16_t bias_shift,
                                         const uint16_t out_shift,
                                         int8_t * Im_out,
                                         const uint16_t dim_im_out,
                                         int16_t * bufferA,
                                         int8_t * bufferB);

  /**
   * @brief Basic int16 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns <code>riscv_MATH_SUCCESS</code>
   *
   */
    void riscv_convolve_HWC_int16_basic(const int16_t * Im_in,
                                          const uint16_t dim_im_in,
                                          const uint16_t ch_im_in,
                                          const int16_t * wt,
                                          const uint16_t ch_im_out,
                                          const uint16_t dim_kernel,
                                          const uint16_t padding,
                                          const uint16_t stride,
                                          const int16_t * bias,
                                          const uint16_t bias_shift,
                                          const uint16_t out_shift,
                                          int16_t * Im_out,
                                          const uint16_t dim_im_out,
                                          int16_t * bufferA,
                                          int8_t * bufferB);



/**
 * @defgroup FC Fully-connected Layer Functions
 *
 * Perform fully-connected layer
 *
 * Fully-connected layer is basically a matrix-vector multiplication
 * with bias. The matrix is the weights and the input/output vectors
 * are the activation values. Supported {weight, activation} precisions
 * include {8-bit, 8-bit}, {16-bit, 16-bit}, and {8-bit, 16-bit}.
 *
 * Here we have two types of kernel functions. The basic function
 * implements the function using regular GEMV approach. The opt functions
 * operates with weights in interleaved formats.
 *
 */

  /**
   * @brief int8 basic fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>riscv_MATH_SUCCESS</code>
   *
   */

    void riscv_fully_connected_int8(const int8_t * pV,
                                      const int8_t * pM,
                                      const uint16_t dim_vec,
                                      const uint16_t num_of_rows,
                                      const uint16_t bias_shift,
                                      const uint16_t out_shift,
                                      const int8_t * bias,
                                      int8_t * pOut,
                                      int16_t * vec_buffer);

  /**
   * @brief int16 basic fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>riscv_MATH_SUCCESS</code>
   *
   */

    void riscv_fully_connected_int16(const int16_t * pV,
                                       const int16_t * pM,
                                       const uint16_t dim_vec,
                                       const uint16_t num_of_rows,
                                       const uint16_t bias_shift,
                                       const uint16_t out_shift,
                                       const int16_t * bias,
                                       int16_t * pOut,
                                       int16_t * vec_buffer);

/**
 * @defgroup Acti Neural Network Activation Functions
 *
 * Perform activation layers, including ReLU (Rectified Linear Unit),
 * sigmoid and tanh
 *
 */

  /**
   * @brief int8 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @return none.
   */

    void      riscv_relu_int8(int8_t *data, uint16_t size);

  /**
   * @brief int8 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       ref_point   new reference point after quantization
   * @return none.
   */

    void      riscv_relu_int8_adj(int8_t *data, uint16_t size, int16_t ref_point);

  /**
   * @brief int16 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @return none.
   */

    void      riscv_relu_int16(int16_t *data, uint16_t size);

/**
 * @defgroup Pooling Neural Network Pooling Functions
 *
 * Perform pooling functions, including max pooling and average pooling
 *
 */

  /**
   * @brief int8 max pooling function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   */

    void      riscv_maxpool_int8_HWC(int8_t * Im_in,
                                 const uint16_t dim_im_in,
                                 const uint16_t ch_im_in,
                                 const uint16_t dim_kernel,
                                 const uint16_t padding,
                                 const uint16_t stride,
                                 const uint16_t dim_im_out,
                                 int8_t * bufferA,
                                 int8_t * Im_out);
  /**
   * @brief int16 max pooling function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   */

    void      riscv_maxpool_int16_HWC(int16_t * Im_in,
                                 const uint16_t dim_im_in,
                                 const uint16_t ch_im_in,
                                 const uint16_t dim_kernel,
                                 const uint16_t padding,
                                 const uint16_t stride,
                                 const uint16_t dim_im_out,
                                 int16_t * bufferA,
                                 int16_t * Im_out);

  /**
   * @brief int8 average pooling function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   */

    void      riscv_avepool_int8_HWC(int8_t * Im_in,
                                 const uint16_t dim_im_in,
                                 const uint16_t ch_im_in,
                                 const uint16_t dim_kernel,
                                 const uint16_t padding,
                                 const uint16_t stride,
                                 const uint16_t dim_im_out,
                                 int8_t * bufferA,
                                 int8_t * Im_out);

/**
 * @defgroup Softmax Softmax Functions
 *
 * EXP(2) based softmax function
 *
 */

  /**
   * @brief int8 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   *
   */

void riscv_softmax_int8(const int8_t * vec_in, const uint16_t dim_vec, int8_t * p_out);

  /**
   * @brief int8 softmax function with batch parameter
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       nb_batches  number of batches
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   */

void riscv_softmax_with_batch_int8(const int8_t * vec_in, const uint16_t nb_batches,const uint16_t dim_vec, int8_t * p_out );
  /**
   * @brief int16 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   */

void riscv_softmax_int16(const int16_t * vec_in, const uint16_t dim_vec, int16_t * p_out);

#endif
