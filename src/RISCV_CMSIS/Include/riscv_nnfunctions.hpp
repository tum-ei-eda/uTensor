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

#define int32_MAX   ((int32_t)(0x7FFFFFFFL))
#define int16_MAX   ((int16_t)(0x7FFF))
#define int8_MAX    ((int8_t)(0x7F))
#define int32_MIN   ((int32_t)(0x80000000L))
#define int16_MIN   ((int16_t)(0x8000))
#define int8_MIN    ((int8_t)(0x80))

#define NN_ROUND(out_shift) ( (0x1u << out_shift) >> 1  )
#define LEFT_SHIFT(_shift)  (_shift > 0 ? _shift : 0)
#define RIGHT_SHIFT(_shift) (_shift > 0 ? 0 : -_shift)
#define MASK_IF_ZERO(x)     (x) == 0 ? ~0 : 0
#define MASK_IF_NON_ZERO(x) (x) != 0 ? ~0 : 0
#define SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define CLAMP(x, h, l) MAX(MIN((x), (h)), (l))


typedef enum
{
  RISCV_MATH_SUCCESS        =  0,        /**< No error */
  RISCV_MATH_ARGUMENT_ERROR = -1,        /**< One or more arguments are incorrect */
  RISCV_MATH_LENGTH_ERROR   = -2,        /**< Length of data buffer is incorrect */
  RISCV_MATH_SIZE_MISMATCH  = -3,        /**< Size of matrices is not compatible with the operation */
  RISCV_MATH_NANINF         = -4,        /**< Not-a-number (NaN) or infinity is generated */
  RISCV_MATH_SINGULAR       = -5,        /**< Input matrix is singular and cannot be inverted */
  RISCV_MATH_TEST_FAILURE   = -6         /**< Test Failed */
} riscv_status;

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


int32_t riscv_fully_connected_s8_get_buffer_size(const uint16_t col_dim);

/**
 ** @brief           Rounding divide by power of two.
 ** @param[in]       dividend - Dividend
 ** @param[in]       exponent - Divisor = power(2, exponent)
 **                             Range: [0, 31]
 ** @return          Rounded result of division. Midpoint is rounded away from zero.
 **
 **/
static inline int32_t riscv_nn_divide_by_power_of_two(const int32_t dividend, const int32_t exponent)
{
  int32_t result = 0;
  const int32_t remainder_mask = (1l << exponent) - 1;
  int32_t remainder = remainder_mask & dividend;

  // Basic division
  result = dividend >> exponent;
  // Adjust 'result' for rounding (mid point away from zero)
  int32_t threshold = remainder_mask >> 1;
  if (result < 0)
  {
    threshold++;
  }
  if (remainder > threshold)
  {
    result++;
  }
  return result;
}

/**
 ** @brief           Saturating doubling high multiply. Result matches
 **                  NEON instruction VQRDMULH.
 ** @param[in]       m1        Multiplicand
 ** @param[in]       m2        Multiplier
 ** @return          Result of multiplication.
 **
 **/
static inline int32_t riscv_nn_sat_doubling_high_mult(const int32_t m1, const int32_t m2)
{
      int32_t result = 0;
      // Rounding offset to add for a right shift of 31
      int64_t mult = 1 << 30;
      
      if ((m1 < 0) ^ (m2 < 0))
      {
        mult = 1 - mult;
      }
      // Gets resolved as a SMLAL instruction
      mult = mult + (int64_t)m1 * m2;
      // Utilize all of the upper 32 bits. This is the doubling step
      // as well.
      result = mult / (1UL << 31);
      if ((m1 == m2) && (m1 == (int32_t)int32_MIN))
      {
        result = int32_MAX;
      }
      return result;
}


/**
 ** @brief           Requantize a given value.
 ** @param[in]       val         Value to be requantized
 ** @param[in]       multiplier  multiplier
 ** @param[in]       shift       left or right shift for 'val * multiplier'
 **
 ** @return          Returns (val * multiplier)/(2 ^ shift)
 **
 **/
static inline int32_t riscv_nn_requantize(const int32_t val, const int32_t multiplier, const int32_t shift)
{
    return riscv_nn_divide_by_power_of_two(riscv_nn_sat_doubling_high_mult(val * (1 << LEFT_SHIFT(shift)), multiplier),
                                               RIGHT_SHIFT(shift));
    
}


/**
 ** @defgroup BasicMath Basic math functions
 **
 ** Element wise add and multiplication functions.
 **
 **/

/**
 ** @brief s8 element wise add of two vectors
 ** @param[in]       input_1_vect            pointer to input vector 1
 ** @param[in]       input_2_vect            pointer to input vector 2
 ** @param[in]       input_1_offset          offset for input 1. Range: Range: -127 to 128
 ** @param[in]       input_1_mult            multiplier for input 1
 ** @param[in]       input_1_shift           shift for input 1
 ** @param[in]       input_2_offset          offset for input 2. Range: Range: -127 to 128
 ** @param[in]       input_2_mult            multiplier for input 2
 ** @param[in]       input_2_shift           shift for input 2
 ** @param[in]       left_shift              input left shift
 ** @param[in,out]   output                  pointer to output vector
 ** @param[in]       out_offset              output offset
 ** @param[in]       out_mult                output multiplier
 ** @param[in]       out_shift               output shift
 ** @param[in]       out_activation_min      minimum value to clamp output to
 ** @param[in]       out_activation_max      maximum value to clamp output to
 ** @param[in]       block_size              number of samples
 ** @return          The function returns    ARM_MATH_SUCCESS
 **/
riscv_status
riscv_elementwise_add_s8(const int8_t *input_1_vect,
                       const int8_t *input_2_vect,
                       const int32_t input_1_offset,
                       const int32_t input_1_mult,
                       const int32_t input_1_shift,
                       const int32_t input_2_offset,
                       const int32_t input_2_mult,
                       const int32_t input_2_shift,
                       const int32_t left_shift,
                       int8_t *output,
                       const int32_t out_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t out_activation_min,
                       const int32_t out_activation_max,
                       const uint32_t block_size);
riscv_status
riscv_elementwise_mul_s8(const int8_t *input_1_vect,
                       const int8_t *input_2_vect,
                       const int32_t input_1_offset,
                       const int32_t input_2_offset,
                       int8_t *output,
                       const int32_t out_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t out_activation_min,
                       const int32_t out_activation_max,
                       const uint32_t block_size);
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
 ** @brief s8 Vector by Matrix (transposed) multiplication
 **
 ** @param[in]      lhs             Input left-hand side vector
 ** @param[in]      rhs             Input right-hand side matrix (transposed)
 ** @param[in]      bias            Input bias
 ** @param[out]     dst             Output vector
 ** @param[in]      lhs_offset      Offset to be added to the input values of the left-hand side vector. Range: -127 to 128
 ** @param[in]      rhs_offset      Offset to be added to the input values of the right-hand side matrix. Range: -127 to 128
 ** @param[in]      dst_offset      Offset to be added to the output values. Range: -127 to 128
 ** @param[in]      dst_multiplier  Output multiplier
 ** @param[in]      dst_shift       Output shift
 ** @param[in]      rhs_cols        Number of columns in the right-hand side input matrix
 ** @param[in]      rhs_rows        Number of rows in the right-hand side input matrix
 ** @param[in]      activation_min  Minimum value to clamp the output to. Range: int8
 ** @param[in]      activation_max  Maximum value to clamp the output to. Range: int8
 **
 ** @return         The function returns <code>ARM_MATH_SUCCESS</code>
 **
 **/
riscv_status
riscv_fully_connected_s8(const int8_t *input,
                       const int8_t *kernel,
                       const uint16_t col_dim,
                       const uint16_t row_dim,
                       const uint16_t nb_batches,
                       const int32_t input_offset,
                       const int32_t filter_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t output_offset,
                       const int32_t *bias,
                       int8_t *output,
                       const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       int16_t *vec_buffer);

riscv_status riscv_nn_vec_mat_mult_t_s8(const int8_t *lhs,
                                    const int8_t *rhs,
                                    const int32_t *bias,
                                    int8_t *dst,
                                    const int32_t lhs_offset,
                                    const int32_t rhs_offset,
                                    const int32_t dst_offset,
                                    const int32_t dst_multiplier,
                                    const int32_t dst_shift,
                                    const int32_t rhs_cols,
                                    const int32_t rhs_rows,
                                    const int32_t activation_min,
                                    const int32_t activation_max);

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
   * @brief int16 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   */

void riscv_softmax_int16(const int16_t * vec_in, const uint16_t dim_vec, int16_t * p_out);

#endif
