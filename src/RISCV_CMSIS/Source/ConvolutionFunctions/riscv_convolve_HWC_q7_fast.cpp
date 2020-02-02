/*
 * Copyright (C) 2010-2018 riscv Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        riscv_convolve_HWC_int8_fast.c
 * Description:  Fast int8 version of convolution
 *
 * $Date:        17. January 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "riscv_nnfunctions.hpp"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

  /**
   * @brief Fast int8 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimention
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
   * @return     The function returns either
   * <code>riscv_MATH_SIZE_MISMATCH</code> or <code>riscv_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
   *
   * bufferB size: 0
   *
   * <b>Input dimension constraints:</b>
   *
   * ch_im_in is multiple of 4    ( because of the SIMD32 read and swap )
   *
   * ch_im_out is multipe of 2    ( bacause 2x2 mat_mult kernel )
   *
   * The im2col converts the int8 tensor input into int16 column, which is stored in
   * bufferA. There is reordering happenning during this im2col process with
   * riscv_int8_to_int16_reordered_no_shift. For every four elements, the second and
   * third elements are swapped.
   *
   * The computation kernel riscv_nn_mat_mult_kernel_int8_int16_reordered does the
   * GEMM computation with the reordered columns.
   *
   * To speed-up the determination of the padding condition, we split the
   * computation into 3x3 parts, i.e., {top, mid, bottom} X {left, mid, right}.
   * This reduces the total number of boundary condition checks and improves
   * the data copying performance.
   */

void
riscv_convolve_HWC_int8_fast(const int8_t * Im_in,
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
                         int8_t * bufferB)
{
    (void)bufferB;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    signed char in_row, in_col;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
    {
      /* check if the input dimension meets the constraints */
      printf("Dimension mismatch\n");
      return;
    }

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = (bias[i] << bias_shift);
                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out +=
                                    Im_in[(in_row * dim_im_in + in_col) * ch_im_in +
                                          l] * wt[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel +
                                                                                            n) * ch_im_in + l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out + k) * ch_im_out] = (int8_t) __SSAT((conv_out >> out_shift), 8);
            }
        }
    }
}

/**
 * @} end of NNConv group
 */
