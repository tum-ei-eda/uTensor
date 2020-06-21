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
 * Title:        riscv_convolve_HWC_int8_basic.c
 * Description:	 int8 version of convolution
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
   * @brief Basic int8 convolution function
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
   * @return     The function returns <code>riscv_MATH_SUCCESS</code>
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
   *
   * bufferB size: 0
   *
   * This basic version is designed to work for any input tensor and weight
   * dimension.
   */

void
riscv_convolve_HWC_int8_basic(const int8_t * Im_in,
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
#if defined(USE_VEXT)
  int8_t    *pBuffer = (int8_t *)bufferA;
  int8_t    *pOut = Im_out;
  int8_t    *im_buffer = (int8_t *)bufferA;
  const int8_t *pA;
  int       i;
  int sum = 0;
  int sumV = 0;
  unsigned char tmp_val = 0;

  for(int i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
  {
    for(int i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
    {
      for(int i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
      {
        for(int i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
        {
          if(i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in) // padding, fill with 0's
          {
            memset(pBuffer, 0, sizeof(int8_t)*ch_im_in);
          }
          else // copy into buffer
          {
            memcpy(pBuffer, (int8_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, sizeof(int8_t)*ch_im_in);
          }
          pBuffer += ch_im_in;
        }
      }
      pA = wt;

      for(i = 0; i < ch_im_out; i++)
      {
        sum = ((int)bias[i] << bias_shift) + NN_ROUND(out_shift);
        const int8_t *pB = im_buffer;
        unsigned short colCnt = ch_im_in * dim_kernel * dim_kernel & 0xFFFC;
        while (colCnt)
        {
          vmul_vv<signed char>(pA, pB, colCnt, &tmp_val, (int8_t *) &sumV);
          sum += (sumV & 0xFF) + ((sumV >> 8) & 0xFF) + ((sumV >> 16) & 0xFF) + ((sumV >> 24) & 0xFF);
          pB += tmp_val;
          pA += tmp_val;
          colCnt -= tmp_val;
        }
        colCnt = ch_im_in * dim_kernel * dim_kernel & 0x3;
        while (colCnt)
        {
          int8_t     inA1 = *pA++;
          int8_t     inB1 = *pB++;
          sum += inA1 * inB1;
          colCnt--;
        }
        *pOut++ = (int8_t) __SSAT((sum >> out_shift), 8);
      }

      /* counter reset */
      pBuffer = im_buffer;
    }
  }
#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    signed char in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = ((int32_t)bias[i] << bias_shift);
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
#endif
}

/**
 * @} end of NNConv group
 */
