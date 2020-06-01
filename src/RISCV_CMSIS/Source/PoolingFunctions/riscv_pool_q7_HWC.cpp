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
 * Title:        riscv_pool_int8_HWC.c
 * Description:  Pooling function implementations
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
 * @addtogroup Pooling
 * @{
 */

static void compare_and_replace_if_larger_q7(int8_t * base,   // base data
                                             const int8_t * target,   // compare target
                                             const uint8_t length  // data length
                                             )
{
    int8_t     *pIn = base;
    const int8_t     *pCom = target;
    uint8_t  cnt = length & 0xFFFC;
    uint8_t   tmp_vl = 0;

    while (cnt)
    {
      vmax_vv<int8_t>(pIn, pCom, cnt, &tmp_vl, pIn);
      cnt -= tmp_vl;
      pIn += tmp_vl;
      pCom += tmp_vl;
    }

    cnt = length & 0x3;
    while(cnt)
    {
      if (*pCom > *pIn)
      {
        *pIn = *pCom;
      }
      cnt--;
      pIn++;
      pCom++;
    }
}

  /**
   * @brief int8 max pooling function
   * @param[in, out]  Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimention
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     Not used
   * @param[in,out]   Im_out      pointer to output tensor
   *
   * @details
   *
   * The pooling function is implemented as split x-pooling then
   * y-pooling.
   *
   * This pooling function is input-destructive. Input data is undefined
   * after calling this function.
   *
   */

void
riscv_maxpool_int8_HWC(int8_t * Im_in,
                   const uint16_t dim_im_in,
                   const uint16_t ch_im_in,
                   const uint16_t dim_kernel,
                   const uint16_t padding,
                   const uint16_t stride, const uint16_t dim_im_out, int8_t * bufferA, int8_t * Im_out)
{
#if defined(USE_VEXT)
#warning "Using V-Extension"

    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_x, i_y;

    // first does the pooling along x axis 
    for (i_y = 0; i_y < dim_im_in; i_y++)
    {
      for (i_x = 0; i_x < dim_im_out; i_x++)
      {
        // for each output pixel 
        int8_t     *target = Im_in + (i_y * dim_im_in + i_x) * ch_im_in;
        int8_t     *win_start;
        int8_t     *win_stop;
        if (i_x * stride - padding < 0)
        { win_start = target; }
        else
        { win_start = Im_in + (i_y * dim_im_in + i_x * stride - padding) * ch_im_in; }

        if (i_x * stride - padding + dim_kernel >= dim_im_in) 
        { win_stop = Im_in + (i_y * dim_im_in + dim_im_in) * ch_im_in; } 
        else 
        { win_stop = Im_in + (i_y * dim_im_in + i_x * stride - padding + dim_kernel) * ch_im_in; }

        // first step is to copy over initial data 
        memmove(target, win_start, ch_im_in);

        // start the max operation from the second part 
        win_start += ch_im_in;
        for (; win_start < win_stop; win_start += ch_im_in)
        {
          compare_and_replace_if_larger_q7(target, win_start, ch_im_in);
        }
      }
    }

    /* then does the pooling along y axis */
    for (i_y = 0; i_y < dim_im_out; i_y++)
    {
      /* for each output row */
      int8_t     *target = Im_out + i_y * dim_im_out * ch_im_in;
      int8_t     *row_start;
      int8_t     *row_end;
      /* setting the starting row */
      if (i_y * stride - padding < 0)
      { row_start = Im_in; } 
      else
      { row_start = Im_in + (i_y * stride - padding) * dim_im_in * ch_im_in; }
      /* setting the stopping row */
      if (i_y * stride - padding + dim_kernel >= dim_im_in)
      { row_end = Im_in + dim_im_in * dim_im_in * ch_im_in; } 
      else
      { row_end = Im_in + (i_y * stride - padding + dim_kernel) * dim_im_in * ch_im_in; }

      /* copy over the first row */
      /* arm_copy_q7(row_start, target, dim_im_out * ch_im_in); */
      memmove(target, row_start, dim_im_out * ch_im_in);

      /* move over to next row */
      row_start += ch_im_in * dim_im_in;

      for (; row_start < row_end; row_start += dim_im_in * ch_im_in)
      {
        compare_and_replace_if_larger_q7(target, row_start, dim_im_out * ch_im_in);
      }
    }
#else
    (void)bufferA;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out; i_x++)
            {
                int       max = -129;
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in && k_x < dim_im_in)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out)] = max;
            }
        }
    }
#endif
}

  /**
   * @brief int8 average pooling function
   * @param[in,out]   Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimention
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size:  2*dim_im_out*ch_im_in
   *
   * The pooling function is implemented as split x-pooling then
   * y-pooling.
   *
   * This pooling function is input-destructive. Input data is undefined
   * after calling this function.
   *
   */

void
riscv_avepool_int8_HWC(int8_t * Im_in,
                   const uint16_t dim_im_in,
                   const uint16_t ch_im_in,
                   const uint16_t dim_kernel,
                   const uint16_t padding,
                   const uint16_t stride, const uint16_t dim_im_out, int8_t * bufferA, int8_t * Im_out)
{
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    (void)bufferA;
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out; i_x++)
            {
                int       sum = 0;
                int       count = 0;
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in && k_x < dim_im_in)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)];
                            count++;
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out)] = sum / count;
            }
        }
    }
}

/**
 * @} end of Pooling group
 */
