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
 * Title:        riscv_convolve_HWC_int16_fast.c
 * Description:  Fast int16 version of convolution
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
   * @brief Fast int16 convolution function
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
   * ch_im_in is multiple of 2
   *
   * ch_im_out is multipe of 2
   *
   */

void
riscv_convolve_HWC_int16_fast(const int16_t * Im_in,
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
                          int8_t * bufferB)
{
    (void)bufferB;
#if defined(USE_VEXT)
    
    int16_t   i_out_y, i_out_x, i_ker_y, i_ker_x;

    int16_t    *pBuffer = bufferA;
    int16_t    *im_buffer = bufferA;
    int16_t    *pOut = Im_out;

    if (ch_im_in % 2 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return;
    }

    /* Run the following code for Cortex-M4 and Cortex-M7 */

    /* This part implements the im2col function */
    for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(int16_t)*ch_im_in);
                    } else
                    {
                        /* arm_copy_q15((int16_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in); */
                        memcpy(pBuffer, (int16_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, sizeof(int16_t)*ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (i_out_x & 0x1)
            {
                int       i;
                /* initialize the matrix pointers for A */
                const int16_t *pA = wt;

                /* set up the second output pointers */
                int16_t    *pOut2 = pOut + ch_im_out;

                /* this loop over rows in A */
                for (i = 0; i < ch_im_out; i += 2)
                {
                    /* setup pointers for B */
                    const int16_t  *pB = im_buffer;
                    const int16_t *pB2 = pB + ch_im_in * dim_kernel * dim_kernel;

                    /* aling the second pointer for A */
                    const int16_t *pA2 = pA + ch_im_in * dim_kernel * dim_kernel;

                    /* init the sum with bias */
                    int32_t     sum =  ((int32_t)bias[i] << bias_shift);
                    int32_t     sum2 = ((int32_t)bias[i] << bias_shift);
                    int32_t     sum3 = ((int32_t)bias[i + 1] << bias_shift);
                    int32_t     sum4 = ((int32_t)bias[i + 1] << bias_shift);

                    uint16_t  colCnt = ch_im_in * dim_kernel * dim_kernel >> 1;
                    /* accumulate over the vector */
                    int tmp_vl = 0;
                    while (colCnt)
                    {
                      asm volatile ("vsetvli %[tmp_vl], %[colCnt], e16 \n" // set register setting to 16-bit values and calculate tmp_vl=min(maxvl=2, colCnt)
                                    "vlw.v v0, (%[pA]) \n " // load from input Matrix into v0
                                    "vlw.v v1, (%[pB]) \n " // load from input Vector int v1
                                    "vlw.v v2, (%[pA2]) \n " // load from input Matrix into v2
                                    "vlw.v v3, (%[pB2]) \n " // load from input Vector int v3
                                    "vlw.v v4, (%[sum])\n " // load from sum into v4
                                    "vmacc.vv v4, v1, v0 \n"  // v4 = v1 * v0 + v4
                                    "vsw.v v4, (%[sum]) \n"   // save v2 into sum
                                    "vlw.v v4, (%[sum2])\n " // load from sum2 into v4
                                    "vmacc.vv v4, v3, v0 \n"  // v4 = v3 * v0 + v4
                                    "vsw.v v4, (%[sum2]) \n"   // save v2 into sum
                                    "vlw.v v4, (%[sum3])\n " // load from sum3 into v4
                                    "vmacc.vv v4, v2, v1 \n"  // v4 = v1 * v2 + v4
                                    "vsw.v v4, (%[sum3]) \n"   // save v2 into sum
                                    "vlw.v v4, (%[sum4])\n " // load from sum4 into v4
                                    "vmacc.vv v4, v2, v3 \n"  // v4 = v2 * v3 + v4
                                    "vsw.v v4, (%[sum4]) \n"   // save v2 into sum
                                    "add %[pB], %[pB], %[tmp_vl] \n"  // adjust address to input vector
                                    "add %[pA], %[pA], %[tmp_vl] \n"  // adjust address to input vector
                                    "add %[pB2], %[pB2], %[tmp_vl] \n"  // adjust address to input vector
                                    "add %[pA2], %[pA2], %[tmp_vl] \n"  // adjust address to input vector
                                    :[sum] "+r"(sum), [sum2] "+r"(sum2), [sum3] "+r"(sum3), [sum4] "+r"(sum4), [pB] "+r"(pB), [pB2] "+r"(pB2), [pA] "+r"(pA), [pA2] "+r"(pA2)
                                    :[colCnt] "r"(colCnt), [tmp_vl] "r"(tmp_vl));
                        /*
                        int32_t     inA1 = arm_nn_read_q15x2_ia(&pA);
                        int32_t     inB1 = arm_nn_read_q15x2_ia(&pB);
                        int32_t     inA2 = arm_nn_read_q15x2_ia(&pA2);
                        int32_t     inB2 = arm_nn_read_q15x2_ia(&pB2);
                        sum = __SMLAD(inA1, inB1, sum);
                        sum2 = __SMLAD(inA1, inB2, sum2);
                        sum3 = __SMLAD(inA2, inB1, sum3);
                        sum4 = __SMLAD(inA2, inB2, sum4);
                        */
                        colCnt--;
                    }           /* while over colCnt */
                    colCnt = ch_im_in * dim_kernel * dim_kernel & 0x1;
                    while (colCnt)
                    {
                        int16_t     inA1 = *pA++;
                        int16_t     inB1 = *pB++;
                        int16_t     inA2 = *pA2++;
                        int16_t     inB2 = *pB2++;

                        sum += inA1 * inB1;
                        sum2 += inA1 * inB2;
                        sum3 += inA2 * inB1;
                        sum4 += inA2 * inB2;
                        colCnt--;
                    }           /* while over colCnt */
                    *pOut++ = (int16_t) __SSAT(sum >> out_shift, 16);
                    *pOut++ = (int16_t) __SSAT(sum3 >> out_shift, 16);
                    *pOut2++ = (int16_t) __SSAT(sum2 >> out_shift, 16);
                    *pOut2++ = (int16_t) __SSAT(sum4 >> out_shift, 16);

                    /* skip the row computed with A2 */
                    pA += ch_im_in * dim_kernel * dim_kernel;
                }               /* for over ch_im_out */

                pOut += ch_im_out;
                /* counter reset */
                pBuffer = im_buffer;
            }
        }
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    signed char in_row, in_col;

    if (ch_im_in % 2 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
       // printf("Dimension mismatch\n");
        return;
    }

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
                Im_out[i + (j * dim_im_out + k) * ch_im_out] = (int16_t) __SSAT((conv_out >> out_shift), 16);
            }
        }
    }
#endif
}

/**
 * @} end of NNConv group
 */
