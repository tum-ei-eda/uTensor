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
 * Title:        riscv_fully_connected_int8.c
 * Description:  int8 basic fully-connected layer function
 *
 * $Date:        17. January 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "riscv_nnfunctions.h"
#include <stdio.h>

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup FC
 * @{
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
   * @details
   *
   * <b>Buffer size:</b>
   *
   * vec_buffer size: dim_vec
   *
   * This basic function is designed to work with regular weight
   * matrix without interleaving.
   *
   */

void
riscv_fully_connected_int8(const int8_t * pV,
                       const int8_t * pM,
                       const uint16_t dim_vec,
                       const uint16_t num_of_rows,
                       const uint16_t bias_shift,
                       const uint16_t out_shift, const int8_t * bias, int8_t * pOut, int16_t * vec_buffer)
{
#if defined(USE_VEXT)
#warning "Using V Extension"
  printf("Using V-Extension\n");
    const int8_t *pB = pM;
    int8_t     *pO = pOut;
    const int8_t *pBias = bias;
    const int8_t    *pA = pV;
    uint16_t  rowCnt = num_of_rows >> 2;

    while (rowCnt)
    {

        int32_t     sum =  ((int32_t)(*pBias++) << bias_shift);
        int32_t     sum2 = ((int32_t)(*pBias++) << bias_shift);
        int32_t     sum3 = ((int32_t)(*pBias++) << bias_shift);
        int32_t     sum4 = ((int32_t)(*pBias++) << bias_shift);

        uint16_t  colCnt = dim_vec >> 2;

        /*
         * register needed:
         * loop counter: colCnt
         * accumulators: sum, sum2, sum3, sum4
         * pointers: pB, pA
         * weight data: inM11, inM12, inM13, inM14
         * activation data: inV
         */
        int tmp_vl = 0;
        while(colCnt)
        {
          asm volatile ("vsetvli %[tmp_vl], %[colCnt], e8 \n" // set register setting to 8-bit values and calculate tmp_vl=min(maxvl, colCnt)
                        "vlb.v v0, (%[pA]) \n " // load from input Matrix into v0
                        "vlb.v v1, (%[pB]) \n " // load from input Vector int v1
                        "vlw.v v2, (%[sum])\n " // load from sum into v2
                        "vmacc.vv v2, v1, v0 \n"  // v2 = v1 * v0 + v2
                        "vsw.v v2, (%[sum]) \n"   // save v2 into sum
                        "add %[pB], %[pB], %[tmp_vl] \n"  // adjust address to input vector
                        "vlb.v v3, (%[pB]) \n " 
                        "vlw.v v4, (%[sum2])\n "
                        "vmacc.vv v4, v3, v0 \n"
                        "vsw.v v4, (%[sum2]) \n"
                        "add %[pB], %[pB], %[tmp_vl] \n"
                        "vlb.v v3, (%[pB]) \n "
                        "vlw.v v4, (%[sum3])\n "
                        "vmacc.vv v4, v3, v0 \n"
                        "vsw.v v4, (%[sum3]) \n"
                        "add %[pB], %[pB], %[tmp_vl] \n"
                        "vlb.v v5, (%[pB]) \n "
                        "vlw.v v6, (%[sum4])\n "
                        "vmacc.vv v6, v5, v0 \n"
                        "vsw.v v6, (%[sum4]) \n"
                        "add %[pA], %[pA], %[tmp_vl] \n" // adjust address to input Matrix
                        :[sum] "+r"(sum), [sum2] "+r"(sum2), [sum3] "+r"(sum3), [sum4] "+r"(sum4),[pB] "+r"(pB), [pA] "+r"(pA)
                        :[colCnt] "r"(colCnt), [tmp_vl] "r"(tmp_vl));
          colCnt = colCnt - tmp_vl;
        }

        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            int16_t     inV = *pA++;
            int8_t      inM = *pB++;
            int8_t      inM2 = *pB++;
            int8_t      inM3 = *pB++;
            int8_t      inM4 = *pB++;

            sum += inV * inM;
            sum2 += inV * inM2;
            sum3 += inV * inM3;
            sum4 += inV * inM4;
            colCnt--;
        }                       /* while over colCnt */
        *pO++ = (int8_t) (__SSAT((sum >> out_shift), 8));
        *pO++ = (int8_t) (__SSAT((sum2 >> out_shift), 8));
        *pO++ = (int8_t) (__SSAT((sum3 >> out_shift), 8));
        *pO++ = (int8_t) (__SSAT((sum4 >> out_shift), 8));

        /* adjust the pointers and counters */
        rowCnt--;
    }

    /* left-over part of the rows */
    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        int32_t   sum = ((int32_t)(*pBias++) << bias_shift);
        uint16_t  colCnt = dim_vec >> 2;

        pA = pV;
        int tmp_vl = 0;
        while(colCnt)
        {
          asm volatile ("vsetvli %[tmp_vl], %[colCnt], e8 \n"     // set register setting to 8-bit values and calculate tmp_vl=min(maxvl, colCnt)
                        "vlb.v v0, (%[pA]) \n "                   // load from input Matrix into v0
                        "vlb.v v1, (%[pB]) \n "                   // load from input Vector int v1
                        "vlw.v v2, (%[sum])\n "                   // load from sum into v2
                        "vmacc.vv v2, v1, v0 \n"                  // v2 = v1 * v0 + v2
                        "vsw.v v2, (%[sum]) \n"                   // save v2 into sum
                        "add %[pB], %[pB], %[tmp_vl] \n"          // adjust address to input vector
                        "add %[pA], %[pA], %[tmp_vl] \n"          // adjust address to input Matrix
                        "vlb.v v3, (%[pA]) \n "                   // load from input Matrix into v0
                        "vlb.v v4, (%[pB]) \n "                   // load from input Vector int v1
                        "vlw.v v5, (%[sum])\n "                   // load from sum into v2
                        "vmacc.vv v5, v3, v4 \n"                  // v2 = v1 * v0 + v2
                        "vsw.v v5, (%[sum]) \n"                   // save v2 into sum
                        "add %[pB], %[pB], %[tmp_vl] \n"          // adjust address to input vector
                        "add %[pA], %[pA], %[tmp_vl] \n"          // adjust address to input Matrix
                        :[sum] "+r"(sum), [pB] "+r"(pB), [pA] "+r"(pA)
                        :[colCnt] "r"(colCnt), [tmp_vl] "r"(tmp_vl));
          colCnt = colCnt - tmp_vl;
        }
        /* left-over of the vector */
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            int16_t     inV = *pA++;
            int8_t      inM = *pB++;
            sum += inV * inM;
            colCnt--;
        }

        *pO++ = (int8_t) (__SSAT((sum >> out_shift), 8));

        rowCnt--;
    }

#else
  printf("Using without V-Extension\n");
    int       i, j;

    /* Run the following code as reference implementation in case there is no V-Extension provided */
    for (i = 0; i < num_of_rows; i++)
    {
        int       ip_out = ((int32_t)(bias[i]) << bias_shift);
        for (j = 0; j < dim_vec; j++)
        {
            ip_out += pV[j] * pM[i * dim_vec + j];
        }
        pOut[i] = (int8_t) __SSAT((ip_out >> out_shift), 8);
    }
#endif
}

/**
 * @} end of FC group
 */
