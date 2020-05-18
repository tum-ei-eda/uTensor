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
 * Title:        riscv_fully_connected_int16.c
 * Description:  int16 basic fully-connected layer function
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
 * @addtogroup FC
 * @{
 */

  /**
   * @brief int16 opt fully-connected layer function
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
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * vec_buffer size: 0
   *
   */

void
riscv_fully_connected_int16(const int16_t * pV,
                        const int16_t * pM,
                        const uint16_t dim_vec,
                        const uint16_t num_of_rows,
                        const uint16_t bias_shift,
                        const uint16_t out_shift,
                        const int16_t * bias,
                        int16_t * pOut,
                        int16_t * vec_buffer)
{
#if defined(USE_VEXT)
#warning "Using V Extension"
  (void)vec_buffer;
  uint16_t rowCnt = num_of_rows;
  uint16_t colCnt = dim_vec & 0xFFFE;
  const int16_t * pA = pV;
  const int16_t * pB = pM;
  const int16_t * pBias = bias;
  int tmp_vl = 0;
  while(rowCnt)
  {
    colCnt = dim_vec & 0xFFFE;
    while(colCnt)
    {
      asm volatile ("vsetvli %[tmp_vl], %[colCnt], e16 \n" // set register setting to 16-bit values and calculate tmp_vl=min(maxvl=2, colCnt)
                   "vlw.v v1, (%[pA]) \n " // load from input Matrix into v0
                   "vlw.v v2, (%[pB]) \n " // load from input Vector int v1
                   "vlw.v v3, (%[pOut])\n " // load from sum into v2
                   //"vlw.v v3, (%[sum])\n " // load from sum into v2
                   "vmacc.vv v3, v2, v1 \n"  // v2 = v1 * v0 + v2
                   "vsw.v v3, (%[pOut]) \n"   // save v2 into sum
                   //"vsw.v v3, (%[sum]) \n"   // save v2 into sum
                   :[tmp_vl] "=r" (tmp_vl), [pOut] "+r"(pOut)
                   //:[sum] "+r"(sum), [tmp_vl] "=r" (tmp_vl)
                   :[colCnt] "r"(colCnt), [pA] "r"(pA), [pB] "r"(pB));
      colCnt -= tmp_vl;
      pA += tmp_vl;
      pB += tmp_vl;
    }

    if(dim_vec & 0x1)
    {
      *pOut +=  (*pA) * (*pB++);
      //sum +=  (*pA) * (*pB++);
    }
    *pOut++ += 0xFFFF & ((*pBias++ << bias_shift) + NN_ROUND(out_shift));
    pA = pV;
    rowCnt--;
  }
#else
  /*
   * Alternative implementation in case there is no V-Extension
   * */

 (void)vec_buffer;
 int       i, j;
 /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
 for (i = 0; i < num_of_rows; i++)
 {
   int       ip_out = ((int32_t)(bias[i]) << bias_shift);
   for (j = 0; j < dim_vec; j++)
   {
       ip_out += pV[j] * pM[i * dim_vec + j];
   }
   pOut[i] = (int16_t) __SSAT((ip_out >> out_shift), 16);
 }
#endif 
}

/**
 * @} end of FC group
 */
