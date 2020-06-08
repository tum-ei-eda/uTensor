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

#include "riscv_nnfunctions.hpp"

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
riscv_fully_connected_int8( const int8_t * pV,
                            const int8_t * pM,
                            const uint16_t dim_vec,
                            const uint16_t num_of_rows,
                            const uint16_t bias_shift,
                            const uint16_t out_shift, 
                            const int8_t * bias, 
                            int8_t * pOut, 
                            int16_t * vec_buffer)
{
#if defined(USE_VEXT)
#warning "Using V Extension"
  (void)vec_buffer;
  uint16_t rowCnt = num_of_rows;
  uint16_t colCnt = dim_vec & 0xFFFC;
  const int8_t * pA = pV;
  const int8_t * pB = pM;
  const int8_t * pBias = bias;
  unsigned char tmp_vl = 0;
  int sum = 0;
  while(rowCnt)
  {
    colCnt = dim_vec & 0xFFFC;
    sum =  ((int)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
    while(colCnt)
    {
      vmacc<signed char>(pA, pB, colCnt, &tmp_vl, &sum);
      colCnt -= tmp_vl;
      pA += tmp_vl;
      pB += tmp_vl;
    }

    colCnt = dim_vec & 0x3;
    while(colCnt)
    {
      colCnt--;
      sum +=  (*pA) * (*pB++);
    }
    *pOut++ =  (signed char) (__SSAT((sum >> out_shift), 8));
    pA = pV;
    rowCnt--;
  }

#else
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

