/*
 * Copyright (C) 2020 riscv Limited or its affiliates. All rights reserved.
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
 * Title:        riscv_nn_vec_mat_mult_t_s8
 * Description:  s8 vector by matrix (transposed) multiplication
 *
 * $Date:        March 17, 2020
 * $Revision:    V.1.0.1
 *
 * Target Processor:  Cortex-M
 *
 * -------------------------------------------------------------------- */

#include "riscv_nnfunctions.hpp"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup NNBasicMath
 * @{
 */

/*
   * s8 vector by matrix (transposed) multiplication
   *
   * Refer header file for details.
   *
   */
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
                                    const int32_t activation_max)
{
  //printf("lhs_offset     %d \n "
  //       "rhs_offset     %d \n "
  //       "dst_offset     %d \n "
  //       "dst_multiplier %d \n "
  //       "dst_shift      %d \n "
  //       "rhs_cols       %d \n "
  //       "rhs_rows       %d \n "
  //       "activation_min %d \n "
  //       "activation_max %d \n ", lhs_offset, rhs_offset, dst_offset, dst_multiplier, dst_shift, rhs_cols, rhs_rows, activation_min, activation_max );
#if defined(USE_VEXT)
  uint16_t rowCnt = rhs_rows;
  uint16_t colCnt = rhs_cols & 0xFFFC;
  int32_t A = 0;
  int32_t B = 0;
  int8_t * pA = (int8_t *)&A;
  int8_t * pB = (int8_t *)&B;
  const int32_t * pBias = bias;
  unsigned char tmp_vl = 0;
  int sum = 0;
  int32_t input_1_offsetV  = 0;
  int32_t input_2_offsetV  = 0;
  int8_t * pinput_1_offsetV  = (int8_t *) &input_1_offsetV;
  int8_t * pinput_2_offsetV  = (int8_t *) &input_2_offsetV;
  int32_t chars_in_int = sizeof(int32_t) / sizeof(char);

  for(int i = 0; i < chars_in_int; i++)
  {
    pinput_1_offsetV[i] = (int8_t)lhs_offset;
    pinput_2_offsetV[i] = (int8_t)rhs_offset;
  }

  while(rowCnt)
  {
    const int8_t *lhs_ptr = &lhs[0];
    const int8_t *rhs_ptr = &rhs[0];
    colCnt = rhs_cols & 0xFFFC;
    sum = *pBias++;
    while(colCnt)
    {
      vadd_vv<int8_t>(lhs_ptr, pinput_1_offsetV, colCnt, &tmp_vl, pA); //works
      vadd_vv<int8_t>(rhs_ptr, pinput_2_offsetV, colCnt, &tmp_vl, pB); //works
      sum += pA[0] * pB[0];
      sum += pA[1] * pB[1];
      sum += pA[2] * pB[2];
      sum += pA[3] * pB[3];
      colCnt -=  tmp_vl;
      rhs_ptr += tmp_vl;
      lhs_ptr += tmp_vl;
    }

    colCnt = rhs_cols & 0x3;
    while(colCnt)
    {
      int32_t rhs_value0 = rhs_ptr[0] + rhs_offset;
      int32_t lhs_value  = lhs_ptr[0] + lhs_offset;
      sum += lhs_value * rhs_value0;
      --colCnt;
      ++rhs_ptr;
      ++lhs_ptr;
    }
    sum = riscv_nn_requantize(sum, dst_multiplier, dst_shift);
    sum += dst_offset;
    // Clamp the result
    sum = MAX(sum, activation_min);
    sum = MIN(sum, activation_max);
    *dst++ =  static_cast<int8_t>(sum);
    rowCnt--;
  }

#else

    for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        const int8_t *rhs_ptr = &rhs[0];

        int32_t res00 = *bias++;
        int32_t res01 = *bias++;

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            int32_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            int32_t rhs_value1 = rhs_ptr[rhs_cols] + rhs_offset;
            int32_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = riscv_nn_requantize(res00, dst_multiplier, dst_shift);
        res01 = riscv_nn_requantize(res01, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);

        *dst++ = (int8_t)res00;
        *dst++ = (int8_t)res01;

        rhs += 2 * rhs_cols;
    }

    if (rhs_rows % 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        const int8_t *rhs_ptr = &rhs[0];

        int32_t res00 = *bias++;

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            int32_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            int32_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = riscv_nn_requantize(res00, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);

        *dst = (int8_t)res00;
    }
#endif

    return RISCV_MATH_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
