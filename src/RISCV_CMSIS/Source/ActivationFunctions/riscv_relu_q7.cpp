/*
 * Copyright (C) 2010-2019 riscv Limited or its affiliates. All rights reserved.
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
 * Title:        riscv_relu_int8.c
 * Description:  int8 version of ReLU
 *
 * $Date:        August 2019
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
 * @addtogroup Acti
 * @{
 */

  /**
   * @brief int8 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   *
   * @details
   *
   * Optimized relu with QSUB instructions.
   *
   */

void riscv_relu_int8(int8_t *data, uint16_t size)
{
#if defined(USE_VEXT)
#warning "Using V-Extenxion"
  int cmp_val = 0;
  unsigned char tmp_vl = 0;
  unsigned short elemCnt = size & 0xFFFC;
  while(elemCnt)
  {
    vmax_vx<signed char>(data, (signed char *)cmp_val, elemCnt, &tmp_vl);
    elemCnt -= tmp_vl;
    data += tmp_vl; 
  }
  elemCnt = size & 0x3;
  while(elemCnt)
  {
    if(*data < 0)
    {
      *data = 0;
    }
    elemCnt--;
    data++;
  }
#else
    /* Run the following code as reference implementation for M cores without DSP extension */
    uint16_t i;
    for (i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
        //printf("data[%d]: %d\n", i, data[i]);
    }
#endif
}
  /**
   * @brief int8 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       ref_point   new refernce point after quantization
   *
   * @details
   *
   * Optimized relu with QSUB instructions.
   *
   */

void riscv_relu_int8_adj(int8_t *data, uint16_t size, int16_t ref_point)
{
    /* Run the following code as reference implementation for cores without DSP extension */

    uint16_t i;

    for (i = 0; i < size; i++)
    {
        if (data[i] < ref_point)
            data[i] = ref_point;
    }
}

/**
 * @} end of Acti group
 */
