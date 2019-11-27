/*
 * Copyright (C) 2010-2019 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_relu_int16.c
 * Description:  int16 version of ReLU
 *
 *
 * $Date:        August 2019
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
 * @addtogroup Acti
 * @{
 */

/**
   * @brief int16 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   *
   * @details
   *
   * Optimized relu with QSUB instructions.
   *
   */

void riscv_relu_int16(int16_t *data, uint16_t size)
{
    /* Run the following code as reference implementation for M cores without DSP extension */
    uint16_t i;
    for (i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
        //printf("data[%d]: %d\n", i, data[i]);
    }
}

/**
 * @} end of Acti group
 */
