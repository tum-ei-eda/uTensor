/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_nn_mat_mult_kernel_q7_q15.c
 * Description:  Matrix-multiplication function for convolution
 *
 * $Date:        17. January 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "riscv_nnfunctions.hpp"

  /**
   * @brief Matrix-multiplication function for convolution.
   *
   * @details Refer to header file for details.
   *
   */

int8_t     *riscv_nn_mat_mult_kernel_q7_q15(const int8_t * pA,
                                        const int16_t * pInBuffer,
                                        const uint16_t ch_im_out,
                                        const uint16_t numCol_A,
                                        const uint16_t bias_shift,
                                        const uint16_t out_shift,
                                        const int8_t * bias,
                                        int8_t * pOut)
{
    /* set up the second output pointers */
    int8_t     *pOut2 = pOut + ch_im_out;
    const int8_t *pBias = bias;

    uint16_t  rowCnt = ch_im_out >> 1;
    /* this loop over rows in A */
    while (rowCnt)
    {
        /* setup pointers for B */
        const int16_t *pB = pInBuffer;
        const int16_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const int8_t *pA2 = pA + numCol_A;

        /* init the sum with bias */
        int32_t     sum =  ((int32_t)(*pBias) << bias_shift);
        int32_t     sum2 = ((int32_t)(*pBias++) << bias_shift);
        int32_t     sum3 = ((int32_t)(*pBias) << bias_shift);
        int32_t     sum4 = ((int32_t)(*pBias++) << bias_shift);

        uint16_t  colCnt = numCol_A >> 2;
        /* accumulate over the vector */
        while (colCnt)
        {
            int32_t     inA11, inA12, inA21, inA22;
            /*

            int32_t     inB1 = arm_nn_read_q15x2_ia(&pB);
            int32_t     inB2 = arm_nn_read_q15x2_ia(&pB2);

            pA = read_and_pad(pA, &inA11, &inA12);
            pA2 = read_and_pad(pA2, &inA21, &inA22);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);
            sum3 = __SMLAD(inA21, inB1, sum3);
            sum4 = __SMLAD(inA21, inB2, sum4);

            inB1 = arm_nn_read_q15x2_ia(&pB);
            inB2 = arm_nn_read_q15x2_ia(&pB2);

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            sum3 = __SMLAD(inA22, inB1, sum3);
            sum4 = __SMLAD(inA22, inB2, sum4);
            */

            colCnt--;
        }                       /* while over colCnt */
        colCnt = numCol_A & 0x3;
        while (colCnt)
        {
            int8_t      inA1 = *pA++;
            int16_t     inB1 = *pB++;
            int8_t      inA2 = *pA2++;
            int16_t     inB2 = *pB2++;

            sum += inA1 * inB1;
            sum2 += inA1 * inB2;
            sum3 += inA2 * inB1;
            sum4 += inA2 * inB2;
            colCnt--;
        }                       /* while over colCnt */
        *pOut++ = (int8_t) __SSAT((sum >> out_shift), 8);
        *pOut++ = (int8_t) __SSAT((sum3 >> out_shift), 8);
        *pOut2++ = (int8_t) __SSAT((sum2 >> out_shift), 8);
        *pOut2++ = (int8_t) __SSAT((sum4 >> out_shift), 8);

        /* skip the row computed with A2 */
        pA += numCol_A;
        rowCnt--;
    }                           /* for over ch_im_out */

    /* compute left-over row if any */
    if (ch_im_out & 0x1)
    {
        /* setup pointers for B */
        const int16_t *pB = pInBuffer;
        const int16_t *pB2 = pB + numCol_A;

        /* load the bias */
        int32_t     sum = ((int32_t)(*pBias) << bias_shift);
        int32_t     sum2 = ((int32_t)(*pBias++) << bias_shift);

        uint16_t  colCnt = numCol_A >> 2;
        while (colCnt)
        {
            int32_t     inA11, inA12;
            /*
            int32_t     inB1 = arm_nn_read_q15x2_ia(&pB);
            int32_t     inB2 = arm_nn_read_q15x2_ia(&pB2);

            pA = read_and_pad(pA, &inA11, &inA12);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);

            inB1 = arm_nn_read_q15x2_ia(&pB);
            inB2 = arm_nn_read_q15x2_ia(&pB2);

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            */

            colCnt--;
        }
        colCnt = numCol_A & 0x3;
        while (colCnt)
        {
            int8_t      inA1 = *pA++;
            int16_t     inB1 = *pB++;
            int16_t     inB2 = *pB2++;

            sum += inA1 * inB1;
            sum2 += inA1 * inB2;
            colCnt--;
        }

        *pOut++ = (int8_t) __SSAT((sum >> out_shift), 8);
        *pOut2++ = (int8_t) __SSAT((sum2 >> out_shift), 8);
    }

    pOut += ch_im_out;

    /* return the new output pointer with offset */
    return pOut;
}
