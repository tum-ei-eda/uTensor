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
 * Title:        riscv_softmax_int8.c
 * Description:  int8 softmax function
 *
 * $Date:        20. February 2018
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
 * @addtogroup Softmax
 * @{
 */

  /**
   * @brief int8 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimention
   * @param[out]      p_out       pointer to output vector
   *
   * @details
   *
   *  Here, instead of typical natural logarithm e based softmax, we use
   *  2-based softmax here, i.e.,:
   *
   *  y_i = 2^(x_i) / sum(2^x_j)
   *
   *  The relative output will be different here.
   *  But mathematically, the gradient will be the same
   *  with a log(2) scaling factor.
   *
   *  If we compare the position of the max value in output of this
   *  function with a reference float32 softmax (and thus using exp)
   *  we see that the position of the max value is sometimes different.
   *
   *  If we do statistics on lot of input vectors we can compute
   *  an average error rate in percent. It is the percent of time
   *  that the max will be at a position different from the one
   *  computed with a reference float32 implementation.
   *
   *  This average error rate is dependent on the vector size.
   *  We have:
   *
   *  Average error rate in percent = -0.555548 + 0.246918 dim_vec
   *  Variance of the error rate = -0.0112281 + 0.0382476 dim_vec
   *
   *
   */

static const int int8BITS = 8;
static const int LOG2int8BITS = 3;

void riscv_softmax_int8(const int8_t * vec_in, const uint16_t dim_vec, int8_t * p_out )
{
#if defined(USE_VEXT)
#warning "Using V-Extension"
    int sum;
    int16_t   i;
    uint8_t   shift;
    int16_t     base;
    uint16_t blkCnt;
    uint8_t tmp_vl;
    int32_t baseV;
    const int32_t pad=0x0d0d0d0d;
    const int onesArr = 0x01010101;
    const int8_t *pIn=vec_in;
    int8_t *pOut = p_out;

    base = -128;


    /* We first search for the maximum */

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    /*
     * So the base is set to max-8, meaning
     * that we ignore really small values.
     * anyway, they will be 0 after shrinking to int8_t.
     */
    base = base - int8BITS;
    baseV = ((base & 0x0FF) << 24) | ((base & 0x0FF) << 16) | ((base & 0x0FF) << 8) | ((base & 0x0FF));
    sum = 0;


    blkCnt = dim_vec & 0xFFFC;

    while(blkCnt)
    {
      
      /*
       * shift = (uint8_t)__USAT(vec_in[i] - base, LOG2Q7BITS);
       * sum += 0x1 << shift;
       */
      vsub_vv<int8_t>(pIn, (int8_t *) &baseV, blkCnt, &tmp_vl, pOut); 
      __USAT8(pOut, LOG2int8BITS); 
      vsll_vv<int8_t>((int8_t *)&onesArr, pOut, blkCnt, &tmp_vl, pOut);
      vmacc<int8_t>(pOut, (int8_t *)&onesArr, blkCnt, &tmp_vl, &sum); 
      blkCnt -= tmp_vl;
      pIn += tmp_vl;
      pOut += tmp_vl;
    }

    blkCnt = dim_vec & 0x3;

    while(blkCnt)
    {
       shift = (uint8_t)__USAT(*pIn++ - base, LOG2int8BITS);
       sum += 0x1 << shift;
       blkCnt--;
    }
  
    /* This is effectively (0x1 << 20) / sum */
    int output_base = (1 << 20) / sum;

    pIn=vec_in;
    pOut=p_out;

    blkCnt = dim_vec & 0xFFFC;
    while(blkCnt)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
      vsub_vv<int8_t>((int8_t *) &pad, pIn, blkCnt, &tmp_vl, pOut);
      vadd_vv<int8_t>((const int8_t *) pOut, (int8_t *)&baseV, blkCnt, &tmp_vl, pOut);
      __USAT8(pOut, 5); 
      *pOut++ = (int8_t) __SSAT((output_base >> *pOut), 8);                                                                                                                 *pOut++ = (int8_t) __SSAT((output_base >> *pOut), 8);
      *pOut++ = (int8_t) __SSAT((output_base >> *pOut), 8);
      *pOut++ = (int8_t) __SSAT((output_base >> *pOut), 8);
      __SSAT8(pOut, 8); 
      blkCnt -= tmp_vl;
      pIn += tmp_vl;
    }

    blkCnt = dim_vec & 0x3;
    while(blkCnt)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        shift = (uint8_t)__USAT(13 + base - *pIn++, 5);
        *pOut++ = (int8_t) __SSAT((output_base >> shift), 8);
        blkCnt --;
    }

#else
    int32_t     sum;
    int16_t   i;
    uint8_t   shift;
    int16_t     base;

    base = -128;

    /* We first search for the maximum */

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }


    /*
     * So the base is set to max-8, meaning
     * that we ignore really small values.
     * anyway, they will be 0 after shrinking to int8_t.
     */
    base = base - int8BITS;

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        shift = (uint8_t)__USAT(vec_in[i] - base, LOG2int8BITS);
        sum += 0x1 << shift;
    }

    /* This is effectively (0x1 << 20) / sum */
    int output_base = (1 << 20) / sum;


    for (i = 0; i < dim_vec; i++)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        shift = (uint8_t)__USAT(13 + base - vec_in[i], 5);
        p_out[i] = (int8_t) __SSAT((output_base >> shift), 8);

    }
#endif
}
/**
 * @} end of Softmax group
 */
