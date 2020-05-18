#ifndef _RISCV_VEXT_HPP_
#define _RISCV_VEXT_HPP_

#include <type_traits>

template <typename T>
static inline void vmacc(const T * Im_in, const T * wt, unsigned short colCnt, unsigned char * tmp_vl, int * pOut)
{
  unsigned char tmp = *tmp_vl;
  if constexpr(std::is_same<T, signed char>::value)
  {
    asm volatile ("vsetvli %[tmp], %[colCnt], e8 \n" // set register setting to 8-bit values and calculate tmp_vl=min(maxvl=2, colCnt)
                :[tmp] "=r" (tmp) 
                :[colCnt] "r"(colCnt));

  }
  else if constexpr(std::is_same<T, short>::value)
  {
    asm volatile ("vsetvli %[tmp], %[colCnt], e16 \n" // set register setting to 16-bit values and calculate tmp_vl=min(maxvl=2, colCnt)
                :[tmp] "=r" (tmp) 
                :[colCnt] "r"(colCnt));
  }
  else
  {
    asm volatile ("vsetvli %[tmp], %[colCnt], e32 \n" // set register setting to 32-bit values and calculate tmp_vl=min(maxvl=2, colCnt)
                :[tmp] "=r" (tmp) 
                :[colCnt] "r"(colCnt));
  }
  asm volatile ("vlw.v v1, (%[Im_in]) \n " // load from input Matrix into v0
                "vlw.v v2, (%[wt]) \n " // load from input Vector int v1
                "vlw.v v3, (%[pOut])\n " // load from sum into v2
                "vmacc.vv v3, v2, v1 \n"  // v2 = v1 * v0 + v2
                "vsw.v v3, (%[pOut]) \n"   // save v2 into sum
                :[pOut] "+r"(pOut)
                :[Im_in] "r"(Im_in), [wt] "r"(wt));
  *tmp_vl = tmp;

};

#endif
