#ifndef _RISCV_VEXT_HPP_
#define _RISCV_VEXT_HPP_

#include <type_traits>

template <typename T>
static inline void vsetvli(unsigned short colCnt, unsigned char * tmp_vl)
{
  unsigned char tmp = *tmp_vl;
  if constexpr(std::is_same<T, signed char>::value)
  {
    asm volatile ("vsetvli %[tmp], %[colCnt], e8 \n" // set register setting to 8-bit values and calculate tmp_vl=min(maxvl=4, colCnt)
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
    asm volatile ("vsetvli %[tmp], %[colCnt], e32 \n" // set register setting to 32-bit values and calculate tmp_vl=min(maxvl=1, colCnt)
                :[tmp] "=r" (tmp) 
                :[colCnt] "r"(colCnt));
  }
  *tmp_vl = tmp;
}

template <typename T>
static inline void vmacc(const T * Im_in, const T * wt, unsigned short colCnt, unsigned char * tmp_vl, int * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[Im_in]) \n " // load from input 1 into v1
                "vlw.v v2, (%[wt]) \n " // load from input 2 int v2
                "vlw.v v3, (%[pOut])\n " // load from sum into v3
                "vmacc.vv v3, v2, v1 \n"  // v3[i] = v1[i] * v2[i] + v3[i]
                "vsw.v v3, (%[pOut]) \n"   // save v3[i] into pOut
                :[pOut] "+r"(pOut)
                :[Im_in] "r"(Im_in), [wt] "r"(wt));

};

template <typename T>
static inline void vadd_vx(const T * op1, const T * op2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[op1]) \n " // load from input Matrix into v0
                "vadd.vx v2, v1, %[op2] \n"  // v2[i] = v1[i] + op2
                "vsw.v v2, (%[pOut]) \n"   // save v2[i] into sum
                :[pOut] "+r"(pOut)
                :[op1] "r"(op1), [op2] "r"(op2));

};

template <typename T>
static inline void vadd_vv(const T * op1, const T * op2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[op1]) \n "  
                "vlw.v v2, (%[op2]) \n "  
                "vlw.v v3, (%[pOut])\n " 
                "vadd.vv v3, v2, v1 \n"  // v3[i] = v1[i] + v2[i]
                "vsw.v v3, (%[pOut]) \n"   // save v3[i] into *pOut
                :[pOut] "+r"(pOut)
                :[op1] "r"(op1), [op2] "r"(op2));

};

template <typename T>
static inline void vmul_vv(const T * op1, const T * op2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[op1]) \n "  
                "vlw.v v2, (%[op2]) \n "  
                "vlw.v v3, (%[pOut])\n " 
                "vmul.vv v3, v2, v1 \n"  // v3[i] = v1[i] * v2[i]
                "vsw.v v3, (%[pOut]) \n"   // save v3[i] into *pOut
                :[pOut] "+r"(pOut)
                :[op1] "r"(op1), [op2] "r"(op2));

};

template <typename T>
static inline void vsub_vv(const T * op1, const T * op2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[op1]) \n " 
                "vlw.v v2, (%[op2]) \n " 
                "vlw.v v3, (%[pOut])\n " 
                "vsub.vv v3, v1, v2 \n"   // v3[i] = v1[i] - v2[i]
                "vsw.v v3, (%[pOut]) \n"  // save v3[i] into *pOut
                :[pOut] "+r"(pOut)
                :[op1] "r"(op1), [op2] "r"(op2));

};

/*
template <typename T>
static inline void vsub_vx(const T * op1, const T * op2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[op1]) \n " // load from input Matrix into v0
                "vsub.vx v2, v1, %[op2] \n"  // v2 = v1 * v0 + v2
                "vsw.v v2, (%[pOut]) \n"   // save v2 into sum
                :[pOut] "+r"(pOut)
                :[op1] "r"(op1), [op2] "r"(op2));

};
*/
template <typename T>
static inline void vmax_vx(const T * pIn, const T * cmp_val, unsigned short colCnt, unsigned char * tmp_vl)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[pIn]) \n"
                "vmax.vx v2, v1, %[cmp_val] \n"
                "vsw.v v2, (%[pIn]) \n"
                :[pIn] "+r" (pIn)
                :[cmp_val]"r"(cmp_val));
}

template <typename T>
static inline void vmax_vv(const T * pIn_1, const T * pIn_2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[pIn_1]) \n"
                "vlw.v v2, (%[pIn_2]) \n"
                "vmax.vv v3, v1, v2 \n"
                "vsw.v v3, (%[pOut]) \n"
                :[pOut] "+r" (pOut)
                :[pIn_1] "r" (pIn_1), [pIn_2] "r" (pIn_2));
}

template <typename T>
static inline void vmin_vv(const T * pIn_1, const T * pIn_2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[pIn_1]) \n"
                "vlw.v v2, (%[pIn_2]) \n"
                "vmin.vv v3, v1, v2 \n"
                "vsw.v v3, (%[pOut]) \n"
                :[pOut] "+r" (pOut)
                :[pIn_1] "r" (pIn_1), [pIn_2] "r" (pIn_2));
}

template <typename T>
static inline void vsll_vv(const T * pIn_1, const T * pIn_2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  //vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[pIn_1]) \n"
                "vlw.v v2, (%[pIn_2]) \n"
                "vsll.vv v3, v1, v2 \n" // v3[i] = v1[i] << v2[i]
                "vsw.v v3, (%[pOut]) \n"
                :[pOut] "+r" (pOut)
                :[pIn_1] "r" (pIn_1), [pIn_2] "r" (pIn_2));
}

template <typename T>
static inline void vsrl_vv(const T * pIn_1, const T * pIn_2, unsigned short colCnt, unsigned char * tmp_vl, T * pOut)
{
  vsetvli<T>(colCnt, tmp_vl);
  asm volatile ("vlw.v v1, (%[pIn_1]) \n"
                "vlw.v v2, (%[pIn_2]) \n"
                "vsrl.vv v3, v1, v2 \n" // v3[i] = v1[i] >> v2[i]
                "vsw.v v3, (%[pOut]) \n"
                :[pOut] "+r" (pOut)
                :[pIn_1] "r" (pIn_1), [pIn_2] "r" (pIn_2));
}


#endif
