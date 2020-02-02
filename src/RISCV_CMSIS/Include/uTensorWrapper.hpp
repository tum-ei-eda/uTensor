#ifndef _UTENSORWRAPPER_H
#define _UTENSORWRAPPER_H

#include "riscv_nnfunctions.hpp"
#include "../../uTensor/util/quantization_utils.hpp"
#include "../../uTensor/core/tensor.hpp"
#include "../../uTensor/core/uTensorBase.hpp"
#include "../../uTensor/util/uTensor_util.hpp"
#include <type_traits>

void SoftmaxRiscv(S_TENSOR input, S_TENSOR output)
{
    if(input->getDim() != 1)
    {
      for(int i = 0; i < input->getDim() - 1; i++)
      {
        if(input->getShape().at(i) != 1)
        {
          printf("Softmax is supported only for flatten Tensor");
          exit(1);
        }
      }
    }

    if (output && output->getSize() == 0)
    {
      output->resize(input->getShape());
    }

    const int16_t * in = (const int16_t *)input->read<int>(0,0);
    int16_t * out = (int16_t *)output->write<int>(0, 0);

    const uint16_t size = (const uint16_t)output->getSize();
    riscv_softmax_int16(in, size, out);
};

template<class T1, class TOut>
void ReluRiscv(S_TENSOR input, S_TENSOR output)
{
    const T1 * in = input->read<T1>(0,0);
    if (output && output->getSize() == 0)
    {
      output->resize(input->getShape());
    }

    TOut * out = output->write<TOut>(0, 0);

    const uint16_t size = (const uint16_t)output->getSize();
    for(uint16_t i = 0; i < size; i++)
    {
      out[i] = in[i];
    }

    if constexpr (std::is_same<int8_t, T1>::value) // if the input tensor has data of int8_t type
    { riscv_relu_int8((int8_t *) out, size); }
    else
    { riscv_relu_int16((int16_t *)out, size); }

};

template <class TIn, class T2, class TOut>
void QuantizedReluRiscv(S_TENSOR input, S_TENSOR in_min, S_TENSOR in_max,
          S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max)
{
    const float input_min = in_min->read<T2>(0, 0)[0];
    const float input_max = in_max->read<T2>(0, 0)[0];
    const TIn* in = input->read<TIn>(0, 0);

    if (output && output->getSize() == 0)
    {
      output->resize(input->getShape());
    }

    TOut * out = output->write<TOut>(0, 0);

    const uint16_t size = (const uint16_t)output->getSize();
    for(uint16_t i = 0; i < size; i++)
    { 
      out[i] = in[i];
    }

    const TOut min_as_quantized = FloatToQuantized<TOut>(0.0f, input_min, input_max);
    if(min_as_quantized == 0)
    {
      riscv_relu_int8((int8_t *)out, size);
    }
    else
    {
      riscv_relu_int8_adj((int8_t *)out, size, (int16_t) min_as_quantized);
    }

    T2* v_out_min = out_min->write<T2>(0, 0);
    *v_out_min = input_min;
    T2* v_out_max = out_max->write<T2>(0, 0);
    *v_out_max = input_max;

}; 

template<typename T>
void SpatialMaxPoolingRiscv(S_TENSOR input, S_TENSOR output, 
                       int window_rows, int window_cols, 
                       int row_stride, int col_stride, 
                       Padding _padding, T pad_value = 0)
{
    if constexpr (!std::is_same<T, int8_t>::value) 
    {
      //printf("RISCV MaxPooling can process only int8_t or char");
      exit(1);
    }

    TensorShape in_shape = input->getShape();
    uint32_t in_rows = in_shape[1];
    uint32_t in_cols = in_shape[2];
    uint32_t in_channels = in_shape[3];

    size_t out_rows, out_cols;
    int pad_top, pad_left;
    if (_padding == VALID)
    {
      out_rows = ((size_t) ceil(((float)(in_rows - window_rows) + 1) / ((float)row_stride)));
      out_cols = ((size_t) ceil(((float)(in_cols - window_cols) + 1) / ((float)col_stride)));
      // no padding for VALID
      pad_top = 0;
      pad_left = 0;
    } 
    else 
    { 
      // SAME padding
      out_rows = ((size_t) ceil(((float)in_rows) / ((float) row_stride)));
      out_cols = ((size_t) ceil(((float)in_cols) / ((float) col_stride)));
      if (in_rows % row_stride == 0) 
      {
        pad_top = std::max(window_rows - row_stride, 0) / 2;
      } 
      else 
      {
        pad_top = std::max(window_rows - (((int) in_rows) % row_stride), 0) / 2;
      }
      
      if (in_cols % col_stride == 0) 
      {
        pad_left = std::max(window_cols - col_stride, 0) / 2;
      } 
      else 
      {
        pad_left = std::max(window_cols - (((int) in_cols) % col_stride), 0) / 2;
      }
    }

    TensorShape out_shape;
    out_shape.clear();
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
    out_shape.push_back(in_channels);
    output->resize(out_shape);

    int8_t *       Im_in      = (int8_t *)       input->read<T>(0, 0);
    const uint16_t dim_im_in  = (const uint16_t) input->getShape()[1]; 
    const uint16_t ch_im_in   = (const uint16_t) input->getShape()[3];
    const uint16_t dim_kernel = (const uint16_t) window_rows;
    const uint16_t padding    = (const uint16_t) pad_top;
    const uint16_t stride     = (const uint16_t) row_stride;
    const uint16_t dim_im_out = (const uint16_t) output->getShape()[0];
    int8_t *       bufferA    = nullptr; 
    int8_t *       Im_out     = (int8_t *)       output->write<T>(0, 0);

    riscv_maxpool_int8_HWC( Im_in,
                            dim_im_in,
                            ch_im_in,
                            dim_kernel,
                            padding,
                            stride,
                            dim_im_out,
                            bufferA,
                            Im_out);

}; 

template<typename T>
void QntMaxPoolingRiscv(S_TENSOR input, S_TENSOR output, 
                       int window_rows, int window_cols, 
                       int row_stride, int col_stride, 
                       Padding _padding, T pad_value = 0)
{
    TensorShape in_shape = input->getShape();
    uint32_t in_rows = in_shape[1];
    uint32_t in_cols = in_shape[2];
    uint32_t in_channels = in_shape[3];

    size_t out_rows, out_cols;
    int pad_top, pad_left;
    if (_padding == VALID)
    {
      out_rows = ((size_t) ceil(((float)(in_rows - window_rows) + 1) / ((float)row_stride)));
      out_cols = ((size_t) ceil(((float)(in_cols - window_cols) + 1) / ((float)col_stride)));
      // no padding for VALID
      pad_top = 0;
      pad_left = 0;
    } 
    else 
    { 
      // SAME padding
      out_rows = ((size_t) ceil(((float)in_rows) / ((float) row_stride)));
      out_cols = ((size_t) ceil(((float)in_cols) / ((float) col_stride)));
      if (in_rows % row_stride == 0) 
      {
        pad_top = std::max(window_rows - row_stride, 0) / 2;
      } 
      else 
      {
        pad_top = std::max(window_rows - (((int) in_rows) % row_stride), 0) / 2;
      }
      
      if (in_cols % col_stride == 0) 
      {
        pad_left = std::max(window_cols - col_stride, 0) / 2;
      } 
      else 
      {
        pad_left = std::max(window_cols - (((int) in_cols) % col_stride), 0) / 2;
      }
    }

    TensorShape out_shape;
    out_shape.clear();
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
    out_shape.push_back(in_channels);
    output->resize(out_shape);

    int8_t *       Im_in      = (int8_t *)       input->read<T>(0, 0);
    const uint16_t dim_im_in  = (const uint16_t) input->getShape()[1]; 
    const uint16_t ch_im_in   = (const uint16_t) input->getShape()[3];
    const uint16_t dim_kernel = (const uint16_t) window_rows;
    const uint16_t padding    = (const uint16_t) pad_top;
    const uint16_t stride     = (const uint16_t) row_stride;
    const uint16_t dim_im_out = (const uint16_t) output->getShape()[0];
    int8_t *       bufferA    = nullptr; 
    int8_t *       Im_out     = (int8_t *)       output->write<T>(0, 0);

    riscv_maxpool_int8_HWC( Im_in,
                        dim_im_in,
                        ch_im_in,
                        dim_kernel,
                        padding,
                        stride,
                        dim_im_out,
                        bufferA,
                        Im_out);
};

template <class T1, class T2, class Toutput>
void ConvRiscv(S_TENSOR input, S_TENSOR filter, S_TENSOR output,
                   std::vector<int32_t> strides_, Padding padding_)
{
    const int32_t batch       = input->getShape()[0];
    const int32_t input_rows  = input->getShape()[1];
    const int32_t input_cols  = input->getShape()[2];
    const int32_t in_depth    = input->getShape()[3];

    const int32_t filter_rows = filter->getShape()[0];
    const int32_t filter_cols = filter->getShape()[1];
    const int32_t out_depth   = filter->getShape()[3];

    std::vector<int32_t> strides = strides_;
    const int stride_rows     = strides_[1];
    const int stride_cols     = strides_[2];

    int32_t out_rows, out_cols;
    if (padding_ == VALID)
    {
      out_rows = (input_rows - filter_rows) / stride_rows + 1;
      out_cols = (input_cols - filter_cols) / stride_cols + 1;
    } else {
      // SAME
      out_rows = input_rows;
      out_cols = input_cols;
    }
    //TensorShape out_shape({batch, out_rows, out_cols, out_depth});
    TensorShape c_shape;
    c_shape.push_back(batch);
    c_shape.push_back(out_rows);
    c_shape.push_back(out_cols);
    c_shape.push_back(out_depth);
    output->resize(c_shape);
     /**
    * @brief Basic int16 convolution function
    * @param[in]       Im_in       pointer to input tensor
    * @param[in]       dim_im_in   input tensor dimention
    * @param[in]       ch_im_in    number of input tensor channels
    * @param[in]       wt          pointer to kernel weights
    * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
    * @param[in]       dim_kernel  filter kernel size
    * @param[in]       padding     padding sizes
    * @param[in]       stride      convolution stride
    * @param[in]       bias        pointer to bias
    * @param[in]       bias_shift  amount of left-shift for bias
    * @param[in]       out_shift   amount of right-shift for output
    * @param[in,out]   Im_out      pointer to output tensor
    * @param[in]       dim_im_out  output tensor dimension
    * @param[in,out]   bufferA     pointer to buffer space for input
    * @param[in,out]   bufferB     pointer to buffer space for output
    * @return                      void 
    *
    * @details
    *
    * <b>Buffer size:</b>
    *
    * bufferA size: ch_im_in*dim_kernel*dim_kernel
    *
    * bufferB size: 0
    *
    * This basic version is designed to work for any input tensor and weight
    * dimension.
    */
    const T1 *      InputImage      = input->read<T1>(0, 0);
    const uint16_t  DimInputImage   = input->getShape()[1]; 
    const uint16_t  ChInputImage    = input->getShape()[3];
    const T2 *      InWeights       = filter->read<T2>(0, 0);
    const uint16_t  ChOutputImage   = filter->getShape()[3];
    const uint16_t  DimFilterKernel = filter->getShape()[1];
    const uint16_t  Padding         = 0;
    const uint16_t  Stride          = filter->getShape()[1];
    const uint16_t  BiasShift       = 0;
    const uint16_t  OutShift        = 0;
    Toutput *       OutputImage     = output->write<Toutput>(0, 0);
    const uint16_t  DimOutputImage  = output->getShape()[1];
    int16_t *       BufferA         = nullptr;
    int8_t *        BufferB         = nullptr; 

    if constexpr (std::is_same<T1, int8_t>::value)
    {
      int8_t InputBias[input_rows];
      for(int i = 0; i < input_rows; i++) { InputBias[i] = 0;}
      riscv_convolve_HWC_int8_basic((const int8_t *)InputImage, DimInputImage, ChInputImage, (const int8_t *)InWeights, ChOutputImage, DimFilterKernel,
                                Padding, Stride,(const int8_t *)InputBias, BiasShift, OutShift, (int8_t *) OutputImage, DimOutputImage, BufferA, BufferB);

    }
    else
    {
      int16_t InputBias[input_rows];
      for(int i = 0; i < input_rows; i++) { InputBias[i] = 0;}
      riscv_convolve_HWC_int16_basic((const int16_t *)InputImage, DimInputImage, ChInputImage, (const int16_t *)InWeights, ChOutputImage, DimFilterKernel,
                                Padding, Stride,(const int16_t *)InputBias, BiasShift, OutShift, (int16_t *) OutputImage, DimOutputImage, BufferA, BufferB);
    }

}; 

template <class T1, class T2, class Toutput>
void MatMul2Riscv(S_TENSOR A, S_TENSOR B, S_TENSOR C,
                     bool transpose_a = false,
                     bool transpose_b = false)
{
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

    if(C->getSize() == 0) 
    {
      TensorShape c_shape;
      c_shape.push_back((A->getShape())[0]); // dimension of input Vector 
      c_shape.push_back((B->getShape())[1]); // dimension of input Matrix
      C->resize(c_shape);
    }

    const uint16_t  NumOfRowsInWeightMatrix = (const uint16_t) C->getShape()[1];
    const uint16_t  NumForBiasShift         = 0;
    const uint16_t  NumOfRShiftForOutput    = 0;
    const uint16_t  DimOfIputVector         = (const uint16_t) C->getShape()[0];
    const T1 *      InputVector             = A->read<T1>(0, 0);
    const T2 *      InputWeightMatrix       = B->read<T2>(0, 0);
    Toutput *       OutputVector            = C->write<Toutput>(0,0); 

    if constexpr (std::is_same<T1, int8_t>::value)
    {
      int8_t InputBias[NumOfRowsInWeightMatrix];
      for(int i = 0; i < NumOfRowsInWeightMatrix; i++) { InputBias[i] = 0;}
      riscv_fully_connected_int8((const int8_t *)InputVector,(const int8_t *)InputWeightMatrix, 
                                DimOfIputVector, NumOfRowsInWeightMatrix, NumForBiasShift, NumOfRShiftForOutput,
                                (const int8_t *) InputBias, (int8_t *)OutputVector, nullptr);
    }
    else
    {
      int16_t InputBias[NumOfRowsInWeightMatrix];
      for(int i = 0; i < NumOfRowsInWeightMatrix; i++) { InputBias[i] = 0;}
      riscv_fully_connected_int16((const int16_t *)InputVector,(const int16_t *)InputWeightMatrix, 
                                DimOfIputVector, NumOfRowsInWeightMatrix, NumForBiasShift, NumOfRShiftForOutput,
                                (const int16_t *) InputBias, (int16_t *)OutputVector, nullptr);
    }
}; 

template <class T1, class T2, class Toutput>
void QuantizedConvRiscv(S_TENSOR input, S_TENSOR filter, S_TENSOR output,
                   S_TENSOR mina, S_TENSOR maxa, 
                   S_TENSOR minb, S_TENSOR maxb, 
                   S_TENSOR outmin, S_TENSOR outmax,
                   std::vector<int32_t> strides_, Padding padding_)
{

    const int32_t batch       = input->getShape()[0];
    const int32_t input_rows  = input->getShape()[1];
    const int32_t input_cols  = input->getShape()[2];
    const int32_t in_depth    = input->getShape()[3];

    const int32_t filter_rows = filter->getShape()[0];
    const int32_t filter_cols = filter->getShape()[1];
    const int32_t out_depth   = filter->getShape()[3];

    std::vector<int32_t> strides = strides_;
    const int stride_rows     = strides_[1];
    const int stride_cols     = strides_[2];

    int32_t out_rows, out_cols;
    if (padding_ == VALID)
    {
      out_rows = (input_rows - filter_rows) / stride_rows + 1;
      out_cols = (input_cols - filter_cols) / stride_cols + 1;
    } else {
      // SAME
      out_rows = input_rows;
      out_cols = input_cols;
    }
    //TensorShape out_shape({batch, out_rows, out_cols, out_depth});
    TensorShape c_shape;
    c_shape.push_back(batch);
    c_shape.push_back(out_rows);
    c_shape.push_back(out_cols);
    c_shape.push_back(out_depth);
    output->resize(c_shape);
     /**
    * @brief Basic int16 convolution function
    * @param[in]       Im_in       pointer to input tensor
    * @param[in]       dim_im_in   input tensor dimention
    * @param[in]       ch_im_in    number of input tensor channels
    * @param[in]       wt          pointer to kernel weights
    * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
    * @param[in]       dim_kernel  filter kernel size
    * @param[in]       padding     padding sizes
    * @param[in]       stride      convolution stride
    * @param[in]       bias        pointer to bias
    * @param[in]       bias_shift  amount of left-shift for bias
    * @param[in]       out_shift   amount of right-shift for output
    * @param[in,out]   Im_out      pointer to output tensor
    * @param[in]       dim_im_out  output tensor dimension
    * @param[in,out]   bufferA     pointer to buffer space for input
    * @param[in,out]   bufferB     pointer to buffer space for output
    * @return     The function returns <code>riscv_MATH_SUCCESS</code>
    *
    * @details
    *
    * <b>Buffer size:</b>
    *
    * bufferA size: ch_im_in*dim_kernel*dim_kernel
    *
    * bufferB size: 0
    *
    * This basic version is designed to work for any input tensor and weight
    * dimension.
    */
    const T1 *      InputImage      = input->read<T1>(0, 0);
    const uint16_t  DimInputImage   = input->getShape()[1]; 
    const uint16_t  ChInputImage    = input->getShape()[3];
    const T2 *      InWeights       = filter->read<T2>(0, 0);
    const uint16_t  ChOutputImage   = filter->getShape()[3];
    const uint16_t  DimFilterKernel = filter->getShape()[1];
    const uint16_t  Padding         = 0;
    const uint16_t  Stride          = filter->getShape()[1];
    uint16_t        Bias[]          = {0}; 
    const uint16_t  BiasShift       = 0;
    const uint16_t  OutShift        = 0;
    Toutput *       OutputImage     = output->write<Toutput>(0, 0);
    const uint16_t  DimOutputImage  = output->getShape()[1];
    int16_t *       BufferA         = nullptr;
    int8_t *        BufferB         = nullptr; 

    riscv_convolve_HWC_int8_basic((const int8_t *)InputImage, DimInputImage, ChInputImage, (const int8_t *)InWeights, ChOutputImage, DimFilterKernel,
                                Padding, Stride,(const int8_t *)Bias, BiasShift, OutShift, (int8_t *) OutputImage, DimOutputImage, BufferA, BufferB);

};

template <class T1, class T2, class Toutput>
void QuantizedMatMul2Riscv(S_TENSOR A, S_TENSOR B, S_TENSOR C,
                     S_TENSOR mina, S_TENSOR minb, S_TENSOR maxa,
                     S_TENSOR maxb, S_TENSOR outmin,
                     S_TENSOR outmax, bool transpose_a = false,
                     bool transpose_b = false)
{
    const float min_a = *(mina->read<float>(0, 0));
    const float max_a = *(maxa->read<float>(0, 0));
    const float min_b = *(minb->read<float>(0, 0));
    const float max_b = *(maxb->read<float>(0, 0));

    //auto tensor allocation
    if(C->getSize() == 0) {
      TensorShape c_shape;
      c_shape.push_back((A->getShape())[0]);
      c_shape.push_back((B->getShape())[1]);
      C->resize(c_shape);
    }

    float min_c_value;
    float max_c_value;

    const uint16_t  NumOfRowsInWeightMatrix = (const uint16_t) C->getShape()[1];
    const uint16_t  NumForBiasShift         = 0;
    const uint16_t  NumOfRShiftForOutput    = 0;
    int8_t          InputBias[NumOfRowsInWeightMatrix];
    const uint16_t  DimOfIputVector         = (const uint16_t) C->getShape()[0];
    const T1 *      InputVector             = A->read<T1>(0, 0);
    const T2 *      InputWeightMatrix       = B->read<T2>(0, 0);
    Toutput *          OutputVector         = C->write<Toutput>(0,0); 
    for(int i = 0; i < NumOfRowsInWeightMatrix; i++) { InputBias[i] = 0;}

    riscv_fully_connected_int8((const int8_t *)InputVector,(const int8_t *)InputWeightMatrix, 
                                DimOfIputVector, NumOfRowsInWeightMatrix, NumForBiasShift, NumOfRShiftForOutput,
                                (const int8_t *) InputBias, (int8_t *)OutputVector, nullptr);
    float* c_min = outmin->write<float>(0, 0);
    *c_min = min_c_value;
    float* c_max = outmax->write<float>(0, 0);
    *c_max = max_c_value;

    //TODO convert int8_t OutputVector container to int32_t, such TOut is of type int32_t


};

#endif
