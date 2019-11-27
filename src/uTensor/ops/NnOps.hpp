#ifndef UTENSOR_NN_OPS
#define UTENSOR_NN_OPS

#include "src/uTensor/util/quantization_utils.hpp"
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#if defined(CMSIS)
#include "riscv_nnfunctions.h"
#endif
#include <math.h>
#include <algorithm>
#include <stdio.h>

void Softmax(S_TENSOR input, S_TENSOR output);

class SoftmaxOp : public Operator {
  public:
  SoftmaxOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
#if defined(CMSIS)
#warning "Using CMSIS Softmax implemetation"

    printf("RISCV Softmax\n");

    S_TENSOR input = inputs[0];
    S_TENSOR output = outputs[0]; 
    
    if(input->getDim() != 1)
    {
      for(int i = 0; i < input->getDim() - 1; i++)
      {
        if(input->getShape().at(i) != 1)
        {
          ERR_EXIT("Softmax is supported only for flatten Tensor");
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
#else
    Softmax(inputs[0], outputs[0]);
#endif
  }
};

template <class TIn, class TOut>
void Relu(S_TENSOR input,
          S_TENSOR output) {

  printf("Relu\n");
  const TIn* in = input->read<TIn>(0, 0);
  if (output && output->getSize() == 0) {
      output->resize(input->getShape());
  }
  TOut* out = output->write<TOut>(0, 0);
  for (uint32_t i = 0; i < output->getSize(); i++) {
    if (in[i] > 0.0) {
      out[i] = in[i];
    } else {
      out[i] = 0.0;
    }
  }
}

template<class T1, class TOut>
class ReluOp : public Operator {
  public:
  ReluOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
#if defined(CMSIS)
#warning "Using CMSIS ReLu implemetation"

    printf("RISCV Relu\n");
    S_TENSOR input = inputs[0];
    S_TENSOR output = outputs[0];
    
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

    riscv_relu_int16((int16_t *)out, size);
#else
    Relu<T1, TOut>(inputs[0], outputs[0]);
#endif
  }
};


template <class TIn, class T2, class TOut>
void QuantizedRelu(S_TENSOR input, S_TENSOR in_min, S_TENSOR in_max,
          S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max) {
  const float input_min = in_min->read<T2>(0, 0)[0];
  const float input_max = in_max->read<T2>(0, 0)[0];
  const TIn* in = input->read<TIn>(0, 0);

  printf("QuantizedRelu\n");
  const TOut min_as_quantized =
      FloatToQuantized<TOut>(0.0f, input_min, input_max);
  if (output && output->getSize() == 0) {
      output->resize(input->getShape());
  }
  TOut* out = output->write<TOut>(0, 0);
  for (uint32_t i = 0; i < output->getSize(); i++) {
    if (in[i] > min_as_quantized) {
      out[i] = in[i];
    } else {
      out[i] = min_as_quantized;
    }
  }
  T2* v_out_min = out_min->write<T2>(0, 0);
  *v_out_min = input_min;
  T2* v_out_max = out_max->write<T2>(0, 0);
  *v_out_max = input_max;
}

template<class T1, class T2, class TOut>
class QuantizedReluOp : public Operator {
  public:
  QuantizedReluOp() {
    n_inputs = 3;
    n_outputs = 3;
  }
  virtual void compute() override {
#if defined(CMSIS)
#warning "Using CMSIS Quantized ReLu implemetation"

    printf("RISCV Quantized Relu\n");

    S_TENSOR input    = inputs[0]; 
    S_TENSOR in_min   = inputs[1];
    S_TENSOR in_max   = inputs[2];
    S_TENSOR output   = outputs[0];
    S_TENSOR out_min  = outputs[1];
    S_TENSOR out_max  = outputs[2];

    const float input_min = in_min->read<T2>(0, 0)[0];
    const float input_max = in_max->read<T2>(0, 0)[0];
    const T1* in = input->read<T1>(0, 0);

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

    riscv_relu_int8((int8_t *)out, size);

    T2* v_out_min = out_min->write<T2>(0, 0);
    *v_out_min = input_min;
    T2* v_out_max = out_max->write<T2>(0, 0);
    *v_out_max = input_max;
#else
    QuantizedRelu<T1, T2, TOut>(inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], outputs[2]);
#endif
  }
};

/**
 * https://github.com/tensorflow/tensorflow/blob/982549ea3423df4270ff154e5c764beb43d472da/tensorflow/core/kernels/pooling_ops_common.h
 * https://github.com/tensorflow/tensorflow/blob/40eef4473bda90442bb55fcc67842f097c024580/tensorflow/core/kernels/maxpooling_op.h
 * https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/core/kernels/quantized_pooling_ops.cc#L109
 * https://github.com/tensorflow/tensorflow/blob/982549ea3423df4270ff154e5c764beb43d472da/tensorflow/core/kernels/eigen_pooling.h#L64
 */
template<typename T>
void SpatialMaxPooling(S_TENSOR input, S_TENSOR output, 
                       int window_rows, int window_cols, 
                       int row_stride, int col_stride, 
                       Padding padding, T pad_value = 0) {
  /*
  * Arguments
  * ---------
  * input : S_TENSOR
  *     the intput tensor, assuming format of `NHWC`
  * output : S_TENSOR
  *     the output tensor
  * 
  * Notes
  * -----
  * - padding
  *   - https://www.tensorflow.org/api_guides/python/nn#convolution
  *   - https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
  */
  TensorShape in_shape = input->getShape();
  uint32_t n_batch = in_shape[0];
  uint32_t in_rows = in_shape[1];
  uint32_t in_cols = in_shape[2];
  uint32_t in_channels = in_shape[3];

  size_t out_rows, out_cols;
  int pad_top, pad_left;
  if (padding == VALID) {
    out_rows = ((size_t) ceil(((float)(in_rows - window_rows) + 1) / ((float)row_stride)));
    out_cols = ((size_t) ceil(((float)(in_cols - window_cols) + 1) / ((float)col_stride)));
    // no padding for VALID
    pad_top = 0;
    pad_left = 0;
  } else { 
    // SAME padding
    out_rows = ((size_t) ceil(((float)in_rows) / ((float) row_stride)));
    out_cols = ((size_t) ceil(((float)in_cols) / ((float) col_stride)));
    if (in_rows % row_stride == 0) {
      pad_top = std::max(window_rows - row_stride, 0) / 2;
    } else {
      pad_top = std::max(window_rows - (((int) in_rows) % row_stride), 0) / 2;
    }
    if (in_cols % col_stride == 0) {
      pad_left = std::max(window_cols - col_stride, 0) / 2;
    } else {
      pad_left = std::max(window_cols - (((int) in_cols) % col_stride), 0) / 2;
    }
  }
  TensorShape out_shape;
  out_shape.clear();
  out_shape.push_back(n_batch);
  out_shape.push_back(out_rows);
  out_shape.push_back(out_cols);
  out_shape.push_back(in_channels);
  output->resize(out_shape);

  // strides
  size_t in_batch_stride = input->getStride(0);
  size_t in_row_stride = input->getStride(1);
  size_t in_col_stride = input->getStride(2);
  size_t in_chnl_stride = input->getStride(3);

  size_t out_batch_stride = output->getStride(0);
  size_t out_row_stride = output->getStride(1);
  size_t out_col_stride = output->getStride(2);
  size_t out_chnl_stride = output->getStride(3);

  for (size_t idx_batch = 0; idx_batch < n_batch; ++idx_batch) {
    for (size_t idx_chnl = 0; idx_chnl < in_channels; ++idx_chnl) {
      size_t in_base_offset = idx_batch * in_batch_stride + idx_chnl * in_chnl_stride;
      for (int out_row_idx = 0; out_row_idx < out_rows; ++out_row_idx) {
        for (int out_col_idx = 0; out_col_idx < out_cols; ++out_col_idx) {
          T max_value;
          int base_row_idx = out_row_idx * row_stride - pad_top;
          int base_col_idx = out_col_idx * col_stride - pad_left;
          // if out of boundary, pad with pad_value
          if (base_row_idx < 0 || 
              base_row_idx >= in_rows ||
              base_col_idx < 0 ||
              base_col_idx >= in_cols) {
            max_value = pad_value;
          } else {
            size_t offset = in_base_offset +
                            ((size_t) base_row_idx) * in_row_stride +
                            ((size_t) base_col_idx) * in_col_stride;
            max_value = *(input->read<T>(offset, 0));
          }
          // scanning window
          for (int i = 0; i < window_rows; ++i) {
            for (int j = 0; j < window_cols; ++j) {
              T current_value;
              if (base_row_idx + i < 0 || 
                  base_row_idx + i >= in_rows ||
                  base_col_idx + j < 0 ||
                  base_col_idx + j >= in_cols) {
                current_value = pad_value;
              } else {
                size_t offset = in_base_offset + 
                                ((size_t) base_row_idx + i) * in_row_stride +
                                ((size_t) base_col_idx + j) * in_col_stride;
                current_value = *(input->read<T>(offset, 0));
              }
              if (current_value > max_value) {
                max_value = current_value;
              }
            }
          }
          // write output
          size_t out_offset = idx_batch * out_batch_stride +
                              idx_chnl * out_chnl_stride + 
                              out_row_idx * out_row_stride + 
                              out_col_idx * out_col_stride;
          *(output->write<T>(out_offset, 0)) = max_value;
        }
      }
    }
  }
}

template<typename T>
class MaxPoolingOp : public Operator {
  public:
  MaxPoolingOp(int window_rows, int window_cols,
               int row_stride, int col_stride,
               Padding padding) : _window_rows(window_rows), _window_cols(window_cols),
                                  _row_stride(row_stride), _col_stride(col_stride) {
    _padding = padding;
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {

#if defined(CMSIS)
#warning "Using CMSIS MaxPooling implemetation"

    printf("RISCV MaxPooling");

    S_TENSOR im_in_s  = inputs[0];
    S_TENSOR im_out_s = outputs[0];

    TensorShape in_shape = im_in_s->getShape();
    uint32_t in_rows = in_shape[1];
    uint32_t in_cols = in_shape[2];
    uint32_t in_channels = in_shape[3];

    size_t out_rows, out_cols;
    int pad_top, pad_left;
    if (_padding == VALID)
    {
      out_rows = ((size_t) ceil(((float)(in_rows - _window_rows) + 1) / ((float)_row_stride)));
      out_cols = ((size_t) ceil(((float)(in_cols - _window_cols) + 1) / ((float)_col_stride)));
      // no padding for VALID
      pad_top = 0;
      pad_left = 0;
    } 
    else 
    { 
      // SAME padding
      out_rows = ((size_t) ceil(((float)in_rows) / ((float) _row_stride)));
      out_cols = ((size_t) ceil(((float)in_cols) / ((float) _col_stride)));
      if (in_rows % _row_stride == 0) 
      {
        pad_top = std::max(_window_rows - _row_stride, 0) / 2;
      } 
      else 
      {
        pad_top = std::max(_window_rows - (((int) in_rows) % _row_stride), 0) / 2;
      }
      
      if (in_cols % _col_stride == 0) 
      {
        pad_left = std::max(_window_cols - _col_stride, 0) / 2;
      } 
      else 
      {
        pad_left = std::max(_window_cols - (((int) in_cols) % _col_stride), 0) / 2;
      }
    }

    TensorShape out_shape;
    out_shape.clear();
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
    out_shape.push_back(in_channels);
    im_out_s->resize(out_shape);

    int8_t *       Im_in      = (int8_t *)       im_in_s->read<T>(0, 0);
    const uint16_t dim_im_in  = (const uint16_t) im_in_s->getShape()[1]; 
    const uint16_t ch_im_in   = (const uint16_t) im_in_s->getShape()[3];
    const uint16_t dim_kernel = (const uint16_t) _window_rows;
    const uint16_t padding    = (const uint16_t) pad_top;
    const uint16_t stride     = (const uint16_t) _row_stride;
    const uint16_t dim_im_out = (const uint16_t) im_out_s->getShape()[0];
    int8_t *       bufferA    = nullptr; 
    int8_t *       Im_out     = (int8_t *)       im_out_s->write<T>(0, 0);

    riscv_maxpool_int8_HWC( Im_in,
                        dim_im_in,
                        ch_im_in,
                        dim_kernel,
                        padding,
                        stride,
                        dim_im_out,
                        bufferA,
                        Im_out);
#else
    SpatialMaxPooling<T>(inputs[0], outputs[0], 
                         _window_rows, _window_cols, 
                         _row_stride, _col_stride, _padding);
#endif
  }

  protected:
  int _window_rows, _window_cols;
  int _row_stride, _col_stride;
  Padding _padding;
};

template<typename T>
class QuantizedMaxPoolingOp : public MaxPoolingOp<T> {
  public:
  QuantizedMaxPoolingOp(int window_rows, int window_cols,
                        int row_stride, int col_stride,
                        Padding padding) : MaxPoolingOp<T>(window_rows, window_cols, row_stride, col_stride, padding){
    this->n_inputs = 3;
    this->n_outputs = 3;
  }
  virtual void compute() override {

#if defined(CMSIS)
#warning "Using CMSIS Quantized MaxPooling implemetation"

    printf("RISCV Quantized MaxPooling\n");
    S_TENSOR im_in_s  = this->inputs[0];
    S_TENSOR im_out_s = this->outputs[0];


    TensorShape in_shape = im_in_s->getShape();
    uint32_t in_rows = in_shape[1];
    uint32_t in_cols = in_shape[2];
    uint32_t in_channels = in_shape[3];

    size_t out_rows, out_cols;
    int pad_top, pad_left;
    if (this->_padding == VALID)
    {
      out_rows = ((size_t) ceil(((float)(in_rows - this->_window_rows) + 1) / ((float)this->_row_stride)));
      out_cols = ((size_t) ceil(((float)(in_cols - this->_window_cols) + 1) / ((float)this->_col_stride)));
      // no padding for VALID
      pad_top = 0;
      pad_left = 0;
    } 
    else 
    { 
      // SAME padding
      out_rows = ((size_t) ceil(((float)in_rows) / ((float) this->_row_stride)));
      out_cols = ((size_t) ceil(((float)in_cols) / ((float) this->_col_stride)));
      if (in_rows % this->_row_stride == 0) 
      {
        pad_top = std::max(this->_window_rows - this->_row_stride, 0) / 2;
      } 
      else 
      {
        pad_top = std::max(this->_window_rows - (((int) in_rows) % this->_row_stride), 0) / 2;
      }
      
      if (in_cols % this->_col_stride == 0) 
      {
        pad_left = std::max(this->_window_cols - this->_col_stride, 0) / 2;
      } 
      else 
      {
        pad_left = std::max(this->_window_cols - (((int) in_cols) % this->_col_stride), 0) / 2;
      }
    }

    TensorShape out_shape;
    out_shape.clear();
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
    out_shape.push_back(in_channels);
    im_out_s->resize(out_shape);

    int8_t *       Im_in      = (int8_t *)       im_in_s->read<T>(0, 0);
    const uint16_t dim_im_in  = (const uint16_t) im_in_s->getShape()[1]; 
    const uint16_t ch_im_in   = (const uint16_t) im_in_s->getShape()[3];
    const uint16_t dim_kernel = (const uint16_t) this->_window_rows;
    const uint16_t padding    = (const uint16_t) pad_top;
    const uint16_t stride     = (const uint16_t) this->_row_stride;
    const uint16_t dim_im_out = (const uint16_t) im_out_s->getShape()[0];
    int8_t *       bufferA    = nullptr; 
    int8_t *       Im_out     = (int8_t *)       im_out_s->write<T>(0, 0);

    riscv_maxpool_int8_HWC( Im_in,
                        dim_im_in,
                        ch_im_in,
                        dim_kernel,
                        padding,
                        stride,
                        dim_im_out,
                        bufferA,
                        Im_out);
#else
    S_TENSOR in_min_tensor = this->inputs[1];
    S_TENSOR in_max_tensor = this->inputs[2];
    float in_min = *(in_min_tensor->read<float>(0, 0));
    float in_max = *(in_max_tensor->read<float>(0, 0));

    // new range
    float new_in_max = in_max > 0 ? in_max : 0;
    float new_in_min = in_min < 0 ? in_min : 0;
    RequantizeManyInNewRange<T, T>(this->inputs[0], this->inputs[0]->getSize(),
                                   in_min, in_max, new_in_min, new_in_max, 
                                   this->inputs[0]);
    // write new range
    S_TENSOR out_min_tensor = this->outputs[1];
    S_TENSOR out_max_tensor = this->outputs[2];
    *(out_min_tensor->write<float>(0, 0)) = new_in_min;
    *(out_max_tensor->write<float>(0, 0)) = new_in_max;
    // pooling
    T pad_value = FloatToQuantized<T>(0, new_in_min, new_in_max);
    SpatialMaxPooling<T>(this->inputs[0], this->outputs[0],
                         this->_window_rows, this->_window_cols,
                         this->_row_stride, this->_col_stride,
                         this->_padding,
                         pad_value);
#endif
  }
};

#endif  // UTENSOR_NN_OPS
