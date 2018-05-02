#include <vector>

#include "caffe/layers/sparse_wino_conv_layer.hpp"

namespace caffe {

template<typename Dtype>
__inline__ void init(Dtype* weight,int num_output_, int channels_) {
  printf("weight init = %f\n", weight[0]);
  Dtype val = weight[0];

  Dtype G[6][3] = {
    {1./4, 0., 0. },
    {-1./6 , -1./6, -1./6},
    {-1./6 , 1./6, -1./6},
    {1./24 , 1./12, 1./6},
    {1./24 ,-1./12 ,1./6 },
    { 0.,0. ,1. }
  };

  Dtype tmp_1[6][3], tmp_2[6][6];


  for(int i = 0; i < 6; ++i){
    for(int j = 0; j < 3; ++j){
      tmp_1[i][j] = (G[i][0] + G[i][1] + G[i][2]) * val;
    }
  }
  for(int i = 0; i < 6; ++i){
    for(int j = 0; j < 6; ++j){
      tmp_2[i][j] = tmp_1[i][0] * G[j][0] + tmp_1[i][1] * G[j][1] + tmp_1[i][2] * G[j][2];
    }
  }

  for(int i = 0; i < 6; ++i)
    for(int k = 0; k < 6; ++k)
    {
      for(int j = 0; j < num_output_ * channels_; ++j)
      weight[ (i * 6  + k) * num_output_ * channels_ + j] = tmp_2[i][k];
    }

  printf("weight after init = %f\n", weight[0]);
}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  vector<int> kernel_shape_;
  kernel_shape_.push_back(36);
  kernel_shape_.push_back(BaseConvolutionLayer<Dtype>::num_output_);
  kernel_shape_.push_back(BaseConvolutionLayer<Dtype>::channels_);
  BaseConvolutionLayer<Dtype>::blobs_[0]->Reshape(kernel_shape_);

  init(BaseConvolutionLayer<Dtype>::blobs_[0]->mutable_cpu_data(),
       BaseConvolutionLayer<Dtype>::num_output_  ,
       BaseConvolutionLayer<Dtype>::channels_);
  // reshape kernel

}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  // reshape input
  vector<int> in_buffer_shape_;
  in_buffer_shape_.push_back(36);
  in_buffer_shape_.push_back(BaseConvolutionLayer<Dtype>::channels_);
  in_buffer_shape_.push_back(((BaseConvolutionLayer<Dtype>::input_shape(1) + 3) / 4) * ((BaseConvolutionLayer<Dtype>::input_shape(2) + 3) / 4));
  //printf("%d %d %d \n", in_buffer_shape_[0], in_buffer_shape_[1], in_buffer_shape_[2]);
  in_buffer_.Reshape(in_buffer_shape_);



  // reshape output   out_diff
  vector<int> out_buffer_shape_;
  out_buffer_shape_.push_back(36);
  out_buffer_shape_.push_back(BaseConvolutionLayer<Dtype>::num_output_);
  out_buffer_shape_.push_back(((BaseConvolutionLayer<Dtype>::input_shape(1) + 3) / 4) * ((BaseConvolutionLayer<Dtype>::input_shape(2) + 3) / 4));
  out_buffer_.Reshape(out_buffer_shape_);
  //printf("%d %d %d \n", out_buffer_shape_[0], out_buffer_shape_[1], out_buffer_shape_[2]);

}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;

    assert(kernel_extent == 6 && stride_data[i] == 1 && pad_data[i] == 5);  //TODO

    const int output_dim = (input_dim + pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }

}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::scatter_img(const Dtype* image, int h, int w, int channel, Dtype* buffer){
  // image[c][h * w]     buffer[36][c][ tile_num ]
  Dtype B_T[6][6] = { {4.0, 0.0, -5.0, 0.0, 1.0, 0.0},
                      {0.0, -4.0, -4.0, 1.0, 1.0, 0.0},
                      {0.0, 4.0, -4.0, -1.0, 1.0, 0.0},
                      {0.0, -2.0, -1.0, 2.0, 1.0, 0.0},
                      {0.0, 2.0, -1.0, -2.0, 1.0, 0.0},
                      {0.0, 4.0, 0.0, -5.0, 0.0, 1.0} };

  int max_h_cnt = (h + 3) / 4, max_w_cnt = (w + 3) / 4;
  int max_dim = max_h_cnt * max_w_cnt;

  for(int i = 0; i < channel; ++i){
    const Dtype* local_image = image + i * h * w;
    for(int h_cnt = 0; h_cnt < max_h_cnt; ++h_cnt){
      for(int w_cnt = 0; w_cnt < max_w_cnt; ++w_cnt){
        int tile_idx = h_cnt * max_w_cnt + w_cnt;
        Dtype patch_buf[6][6], tmp_buf[6][6];
        caffe_set(36, (Dtype)0., &patch_buf[0][0]);

        for(int h_off = 0; h_off < 6; ++h_off){
          int local_h = h_cnt * 4 + h_off - 1;  // why -1?
          if(local_h < 0) continue;
          if(local_h >= h ) break;

          for(int w_off = 0; w_off < 6; ++w_off){
            int local_w = w_cnt * 4 + w_off - 1;  // why -1?
            if(local_w < 0) continue;
            if(local_w >= w ) break;

            patch_buf[h_off][w_off] = local_image[local_h * w + local_w];

          }
        }
        /*
        for(int h_off = 0; h_off < 6; ++h_off){
          for(int w_off = 0; w_off < 6; ++w_off)
            printf("%f ", patch_buf[h_off][w_off]);
          printf("\n");
        }
        printf("---------------------\n");*/
        for(int h_off = 0; h_off < 6; ++h_off) {
          for(int w_off = 0; w_off < 6; ++w_off){
            tmp_buf[h_off][w_off] = B_T[h_off][0] * patch_buf[0][w_off] + B_T[h_off][1] * patch_buf[1][w_off] + B_T[h_off][2] * patch_buf[2][w_off]
            + B_T[h_off][3] * patch_buf[3][w_off] + B_T[h_off][4] * patch_buf[4][w_off] + B_T[h_off][5] * patch_buf[5][w_off] ;
          }
        }
        for(int h_off = 0; h_off < 6; ++h_off){
          for(int w_off = 0; w_off < 6; ++w_off){
            int k_idx = h_off * 6 + w_off;
            buffer[(k_idx * channel + i) * max_dim + tile_idx]
            = tmp_buf[h_off][0] * B_T[w_off][0] + tmp_buf[h_off][1] * B_T[w_off][1] + tmp_buf[h_off][2] * B_T[w_off][2]
            + tmp_buf[h_off][3] * B_T[w_off][3] + tmp_buf[h_off][4] * B_T[w_off][4] + tmp_buf[h_off][5] * B_T[w_off][5] ;
            //		printf("%f ", buffer[(k_idx * channel +i) * max_dim + tile_idx]);
          }
          //	printf("\n");
        }

      //	printf("---------------------\n");
      //	printf("---------------------\n");
      }
    }
  }


}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::reverse_scatter_img(const Dtype* buffer, int h, int w, int channel, Dtype* image){
// buffer[36][c][tilenum]  image[c][h * w]
  Dtype B_T[6][6] = { {4.0, 0.0, -5.0, 0.0, 1.0, 0.0},
                      {0.0, -4.0, -4.0, 1.0, 1.0, 0.0},
                      {0.0, 4.0, -4.0, -1.0, 1.0, 0.0},
                      {0.0, -2.0, -1.0, 2.0, 1.0, 0.0},
                      {0.0, 2.0, -1.0, -2.0, 1.0, 0.0},
                      {0.0, 4.0, 0.0, -5.0, 0.0, 1.0} };

  int max_h_cnt = (h + 3) / 4, max_w_cnt = (w + 3) / 4;
  int max_dim = max_h_cnt * max_w_cnt;

  for(int i = 0; i < channel; ++i){
    Dtype* local_image = image + i * h * w;
    for(int h_cnt = 0; h_cnt < max_h_cnt; ++h_cnt){
      for(int w_cnt = 0; w_cnt < max_w_cnt; ++w_cnt){
        int tile_idx = h_cnt * max_w_cnt + w_cnt;

        Dtype patch_buf[6][6], tmp_buf[6][6];
        caffe_set(36, (Dtype)0., &patch_buf[0][0]);

        for(int h_off = 0; h_off < 6; ++h_off)
          for(int w_off = 0; w_off < 6; ++w_off){
            int k_idx = h_off * 6 + w_off;
            patch_buf[h_off][w_off] = buffer[ (k_idx * channel + i) * max_dim + tile_idx];
          }

        for(int h_off = 0; h_off < 6; ++h_off)
          for(int w_off = 0; w_off < 6; ++w_off){
            tmp_buf[h_off][w_off] = B_T[0][h_off] * patch_buf[0][w_off] + B_T[1][h_off] * patch_buf[1][w_off] + B_T[2][h_off] * patch_buf[2][w_off]
            + B_T[3][h_off] * patch_buf[3][w_off] + B_T[4][h_off] * patch_buf[4][w_off] + B_T[5][h_off] * patch_buf[5][w_off] ;
          }

        for(int h_off = 0; h_off < 6; ++h_off){
          int local_h = h_cnt * 4 + h_off - 1;
          if(local_h < 0) continue;
          if(local_h >= h) break;
          for(int w_off = 0; w_off < 6; ++w_off){
            int local_w = w_cnt * 4 + w_off - 1;
            if(local_w < 0) continue;
            if(local_w >= h) break;
            local_image[local_h * w + local_w]
            += tmp_buf[h_off][0] * B_T[0][w_off] + tmp_buf[h_off][1] * B_T[1][w_off] + tmp_buf[h_off][2] * B_T[2][w_off]
             + tmp_buf[h_off][3] * B_T[3][w_off] + tmp_buf[h_off][4] * B_T[4][w_off] + tmp_buf[h_off][5] * B_T[5][w_off] ;
          }
        }
      }

    }
  }

}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::gather_img(const Dtype* buffer, int h, int w, int channel, Dtype* image){
// buffer[36][c][tile_num]   image[c][h * w]

  Dtype A_T[4][6] = { {1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
                      {0.0, 1.0, -1.0, 2.0, -2.0, 0.0},
                      {0.0, 1.0, 1.0, 4.0, 4.0, 0.0},
                      {0.0, 1.0, -1.0, 8.0, -8.0, 1.0}};

  int max_h_cnt = (h + 3) / 4, max_w_cnt = (w + 3) / 4;
  int max_dim = max_h_cnt * max_w_cnt;

  for(int i = 0; i < channel; ++i){
    Dtype* local_image = image + i * h * w;
    for(int h_cnt = 0; h_cnt < max_h_cnt; ++h_cnt){
      for(int w_cnt = 0; w_cnt < max_w_cnt; ++w_cnt){
        int tile_idx = h_cnt * max_w_cnt + w_cnt;
        Dtype patch_buf[6][6], tmp_buf[4][6];
        caffe_set(36, (Dtype)0., &patch_buf[0][0]);

//	printf("---------------------\n");
        for(int h_off = 0; h_off < 6; ++h_off){
          for(int w_off = 0; w_off < 6; ++w_off){
            int k_idx = h_off * 6 + w_off;
            patch_buf[h_off][w_off] = buffer[ (k_idx * channel + i) * max_dim + tile_idx];
//	printf("%f ", patch_buf[h_off][w_off]);
          }
//	printf("\n");
        }
//	printf("---------------------\n");

        for(int h_off = 0; h_off < 4; ++h_off){
          for(int w_off = 0; w_off < 6; ++w_off){
            tmp_buf[h_off][w_off] = A_T[h_off][0] * patch_buf[0][w_off]
                                  + A_T[h_off][1] * patch_buf[1][w_off]
                                  + A_T[h_off][2] * patch_buf[2][w_off]
                                  + A_T[h_off][3] * patch_buf[3][w_off]
                                  + A_T[h_off][4] * patch_buf[4][w_off]
                                  + A_T[h_off][5] * patch_buf[5][w_off] ;
//	printf("%f ", tmp_buf[h_off][w_off]);
          }
//	printf("\n");
        }
//	printf("*************************************\n");

        for(int h_off = 0; h_off < 4; ++h_off) {
          for(int w_off = 0; w_off < 4; ++w_off){
            int local_h = h_cnt * 4 + h_off, local_w = w_cnt * 4 + w_off;
            local_image[local_h * w + local_w]
            = tmp_buf[h_off][0] * A_T[w_off][0] + tmp_buf[h_off][1] * A_T[w_off][1]
            + tmp_buf[h_off][2] * A_T[w_off][2]
            + tmp_buf[h_off][3] * A_T[w_off][3] + tmp_buf[h_off][4] * A_T[w_off][4]
            + tmp_buf[h_off][5] * A_T[w_off][5] ;
//	printf("%f ", local_image[local_h * w + local_w]);
          }
//	printf("\n");
        }
//	printf("#####################################\n");
      }
    }
  }
}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::wino_forward(const Dtype* bottom, const Dtype* weight, Dtype* top){

  int bottom_h = this->input_shape(1), bottom_w = this->input_shape(2);
  //col_buffer 36 C H/4 * W/4

  int in_buffer_dim = ((bottom_h + 3) / 4) * ((bottom_w + 3) / 4);

  //printf("in_buffer_dim = %d\n", in_buffer_dim);

  Dtype* in_buffer_ptr = in_buffer_.mutable_cpu_data();
  Dtype* out_buffer_ptr = out_buffer_.mutable_cpu_data();


  scatter_img(bottom, bottom_h, bottom_w, this->channels_ , in_buffer_ptr);

  const Dtype* const_in_ptr = in_buffer_.cpu_data();

  for( int i = 0; i < 36; ++i)
  {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, this->num_output_, in_buffer_dim, this->channels_,
      (Dtype)1., weight + i * this->num_output_ * this->channels_, const_in_ptr + i * this->channels_ * in_buffer_dim,
      (Dtype)0., out_buffer_ptr + i * this->num_output_ * in_buffer_dim);  // why 0.?

      /*	printf("weight : \n");
      for(int m = 0; m < this->num_output_; ++m){
      for(int n = 0; n < this->channels_; ++n)
      printf("%f ", weight[i * this->num_output_ * this->channels_ + m * this->channels_ + n]);
      printf("\n");
    }

    printf("input : \n");
    for(int m = 0; m < this->channels_; ++m){
    for(int n = 0; n < in_buffer_dim; ++n)
    printf("%f ", const_in_ptr[i * in_buffer_dim * this->channels_ + m * in_buffer_dim + n]);
    printf("\n");
  }
  printf("output : \n");
  for(int m = 0; m < this->num_output_; ++m){
  for(int n = 0; n < in_buffer_dim; ++n)
  printf("%f ", out_buffer_ptr[i * this->num_output_ * in_buffer_dim + m * in_buffer_dim + n]);
  printf("\n");
}*/
  }
  const Dtype* const_out_ptr = out_buffer_.cpu_data();
  gather_img(const_out_ptr, bottom_h, bottom_w, this->num_output_, top);
}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::reverse_gather_img(const Dtype* image, int h, int w, int channel,  Dtype* buffer){
  // image[c][h * w]     buffer[36][c][tile_num]
  Dtype A_T[4][6] = { {1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
                      {0.0, 1.0, -1.0, 2.0, -2.0, 0.0},
                      {0.0, 1.0, 1.0, 4.0, 4.0, 0.0},
                      {0.0, 1.0, -1.0, 8.0, -8.0, 1.0}};
  int max_h_cnt = (h + 3) / 4, max_w_cnt = (w + 3) / 4;
  int max_dim = max_h_cnt * max_w_cnt;

  for(int i = 0; i < channel; ++i){
    const Dtype* local_image = image + i * h * w;
    for(int h_cnt = 0; h_cnt < max_h_cnt; ++h_cnt){
      for(int w_cnt = 0; w_cnt < max_w_cnt; ++w_cnt){
        int tile_idx = h_cnt * max_w_cnt + w_cnt;

        Dtype patch_buf[4][4], tmp_buf[6][4];
        caffe_set(16, (Dtype)0., &patch_buf[0][0]);

        for(int h_off = 0; h_off < 4; ++h_off){
          int local_h = h_cnt * 4 + h_off;
          if(local_h >= h) break;

          for(int w_off = 0; w_off < 4; ++w_off){
            int local_w = w_cnt * 4 + w_off;
            if(local_w >= w) break;
            patch_buf[h_off][w_off] = local_image[local_h * w + local_w];

          }
        }

        for(int h_off = 0; h_off < 6; ++h_off){
          for(int w_off = 0; w_off < 4; ++w_off){
            tmp_buf[h_off][w_off] = A_T[0][h_off] * patch_buf[0][w_off] + A_T[1][h_off] * patch_buf[1][w_off] + A_T[2][h_off] * patch_buf[2][w_off] + A_T[3][h_off] * patch_buf[3][w_off] ;
          }
        }

        for(int h_off = 0; h_off < 6; ++h_off){
          for(int w_off = 0; w_off < 6; ++w_off){
            int k_idx = h_off * 6 + w_off;
            buffer[ (k_idx * channel + i) * max_dim + tile_idx]
            = tmp_buf[h_off][0] * A_T[0][w_off] + tmp_buf[h_off][1] * A_T[1][w_off] + tmp_buf[h_off][2] * A_T[2][w_off] + tmp_buf[h_off][3] * A_T[3][w_off] ;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::update_weight(const Dtype* bottom, const Dtype* top_diff, Dtype* weight_diff){

  int bottom_h = this->input_shape(1), bottom_w = this->input_shape(2);
  int in_buffer_dim = ((bottom_h + 3) / 4) * ((bottom_w + 3) / 4);

  Dtype* in_buffer_ptr = in_buffer_.mutable_cpu_data();

  Dtype* out_buffer_ptr = out_buffer_.mutable_cpu_diff();

  scatter_img(bottom, bottom_h, bottom_w, this->channels_, in_buffer_ptr);

  reverse_gather_img(top_diff, bottom_h, bottom_w, this->num_output_, out_buffer_ptr);

  const Dtype* const_in_ptr = in_buffer_.cpu_data();
  const Dtype* const_out_ptr = out_buffer_.cpu_diff();

  for(int i = 0; i < 36; ++i){

    caffe_cpu_gemm(CblasNoTrans, CblasTrans, this->num_output_, this->channels_, in_buffer_dim,
      (Dtype)1., const_out_ptr + i * this->num_output_ * in_buffer_dim, const_in_ptr + i * this->channels_ * in_buffer_dim,
      (Dtype)1., weight_diff + i * this->num_output_ * this->channels_);

  }


}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::update_bottom(const Dtype* top_diff, const Dtype* weight, Dtype* bottom_diff){

  int bottom_h = this->input_shape(1), bottom_w = this->input_shape(2);
  int bottom_dim = bottom_h * bottom_w;
  int in_buffer_dim = ((bottom_h + 3) / 4) * ((bottom_w + 3) / 4);


  // top_diff has already been tiled by update_weight in
  Dtype* out_buffer_ptr = out_buffer_.mutable_cpu_diff();
  reverse_gather_img(top_diff, bottom_h, bottom_w, this->num_output_, out_buffer_ptr);

  const Dtype* const_out_ptr = out_buffer_.cpu_diff();

  Dtype* in_buffer_ptr = in_buffer_.mutable_cpu_diff();

  /*
  Dtype A[6] = {1 ,3, 2, 2, 3, 1};
  Dtype B[2] = {2, 1};
  Dtype C[3];

  caffe_cpu_gemm(CblasNoTrans,CblasTrans, 3, 1, 2, (Dtype)1.0, A, B, (Dtype)0.0, C);

  printf("rst: %f %f %f\n", C[0], C[1], C[2]);
  */
  for(int i = 0; i < 36; ++i){

    caffe_cpu_gemm(CblasTrans, CblasNoTrans, this->channels_, in_buffer_dim, this->num_output_,
      (Dtype)1., weight + i * this->num_output_ * this->channels_, const_out_ptr + i * this->num_output_ * in_buffer_dim,
      (Dtype)0., in_buffer_ptr + i * this->channels_ * in_buffer_dim);

  }
  const Dtype* const_in_ptr  = in_buffer_.cpu_diff();
  caffe_set(this->channels_ * bottom_dim, (Dtype)0., bottom_diff);
  reverse_scatter_img(const_in_ptr, bottom_h, bottom_w, this->channels_, bottom_diff);

}

template <typename Dtype>
inline void trans_weight_partial(Dtype* weight_diff, int num_output_, int channels_){

  Dtype T[6][6] = { {1./16, -1./24, -1./24, 1./96, 1./96, 0. },
                    {-1./24, 1./12, 1./36, -7./144, -1./72, -1./6},
                    {-1./24, 1./36, 1./12, -1./48, -7./144, -1./6 },
                    {1./96, -7./144, -1./48, 21./576, 13./576, 1./6},
                    {1./96, -1./72, -7./144, 13./576, 21./576, 1./6},
                    {0., -1/6, -1./6, 1./6, 1./6, 1.}};


  for(int i = 0; i < num_output_; ++i){
    for(int j = 0; j < channels_; ++j){

      Dtype patch[6][6], tmp[6][6];

      for(int h_off = 0; h_off < 6; ++h_off){
        for(int w_off = 0; w_off < 6; ++w_off){
          int k_idx= h_off * 6 + w_off;
          patch[h_off][w_off] = weight_diff[(k_idx * num_output_ + i) * channels_ + j];
        }
      }


      for(int h_off = 0; h_off < 6; ++h_off){
        for(int w_off = 0; w_off < 6; ++w_off){
          tmp[h_off][w_off] = T[h_off][0] * patch[0][w_off] + T[h_off][1] * patch[1][w_off] + T[h_off][2] * patch[2][w_off]
          + T[h_off][3] * patch[3][w_off] + T[h_off][4] * patch[4][w_off] + T[h_off][5] * patch[5][w_off] ;
        }
      }

      for(int h_off = 0; h_off < 6; ++h_off){
        for(int w_off = 0; w_off < 6; ++w_off){
          int k_idx = h_off * 6 + w_off;
          weight_diff[(k_idx * num_output_ + i) * channels_ + j]
          = tmp[h_off][0] * T[0][w_off] + tmp[h_off][1] * T[1][w_off] + tmp[h_off][2] * T[2][w_off]
          + tmp[h_off][3] * T[3][w_off] + tmp[h_off][4] * T[4][w_off] + tmp[h_off][5] * T[5][w_off] ;
        }
      }
    }
  }
}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {

    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
//printf("bottom_sz = %d in_buf sz = %d  weight sz = %d  out_buf sz = %d top_sz = %d\n", bottom[i]->count() / this->num_, in_buffer_.count(), this->blobs_[0]->count(),  out_buffer_.count(), top[i]->count()/ this->num_);
//printf("bottom_dim = %d top_dim = %d\n", this->bottom_dim_, this->top_dim_);
    for (int n = 0; n < this->num_; ++n) {

//	caffe_set(this->bottom_dim_, (Dtype)1., bottom_data_2 + n* this->bottom_dim_);
//	init(weight_2, this->num_output_, this->channels_);
       this->wino_forward(bottom_data + n * this->bottom_dim_, weight,
           top_data + n * this->top_dim_);

/*	for(int s = 0; s < this->top_dim_; ++s){
	 if( top_data[n * this->top_dim_ + s] > 180.f) {
	  printf("%f %d %d %d ", top_data[ n * this->top_dim_ + s], n , this->top_dim_ , s);
	  exit(0);
  	 }
	}
*/
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }

  }

}

template <typename Dtype>
void SparseWinoConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

//  printf(" %f | ", weight[0]);

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }


    if (this->param_propagate_down_[0] || propagate_down[i]) {

      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
       if (this->param_propagate_down_[0]) {
         this->update_weight(bottom_data + n * this->bottom_dim_,
             top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {

         this->update_bottom(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  trans_weight_partial(weight_diff, this->num_output_, this->channels_);
}

//#ifdef CPU_ONLY
//STUB_GPU(SparseWinoConvLayer);
//#endif

INSTANTIATE_CLASS(SparseWinoConvLayer);
REGISTER_LAYER_CLASS(SparseWinoConv);
}  // namespace caffe
