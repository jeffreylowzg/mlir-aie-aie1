//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#define TWID1 {{ 32767,      0}}
#define TWID2 {{ 32767,      0}, {     0, -32768}}
#define TWID4 {{ 32767,      0}, { 23170, -23170},\
               {     0, -32768}, {-23170, -23170}}
#define TWID8 {{ 32767,      0}, { 30274, -12540},\
               { 23170, -23170}, { 12540, -30274},\
               {     0, -32768}, {-12540, -30274},\
               {-23170, -23170}, {-30274, -12540}}
#define TWID16 {{ 32767,      0}, { 32138,  -6393},\
                { 30274, -12540}, { 27246, -18205},\
                { 23170, -23170}, { 18205, -27246},\
                { 12540, -30274}, {  6393, -32138},\
                {     0, -32768}, { -6393, -32138},\
                {-12540, -30274}, {-18205, -27246},\
                {-23170, -23170}, {-27246, -18205},\
                {-30274, -12540}, {-32138,  -6393}}

#define HANN_1  {0  , 0  , 1  , 1  , \
                 2  , 2  , 4  , 4  , \ 
                 8  , 8  , 12 , 12 , \ 
                 16 , 16 , 21 , 21 }

#define HANN_2  {26 , 26 , 31 , 31 , \ 
                 36 , 36 , 40 , 40 , \
                 44 , 44 , 47 , 47 , \ 
                 49 , 49 , 50 , 50 }

#define HANN_3  {50 , 50 , 49 , 49 , \ 
                 47 , 47 , 44 , 44 , \
                 40 , 40 , 36 , 36 , \ 
                 31 , 31 , 26 , 26 }
                 
#define HANN_4  {21 , 21 , 16 , 16 , \ 
                 12 , 12 , 8  , 8  , \
                 4  , 4  , 2  , 2  , \
                 1  , 1  , 0  , 0 }

template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *a,T_out *c) {
  constexpr int vec_factor = 16;
  constexpr int shuf_factor = 32;
  constexpr int stride = 4;
  constexpr int stride_complex = 8;
  constexpr int sample_length = 128;
  
  event0();
  static constexpr T_in hann1_arr[vec_factor] = HANN_1;
  static constexpr T_in hann2_arr[vec_factor] = HANN_2;
  static constexpr T_in hann3_arr[vec_factor] = HANN_3;
  static constexpr T_in hann4_arr[vec_factor] = HANN_4;

  aie::vector<T_in, vec_factor> hann1;
  aie::vector<T_in, vec_factor> hann2;
  aie::vector<T_in, vec_factor> hann3;
  aie::vector<T_in, vec_factor> hann4;

  hann1 = aie::load_v<vec_factor>(hann1_arr);
  hann2 = aie::load_v<vec_factor>(hann2_arr);
  hann3 = aie::load_v<vec_factor>(hann3_arr);
  hann4 = aie::load_v<vec_factor>(hann4_arr);

  aie::vector<T_out, vec_factor> cout1;
  aie::vector<T_out, vec_factor> cout2;
  aie::vector<T_out, vec_factor> cout3;
  aie::vector<T_out, vec_factor> cout4;

  aie::vector<T_in, vec_factor>  A0;
  aie::vector<T_in, vec_factor>  A1;
  aie::vector<T_in, vec_factor>  A2;
  aie::vector<T_in, vec_factor>  A3;

  aie::accum<acc48, vec_factor> acc1;
  aie::accum<acc48, vec_factor> acc2;
  aie::accum<acc48, vec_factor> acc3;
  aie::accum<acc48, vec_factor> acc4;

  const int loops = N / (2*128); //2
  const int F = ((128-shuf_factor) / stride) + 1; //25
    T_in *__restrict pA_stride = a; //increment by 256
    T_in *__restrict pA_window = a + vec_factor; 
    T_out *__restrict pC1 = c; //x*25*32=x*800
    for (int i = 0; i < F; i++) //every complete for is 25*64=400
      chess_prepare_for_pipelining chess_loop_range(16, ) {
        A0 = aie::load_v<vec_factor>(pA_stride);
        pA_stride += stride_complex;
        A1 = aie::load_v<vec_factor>(pA_window);
        pA_window = pA_window + vec_factor;
        A2 = aie::load_v<vec_factor>(pA_window);
        pA_window = pA_window + vec_factor;
        A3 = aie::load_v<vec_factor>(pA_window);
        pA_window = pA_stride + vec_factor;
        
        //apply hann window
        acc1 = aie::mul(hann1, A0);
        acc2 = aie::mul(hann2, A1);
        acc3 = aie::mul(hann3, A2);
        acc4 = aie::mul(hann4, A3);
        
        //cast back to v16int16
        cout1 = acc1.to_vector<T_in>();
        cout2 = acc2.to_vector<T_in>();
        cout3 = acc3.to_vector<T_in>();
        cout4 = acc4.to_vector<T_in>();
        
        aie::store_v(pC1, cout1);
        pC1 += vec_factor;
        aie::store_v(pC1, cout2);
        pC1 += vec_factor;
        aie::store_v(pC1, cout3);
        pC1 += vec_factor;
        aie::store_v(pC1, cout4);
        pC1 += vec_factor;
      }
  // }
  event1();
}

extern "C" {


void eltwise_mul_int16_vector(int16 *a_in, int16 *c_out) {
  eltwise_vmul<int16, int16, 256>(a_in,c_out);
}

} // extern "C"

// Assume factor is at least 16
template <typename TT_TWID, typename TT_DATA>
void fft_vectorized(TT_DATA *ibuf, TT_DATA *__restrict obuf) {
  aie::set_rounding(aie::rounding_mode::positive_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
  
  constexpr unsigned int N = 32;
  constexpr unsigned int SHIFT_TW = 15; // Indicates the decimal point of the twiddles
  constexpr unsigned int SHIFT_DT = 15; // Shift applied to apply to dit outputs
  constexpr bool     INVERSE  = false;
  constexpr unsigned int REPEAT   = 25; // number of loops 

  alignas(aie::vector_decl_align) static constexpr TT_TWID    tw1[ 1] = TWID1;
  alignas(aie::vector_decl_align) static constexpr TT_TWID    tw2[ 2] = TWID2;
  alignas(aie::vector_decl_align) static constexpr TT_TWID    tw4[ 4] = TWID4;
  alignas(aie::vector_decl_align) static constexpr TT_TWID    tw8[ 8] = TWID8;
  alignas(aie::vector_decl_align) static constexpr TT_TWID   tw16[16] = TWID16;

  alignas(aie::vector_decl_align) TT_DATA tbuf[N];

  unsigned long long st, et;
  event0();
  // st = get_cycles();

  // Perform FFT:
  for (int rr=0; rr < REPEAT; rr++)
      chess_prepare_for_pipelining
      chess_loop_range(REPEAT,)
      {
      // aie::fft_dit_r2_stage<16>(ibuf, tw1,  N, SHIFT_TW, SHIFT_DT, INVERSE, obuf);
      aie::fft_dit_r2_stage<16>(ibuf, tw1,  N, SHIFT_TW, SHIFT_DT, INVERSE, tbuf);
      aie::fft_dit_r2_stage< 8>(tbuf, tw2,  N, SHIFT_TW, SHIFT_DT, INVERSE, ibuf);
      aie::fft_dit_r2_stage< 4>(ibuf, tw4,  N, SHIFT_TW, SHIFT_DT, INVERSE, tbuf);
      aie::fft_dit_r2_stage< 2>(tbuf, tw8,  N, SHIFT_TW, SHIFT_DT, INVERSE, ibuf);
      aie::fft_dit_r2_stage< 1>(ibuf, tw16, N, SHIFT_TW, SHIFT_DT, INVERSE, obuf);
      ibuf += N;
      obuf += N;

      }

  event1();
}

extern "C" {

void fft_vec_int16(cint16 *a_in, cint16 *c_out) {
  fft_vectorized<cint16, cint16>(a_in, c_out);
}

} // extern "C"


template <typename T, int N>
__attribute__((noinline)) void passThrough_aie(T *restrict in, T *restrict out,
                                               const int32_t height,
                                               const int32_t width) {
  event0();

  // v32int16 *restrict outPtr = (v32int16 *)out;
  // v32int16 *restrict inPtr =  (v32int16 *)in;
  int16 *restrict outPtr = (int16 *)out;
  int16 *restrict inPtr =  (int16 *)in;

  // for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  for (int j = 0; j < (height * width); j ++) // Nx samples per loop
    // chess_prepare_for_pipelining chess_loop_range(32, ) 
    { *outPtr++ = *inPtr++; }

  event1();
}

extern "C" {

#if BIT_WIDTH == 8

void passThroughLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
  printf("passThroughLine BIT_WIDTH\n");
  passThrough_aie<uint8_t, 64>(in, out, 1, lineWidth);
}

void passThroughTile(uint8_t *in, uint8_t *out, int32_t tileHeight,
                     int32_t tileWidth) {
  printf("passThroughTile BIT_WIDTH\n");
  passThrough_aie<uint8_t, 64>(in, out, tileHeight, tileWidth);
}

#elif BIT_WIDTH == 16

void passThroughLine(int16_t *in, int16_t *out, int32_t lineWidth) {
  printf("passThroughLine BIT_WIDTH\n");
  passThrough_aie<int16_t, 32>(in, out, 1, lineWidth);
}

void passThroughTile(int16_t *in, int16_t *out, int32_t tileHeight,
                     int32_t tileWidth) {
  printf("passThroughTile BIT_WIDTH\n");
  passThrough_aie<int16_t, 32>(in, out, tileHeight, tileWidth);
}

#else // 32

void passThroughLine(int32_t *in, int32_t *out, int32_t lineWidth) {
  printf("passThroughLine BIT_WIDTH\n");
  passThrough_aie<int32_t, 16>(in, out, 1, lineWidth);
}

void passThroughTile(int32_t *in, int32_t *out, int32_t tileHeight,
                     int32_t tileWidth) {
  printf("passThroughTile BIT_WIDTH\n");
  passThrough_aie<int32_t, 16>(in, out, tileHeight, tileWidth);
}

#endif

} // extern "C"


// Assume factor is at least 16
template <typename T_in, typename T_out>
void absolute(T_in *ibuf, T_out *obuf) {
  aie::set_rounding(aie::rounding_mode::positive_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
  
  constexpr unsigned int REPEAT   = 50; // number of loops 
  constexpr unsigned int vec_length = 16;
  // constexpr unsigned int loops = N*REPEAT / vec_length;

  event0();
  T_in *__restrict input = ibuf; //input
  T_out *__restrict output = obuf; //output
  // Perform FFT:
  for (int rr=0; rr < REPEAT; rr++)
      chess_prepare_for_pipelining
      chess_loop_range(REPEAT,)
      {
      aie::vector<T_in,vec_length> in_vec = aie::load_v<vec_length>(input);
      aie::vector<T_out,vec_length> out_vec=aie::abs_square(in_vec);
      aie::store_v(output, out_vec);
      input  += vec_length;
      output += vec_length;
      }

  event1();
}

extern "C" {

void abs_int16(cint16 *a_in, int *c_out) {
  absolute<cint16, int32>(a_in, c_out);
}

} // extern "C"