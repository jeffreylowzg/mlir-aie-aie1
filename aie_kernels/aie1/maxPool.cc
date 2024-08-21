//===- passThrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "maxpool_params.h"

inline __attribute__ ((always_inline))
v32uint8 load_u(uint8* actsIn, const unsigned int mr) {
    v32uint8 v = *(v32uint8*)actsIn;
    return v;
}

inline __attribute__ ((always_inline))
v32uint8 load_d(uint8* actsIn, const unsigned int mr) {
    v32uint8 v =*(v32uint8*)actsIn;
    return v;
}

inline __attribute__ ((always_inline))
v32uint8 load_d_skip(uint8* actsIn, const unsigned int mr, const unsigned int M) {
    v32uint8 v = *(v32uint8*)actsIn;
    return v;
}

inline __attribute__ ((always_inline))
v16uint8 load_u_s(uint8* actsIn, const unsigned int mr) {
    v16uint8 v = *(v16uint8*)actsIn;
    return v;
}

inline __attribute__ ((always_inline))
v16uint8 load_d_s(uint8* actsIn, const unsigned int mr) {
    v16uint8 v =  *(v16uint8*)actsIn;
    return v;
}

inline __attribute__ ((always_inline))
v16uint8 load_d_s_skip(uint8* actsIn, const unsigned int mr, const unsigned int M) {
    v16uint8 v =  *(v16uint8*)actsIn;
    return v;
}


inline __attribute__ ((always_inline))
void maxpool_int8_2x2(uint8* restrict actsIn,
                      uint8* restrict actsOut,
                      const unsigned int M,
                      const unsigned int N,
                      const unsigned int C) {
    // v64uint8 *restrict outPtr = (v64uint8 *)actsOut;
    // v64uint8 *restrict inPtr = (v64uint8 *)actsIn;
    // for (int i=0;i<M*N*C;i=i+64)
    // {
    //   *outPtr++ = *inPtr++;
    // } 
    const unsigned int MP_S = 2;
    const unsigned int MP_W = 2;
    uint8* actsIn_head = actsIn; 
    for(unsigned int c = 0; c < C/8; c++) { 
            actsIn = actsIn_head + c * (N*MR*8);
            for(unsigned int h = 0; h < (N-(MP_W-1)); h+=MP_S) { 
                    for(unsigned int w = 0; w < (M-(2*MP_W-1)); w+=(2*MP_S))  {
                            v32uint8 upRow = undef_v32uint8();
                            v32uint8 downRow = undef_v32uint8();
                            upRow = load_u(actsIn, MR);
                            actsIn = actsIn + (16*(MR/2));

                            if((w+2*MP_S) < (M-(2*MP_W-1))) {
                                downRow = load_d(actsIn, MR);
                                actsIn = actsIn - (16*(MR/2 - 2));
                            } else {
                                if(((M % 4) != 0) && (((M % 4) == 2) || ((M % 4) == 3))) {
                                    downRow = load_d(actsIn, MR);
                                    actsIn = actsIn - (16*(MR/2 - 2));
                                } else {
                                    downRow = load_d_skip(actsIn, MR, M);
                                    const unsigned int cond = (((M%4) == 2) || ((M%4) == 3));
                                    const unsigned int read = 4 * (M / 4) + 2 * cond;
                                    const unsigned int remaining = MR - read;
                                    actsIn = actsIn - (16*( MR/2 - 2 - remaining / 2 - MR/2));
                                }
                            }

                            v32int16 chess_storage(xa) upRowUnpacked = unpack(upRow);
                            v32int16 chess_storage(xb) downRowUnpacked = unpack(downRow);

                            v32int16 chess_storage(xd) vMax = max32(upRowUnpacked, downRowUnpacked);
                            v32int16 chess_storage(xc) max0 = max32(vMax, 0, 0x0a080200, 0x00000000, 0x3210,0, 0x0e0c0604, 0x00000000, 0x3210);

                            // window_write(actsOut, as_v16uint8(pack(ext_w(max0, 0))));
                            *(v16uint8*)actsOut = as_v16uint8(pack(ext_w(max0, 0)));
                            actsOut = actsOut + 16;
                        }

                    if((M % 4) != 0) { // still have some stuff to store
                        if(((M % 4) == 2) || ((M % 4) == 3)) { // can do a last MP with two last entries
                            // need to load 128 bits because then overlap with next row
                            v16uint8 upRow = undef_v16uint8();
                            v16uint8 downRow = undef_v16uint8();
                            upRow = load_u_s(actsIn, MR);
                            actsIn = actsIn + (16*(MR/2));
                            downRow = load_d_s_skip(actsIn, MR, M);
                            const unsigned int cond = (((M%4) == 2) || ((M%4) == 3)) && ((M%4) != 0);
                            const unsigned int read = 4 * (M / 4) + 2 * cond;
                            const unsigned int remaining = MR - read;
                            actsIn = actsIn - (16*(MR/2 - 1 - remaining / 2 - MR/2));

                            v32int16 chess_storage(xa) upRowUnpacked = concat(unpack(upRow), undef_v16int16());
                            v32int16 chess_storage(xb) downRowUnpacked = concat(unpack(downRow), undef_v16int16());

                            v32int16 chess_storage(xd) vMax = max32(upRowUnpacked, downRowUnpacked);
                            v32int16 chess_storage(xc) max0 = max32(vMax, 0, 0x0a080200, 0x00000000, 0x3210, 0, 0x0e0c0604, 0x00000000, 0x3210);

                            *(v16uint8*)actsOut = as_v16uint8(pack(ext_w(max0, 0)));
                            actsOut = actsOut + 16;
                        } // otherwise skip last pixel as cannot do maxpool anyway
                    }
                }
        }
}

extern "C" {
    void maxpool_int8(uint8* restrict actsIn,
                          uint8* restrict actsOut,
                          const unsigned int M,
                          const unsigned int N,
                          const unsigned int C)
 {
    maxpool_int8_2x2( actsIn, 
                    actsOut, 
                    M,
                    N,
                    C) ;
    }
}
