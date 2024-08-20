#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx

# Deciphering the command line arguments
if len(sys.argv) < 3:
    raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

if sys.argv[1] == "npu":
    dev = AIEDevice.npu
elif sys.argv[1] == "xcvc1902":
    dev = AIEDevice.xcvc1902
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

col = int(sys.argv[2])


def my_vector_scalar():
    N = 128000
    N_in_complex = N * 2

    n = 32 * 4 #128
    n_in_complex = n * 2 #256

    N_div_n = N // n #1000
    loops = n // 128 #1
    n_window = loops * (((128 - 32)//4)+1) #25
    n_out = 32 * n_window #800
    n_out_in_complex = n_out * 2 #1600
    N_out = N_div_n * n_out_in_complex #1600000
    # N_out = N_div_n * n_out #1000 x 800 = 800000

    buffer_depth = 2

    overlap_size = ((N_in_complex - 32) // 4) * 32

    vectorized = True
    enable_tracing = False

    if enable_tracing and sys.argv[1] == "xcvc1902":
        raise ValueError(
            "[ERROR] Trace is currently not supported with device xcvc1902"
        )

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(n_in_complex, T.i16())
            out_memRef_ty = T.memref(n_out_in_complex, T.i16())
            half_memRef_ty = T.memref(n_out, T.i32())

            # AIE Core Function declarations
            fft32_vec = external_func(
            "fft_vec_int16", inputs=[out_memRef_ty, out_memRef_ty]
            )
            hann_vec = external_func(
            "eltwise_mul_int16_vector", inputs=[memRef_ty, out_memRef_ty]
            )
            passThroughLine = external_func(
            "passThroughLine", inputs=[out_memRef_ty, out_memRef_ty, T.i32()]
            )
            abs_complex = external_func(
            "abs_int16", inputs=[out_memRef_ty, half_memRef_ty]
            )

            # Tile declarations
            ShimTile = tile(col, 0)
            compute_tile2_col, compute_tile2_row = col, 2
            ComputeTile2 = tile(compute_tile2_col, compute_tile2_row)
            ComputeTile3 = tile(compute_tile2_col, compute_tile2_row+1)
            ComputeTile4 = tile(compute_tile2_col, compute_tile2_row+2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
            of_tmp = object_fifo("temp", ComputeTile2, ComputeTile3, buffer_depth, out_memRef_ty)
            of_abs = object_fifo("abs", ComputeTile3, ComputeTile4, buffer_depth, out_memRef_ty)
            of_out = object_fifo("out", ComputeTile4, ShimTile, buffer_depth, half_memRef_ty)

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "fft_hann_window.o")
            def core_body():
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    for _ in for_(N_div_n):
                        elem_out = of_tmp.acquire(ObjectFifoPort.Produce, 1)
                        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                        call(hann_vec, [elem_in, elem_out])
                        of_in.release(ObjectFifoPort.Consume, 1)
                        of_tmp.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            @core(ComputeTile3, "fft_hann_window.o")
            def core_body():
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    for _ in for_(N_div_n):
                        elem_out2 = of_abs.acquire(ObjectFifoPort.Produce, 1)
                        elem_in2 = of_tmp.acquire(ObjectFifoPort.Consume, 1)
                        call(fft32_vec, [elem_in2, elem_out2])
                        of_tmp.release(ObjectFifoPort.Consume, 1)
                        of_abs.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            @core(ComputeTile4, "fft_hann_window.o")
            def core_body():
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    for _ in for_(N_div_n):
                        elem_out3 = of_out.acquire(ObjectFifoPort.Produce, 1)
                        elem_in3 = of_abs.acquire(ObjectFifoPort.Consume, 1)
                        call(abs_complex, [elem_in3, elem_out3])
                        # call(passThroughLine, [elem_in3, elem_out3, n_out_in_complex])
                        of_abs.release(ObjectFifoPort.Consume, 1)
                        of_out.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(n, T.i32())
            out_ty = T.memref(n_out, T.i32())

            @FuncOp.from_py_func(tensor_ty, out_ty)
            def sequence(A, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N_out])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N_in_complex])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_scalar()
