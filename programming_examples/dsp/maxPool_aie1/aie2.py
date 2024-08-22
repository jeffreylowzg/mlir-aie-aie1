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
    M = 12
    N = 12
    C = 8
    input_length = M * N * C
    input_byte = input_length // 4  # chop input in 4 sub-tensors
    # output_length = input_length // 4
    output_length = input_length 
    output_byte = output_length // 4  # chop input in 4 sub-tensors

    vectorized = True
    enable_tracing = False

    if enable_tracing and sys.argv[1] == "xcvc1902":
        raise ValueError(
            "[ERROR] Trace is currently not supported with device xcvc1902"
        )

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(input_length, T.ui8())
            out_memRef_ty = T.memref(output_length, T.ui8())

            # AIE Core Function declarations
            maxpool = external_func(
            "maxpool_int8", inputs=[memRef_ty, out_memRef_ty, T.i32(), T.i32(), T.i32()]
            )

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile2 = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, out_memRef_ty)

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile2, "maxPool.o")
            def core_body():
                for _ in for_(sys.maxsize):
                    elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in.acquire(ObjectFifoPort.Consume, 1)
                    call(maxpool, [elemIn, elemOut, M, N, C])
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(input_length, T.i32())
            out_ty = T.memref(output_length, T.i32())

            @FuncOp.from_py_func(tensor_ty, out_ty)
            def sequence(A, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, input_byte])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, output_byte])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_scalar()
