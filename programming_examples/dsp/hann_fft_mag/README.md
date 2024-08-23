<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Spectrogram Generation</ins>

This design performs the computation for the DSP processing required for Auto Modulation Recognition. The expected inputs would be quantized complex signal in int16. 
The three kernels used are:
1. Overlap-and-save + Hann Window
   Inputs:  complex int16
   Outputs: complex int16
2. FFT 32
   Inputs:  complex int16
   Outputs: complex int16
3. Magnitude (complex to real conversion)
   Inputs:  complex int16
   Outputs: int32

![image](https://github.com/user-attachments/assets/b0d09dff-be8d-4c01-848b-a9ca3b2c2537)

Per the image above, each compute tile will be used to call each 1 of these kernels.

First compute tile performs `overlap and save` of size `128*2(sample length in complex)`. The kernel produces `32*25(windows of fft length in complex)`

Second compute tile performs `FFT32` using the Vitis DSP AIE API. The kernel takes in and produces the same `32*25(windows of fft length in complex)`. The input data is in discrete time and the output is in frequency data.

Third compute tile performs `absolute` of size `32*25(windows of fft length in complex)` to produce `32*25(windows of fft length)`. This converts the complex inputs into real outputs for spectrogram generation.

To compile and run the design for VCK5000:
```
make vck5000
./test.elf
```
Verification:
For verification, spectrogram.py compares the generated output with a matlab code that does the same processing with RadioML inputs.
