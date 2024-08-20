//===- test_vck5000.cpp -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <xaiengine.h>
#include <fstream>

#include "memory_allocator.h"
#include "test_library.h"

#include "aie_data_movement.cpp"
#include "aie_inc.cpp"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// constexpr int FFT_LENGTH = 4096*4;
constexpr int FFT_LENGTH = 128000;
constexpr int FFT_LENGTH_OUT = ((FFT_LENGTH)/128)*(((128-32)/4)+1)*32;

void hsa_check_status(const std::string func_name, hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char *status_string(new char[1024]);
    hsa_status_string(status, &status_string);
    std::cout << func_name << " failed: " << status_string << std::endl;
    delete[] status_string;
  } else {
    std::cout << func_name << " success" << std::endl;
  }
}

bool read_data_from_file(const std::string& filename, int16_t* data, size_t size) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  for (size_t i = 0; i < size; ++i) {
    if (!(file >> data[i])) {
      std::cerr << "Failed to read data at index " << i << std::endl;
      return false;
    }
  }

  file.close();
  return true;
}

bool write_data_to_file(const std::string& filename, const int32_t* data, size_t size) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  for (size_t i = 0; i < size; ++i) {
    file << data[i];
    if ((i + 1) % 4 == 0) {
      file << std::endl;
    } else {
      file << " ";
    }
  }

  file.close();
  return true;
}

int main(int argc, char *argv[]) {

  std::vector<hsa_queue_t *> queues;
  uint32_t aie_max_queue_size(0);

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();

  // This is going to initialize HSA, create a queue
  // and get an agent
  int ret = mlir_aie_init_device(xaie);

  if (ret) {
    std::cout << "[ERROR] Error when calling mlir_aie_init_device)"
              << std::endl;
    return -1;
  }

  // Getting access to all of the HSA agents
  std::vector<hsa_agent_t> agents = xaie->agents;

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  hsa_queue_t *q = xaie->cmd_queue;

  // Adding to our vector of queues
  queues.push_back(q);
  assert(queues.size() > 0 && "No queues were sucesfully created!");

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);
  
  
  // Allocating some device memory
  ext_mem_model_t buf0, buf1;
  int16_t *in_a = (int16_t *)mlir_aie_mem_alloc(xaie, buf0, FFT_LENGTH*2);
  int32_t *out = (int32_t *)mlir_aie_mem_alloc(xaie, buf1, FFT_LENGTH_OUT*2);

  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_sync_mem_dev(buf1);

  if (in_a == nullptr || out == nullptr) {
    std::cout << "Could not allocate in device memory" << std::endl;
    return -1;
  }
  
  // init variables
  if (!read_data_from_file("../data/fft32_r2_window/sig0_i_gaussian.txt", in_a, FFT_LENGTH*2)) {
    std::cerr << "Error reading data from file" << std::endl;
    return -1;
  }

  // Pass arguments in the order of dma_memcpys in the mlir
  invoke_data_movement(queues[0], &agents[0], out, in_a);

  int errors = 0;
  
  if (!write_data_to_file("../data/fft32_r2_window/sig0_o.txt", out, FFT_LENGTH_OUT)) {
    std::cerr << "Error writing data to file" << std::endl;
    return -1;
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);

  // Shutdown AIR and HSA
  mlir_aie_deinit_libxaie(xaie);

}

