#pragma once

#include "LDLDecompOp.h"

class LDLDecompNoBlockOp : public LDLDecompOp {
 public:
  LDLDecompNoBlockOp(SimulationConfig config,
                     Model* model,
                     const std::string& name,
                     std::map<std::string, std::string>& attributes,
                     uint32_t target_core = 0);

  LDLDecompNoBlockOp(SimulationConfig config,
                     MappingTable& mapping_table,
                     const std::vector<uint32_t>& matrix_shape,
                     uint32_t target_core = 0);
};
