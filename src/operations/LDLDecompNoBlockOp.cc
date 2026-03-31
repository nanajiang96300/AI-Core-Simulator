#include "LDLDecompNoBlockOp.h"

LDLDecompNoBlockOp::LDLDecompNoBlockOp(
    SimulationConfig config,
    Model* model,
    const std::string& name,
    std::map<std::string, std::string>& attributes,
    uint32_t target_core)
    : LDLDecompOp(
          config,
          model,
          name,
          [&attributes]() -> std::map<std::string, std::string>& {
            attributes["block_size"] = "1";
            attributes["pack_blocks"] = "1";
            return attributes;
          }(),
          target_core) {
  _optype = "LDLDecompNoBlockOp";
}

LDLDecompNoBlockOp::LDLDecompNoBlockOp(
    SimulationConfig config,
    MappingTable& mapping_table,
    const std::vector<uint32_t>& matrix_shape,
    uint32_t target_core)
    : LDLDecompOp(config, mapping_table, matrix_shape, target_core) {
  _optype = "LDLDecompNoBlockOp";
}
