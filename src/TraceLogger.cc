#include "TraceLogger.h"

#include <fstream>

TraceLogger& TraceLogger::instance() {
  static TraceLogger logger;
  return logger;
}

void TraceLogger::log_event(const std::string& unit,
                            const std::string& name,
                            std::uint64_t start_cycle,
                            std::uint64_t end_cycle) {
  // Delegate to the singleton instance to centralize locking.
  TraceLogger::instance().add_event(unit, name, start_cycle, end_cycle);
}

void TraceLogger::add_event(const std::string& unit,
                            const std::string& name,
                            std::uint64_t start_cycle,
                            std::uint64_t end_cycle) {
  if (end_cycle < start_cycle) {
    // Ignore malformed events quietly; users can validate if needed.
    return;
  }

  std::lock_guard<std::mutex> lock(_mutex);
  _events.push_back(TraceEvent{.name = name,
                               .unit = unit,
                               .start_cycle = start_cycle,
                               .end_cycle = end_cycle});
}

void TraceLogger::dump_to_csv(const std::string& filename) {
  std::lock_guard<std::mutex> lock(_mutex);

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    return;  // For now, silently fail; caller can check filesystem separately.
  }

  ofs << "name,unit,start_cycle,end_cycle\n";
  for (const auto& ev : _events) {
    ofs << '"' << ev.name << '"' << ','
        << '"' << ev.unit << '"' << ','
        << ev.start_cycle << ','
        << ev.end_cycle << '\n';
  }
}
