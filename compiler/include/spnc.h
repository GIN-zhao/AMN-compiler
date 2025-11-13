#ifndef SPNC_H
#define SPNC_H

/**
 * @file spnc.h
 * @author GIN <gincodebug@gmail.com>
 * @date 2025-11-01
 * @brief Brief description of this file
 */
#include <Kernel.h>
#include <map>
#include <string.h>
#include <string>
namespace spnc {
using options_t = std::map<std::string, std::string>;
class spn_compiler {
public:
  static Kernel compilerQuery(const std::string &inputFile,
                              const options_t &options);

  static Kernel isTargetSupported(const std::string &target);

  static Kernel isFeatureSupported(const std::string &feature);

  static Kernel getHostArchitechture();
};

} // namespace spnc

#endif // SPNC_H