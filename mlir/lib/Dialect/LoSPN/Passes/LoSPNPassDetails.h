#ifndef LOSPNPASSDETAILS_H
#define LOSPNPASSDETAILS_H

/**
 * @file LoSPNPassDetails.h
 * @author GIN <gincodebug@gmail.com>
 * @date 2025-11-15
 * @brief Brief description of this file
 */

#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace spn {
namespace low {
#define GEN_PASS_CLASSES
#include "LoSPN/LoSPNPasses.h.inc"
} // namespace low
} // namespace spn
} // namespace mlir

#endif // LOSPNPASSDETAILS_H