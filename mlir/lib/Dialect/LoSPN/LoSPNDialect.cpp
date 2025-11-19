#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNTypes.h"
#include "mlir/IR/DialectImplementation.h"
using namespace mlir;
using namespace mlir::spn::low;

void LoSPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoSPN/LoSPNOps.cpp.inc"
      >();
  addTypes<LogType>();
}

::mlir::Type LoSPNDialect::parseType(::mlir::DialectAsmParser &parser) const {
  if (parser.parseKeyword("log") || parser.parseLess()) {
    return Type();
  }
  mlir::Type baseType;

  if (parser.parseType(baseType))
    return Type();

  if (!baseType.isa<FloatType>())
    return Type();

  if (parser.parseGreater())
    return Type();

  return LogType::get(baseType);
}
void LoSPNDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &os) const {
  os << "LogType";
}
#include "LoSPN/LoSPNOpsDialect.cpp.inc"
