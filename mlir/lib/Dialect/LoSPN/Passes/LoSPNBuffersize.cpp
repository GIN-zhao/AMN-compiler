#include "../Bufferize/LoSPNBufferizationPatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPNPassDetails.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/Bufferize.h"
#include <algorithm>
#include <memory>
using namespace mlir;
using namespace mlir::spn::low;

namespace {
struct LoSPNBufferize : public LoSPNBufferizeBase<LoSPNBufferize> {
protected:
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<LoSPNDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();

    target.addIllegalOp<SPNBatchExtract, SPNBatchCollect>();

    BufferizeTypeConverter typeConverter;

    target.addDynamicallyLegalOp<SPNTask>([&](SPNTask op) {
      if (!op.results().empty())
        return false;
      for (auto it : op.inputs()) {
        if (!typeConverter.isLegal(it.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<SPNKernel>([&](SPNKernel op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    target.addDynamicallyLegalOp<SPNReturn>([&](SPNReturn op) {
      return std::all_of(op->result_begin(), op->result_end(),
                         [&](OpResult op) {
                           return typeConverter.isLegal(op.getType()) &&
                                  !op.getType().isa<MemRefType>();
                         });
    });
    RewritePatternSet pattern(&getContext());
    mlir::spn::low::populateLoSPNBufferizationPatterns(pattern, &getContext(),
                                                       typeConverter);
    FrozenRewritePatternSet frozenPatterns(std::move(pattern));
    auto op = getOperation();
    if (failed(applyPartialConversion(op, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::spn::low::createLoSPNBufferizePass() {
  return std::make_unique<LoSPNBufferize>();
}