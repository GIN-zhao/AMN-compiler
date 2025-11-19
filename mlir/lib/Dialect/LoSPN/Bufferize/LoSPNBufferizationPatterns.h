#ifndef LOSPNBUFFERIZATIONPATTERNS_H
#define LOSPNBUFFERIZATIONPATTERNS_H

/**
 * @file LoSPNBufferizationPatterns.h
 * @author GIN <gincodebug@gmail.com>
 * @date 2025-11-14
 * @brief Brief description of this file
 */

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace spn {
namespace low {
struct TaskBufferize : OpConversionPattern<SPNTask> {
  using OpConversionPattern<SPNTask>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SPNTask op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
struct KernelBufferize : OpConversionPattern<SPNKernel> {
  using OpConversionPattern<SPNKernel>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SPNKernel op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
struct BatchExtractBufferize : OpConversionPattern<SPNBatchExtract> {
  using OpConversionPattern<SPNBatchExtract>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SPNBatchExtract op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

static inline void
populateLoSPNBufferizationPatterns(OwningRewritePatternList &patterns,
                                   MLIRContext *context,
                                   TypeConverter &typeConverter) {
  patterns.insert<KernelBufferize, TaskBufferize>(typeConverter, context);
  patterns.insert<BatchExtractBufferize>(typeConverter, context);
}
} // namespace low
} // namespace spn
} // namespace mlir

#endif // LOSPNBUFFERIZATIONPATTERNS_H