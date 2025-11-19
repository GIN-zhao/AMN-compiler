#ifndef NODEPATTERNS_H
#define NODEPATTERNS_H

/**
 * @file NodePatterns.h
 * @author GIN <gincodebug@gmail.com>
 * @date 2025-11-18
 * @brief Brief description of this file
 */

#include "HiSPN/HiSPNOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace spn {
namespace {
template <typename... Q> struct QueryCheck;

template <> struct QueryCheck<> {
  static bool queryCheck(Operation *op) { return false; }
};

template <typename Q, typename... Qs> struct QueryCheck<Q, Qs...> {
  static bool queryCheck(Operation *op) {
    return isa<Q>(op) || QueryCheck<Qs...>::queryCheck(op);
  }
};

template <typename... Qs> bool checkQuery(Operation *op) {
  return QueryCheck<Qs...>::queryCheck(op);
}

} // namespace

template <typename SourceOp, typename NodePattern, typename... Queries>
struct NodeLowering : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!checkQuery<Queries...>(op.getEnclosingQuery())) {
      return rewriter.notifyMatchFailure(op, "Enclosing query not match");
    }
    return static_cast<NodePattern *>(this)->matchAndRewriteChecked(
        op, operands, rewriter);
  }
};

struct ProductNodeLowering
    : public NodeLowering<high::ProductNode, ProductNodeLowering,
                          high::JointQuery> {

  LogicalResult
  matchAndRewriteChecked(high::ProductNode op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const;

  Value splitProduct(high::ProductNode op, ArrayRef<Value> operands,
                     ConversionPatternRewriter &rewriter) const;
};
struct SumNodeLowering
    : NodeLowering<high::SumNode, SumNodeLowering, high::JointQuery> {
  using NodeLowering<high::SumNode, SumNodeLowering,
                     high::JointQuery>::NodeLowering;
  LogicalResult
  matchAndRewriteChecked(high::SumNode op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const;

  Value splitWeightSum(high::SumNode op, ArrayRef<Value> operands,
                       ArrayRef<double> weights,
                       ConversionPatternRewriter &rewriter) const;
};

struct HistogramNodeLowering
    : NodeLowering<high::HistogramNode, HistogramNodeLowering,
                   high::JointQuery> {
  using NodeLowering<high::HistogramNode, HistogramNodeLowering,
                     high::JointQuery>::NodeLowering;
  LogicalResult
  matchAndRewriteChecked(high::HistogramNode op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const;
};

struct GaussianNodeLowering
    : NodeLowering<high::GaussianNode, GaussianNodeLowering, high::JointQuery> {
  using NodeLowering<high::GaussianNode, GaussianNodeLowering,
                     high::JointQuery>::NodeLowering;
  LogicalResult
  matchAndRewriteChecked(high::GaussianNode op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const;
};
struct CategoricalNodeLowering
    : NodeLowering<high::CategoricalNode, CategoricalNodeLowering,
                   high::JointQuery> {
  using NodeLowering<high::CategoricalNode, CategoricalNodeLowering,
                     high::JointQuery>::NodeLowering;
  LogicalResult
  matchAndRewriteChecked(high::CategoricalNode op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const;
};

struct RootNodeLowering
    : NodeLowering<high::RootNode, RootNodeLowering, high::JointQuery> {
  using NodeLowering<high::RootNode, RootNodeLowering,
                     high::JointQuery>::NodeLowering;
  LogicalResult
  matchAndRewriteChecked(high::RootNode op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const;
};

static inline void
populateHiSPNtoLoSPNNodePatterns(OwningRewritePatternList &patterns,
                                 MLIRContext *context,
                                 TypeConverter &typeConverter) {
  patterns.insert<SumNodeLowering, ProductNodeLowering>(typeConverter, context);
  patterns.insert<HistogramNodeLowering, CategoricalNodeLowering,
                  GaussianNodeLowering>(typeConverter, context);
  patterns.insert<RootNodeLowering>(typeConverter, context);
}
} // namespace spn
} // namespace mlir

#endif // NODEPATTERNS_H