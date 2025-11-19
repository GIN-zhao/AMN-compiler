#include "HiSPNtoLoSPN/NodePatterns.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::high;

Value ProductNodeLowering::splitProduct(
    high::ProductNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (operands.size() == 1)
    return operands[0];
  if (operands.size() == 2)
    return rewriter.create<low::SPNMul>(op.getLoc(), operands[0], operands[1]);
  auto pivot = llvm::divideCeil(operands.size(), 2);
  SmallVector<Value, 10> leftOperands;
  SmallVector<Value, 10> rightOperands;

  for (unsigned i = 0; i < operands.size(); i++) {
    if (i < pivot)
      leftOperands.push_back(operands[i]);
    else
      rightOperands.push_back(operands[i]);
  }
  auto leftTree = splitProduct(op, leftOperands, rewriter);
  auto rightTree = splitProduct(op, rightOperands, rewriter);

  return rewriter.create<low::SPNMul>(op.getLoc(), leftTree, rightTree);
}
LogicalResult ProductNodeLowering::matchAndRewriteChecked(
    high::ProductNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOp(op, {splitProduct(op, operands, rewriter)});
  return success();
}

Value SumNodeLowering::splitWeightSum(
    high::SumNode op, ArrayRef<Value> operands, ArrayRef<double> weights,
    ConversionPatternRewriter &rewriter) const {
  assert(weights.size() == operands.size() &&
         "Expecting identical numberof operands and weights");
  if (operands.size() == 1) {

    auto type = typeConverter->convertType(op.getType());
    double weight = weights[0];
    if (type.isa<low::LogType>()) {
      weight = log(weight);
    }

    auto constant = rewriter.create<low::SPNConstant>(
        op->getLoc(), type, TypeAttr::get(type),
        rewriter.getF64FloatAttr(weight));

    return rewriter.create<low::SPNMul>(op.getLoc(), operands[0], constant);

  } else {
    SmallVector<Value, 10> leftOps;
    SmallVector<Value, 10> rightOps;
    SmallVector<double, 10> leftWeights;
    SmallVector<double, 10> rightWeights;
    auto pivot = llvm::divideCeil(operands.size(), 2);

    unsigned count = 0;
    for (auto ov : llvm::zip(operands, weights)) {
      if (count < pivot) {
        leftOps.push_back(std::get<0>(ov));
        leftWeights.push_back(std::get<1>(ov));
      } else {
        rightOps.push_back(std::get<0>(ov));
        rightWeights.push_back(std::get<1>(ov));
      }
      count++;
    }
    auto leftTree = splitWeightSum(op, leftOps, leftWeights, rewriter);
    auto rightTree = splitWeightSum(op, rightOps, rightWeights, rewriter);

    return rewriter.create<low::SPNAdd>(op.getLoc(), leftTree, rightTree);
  }
}
LogicalResult SumNodeLowering::matchAndRewriteChecked(
    high::SumNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<double, 10> weights;
  for (auto w : op.weights().getValue()) {
    weights.push_back(w.cast<FloatAttr>().getValueAsDouble());
  }
  rewriter.replaceOp(op, {splitWeightSum(op, operands, weights, rewriter)});

  return success();
}

LogicalResult HistogramNodeLowering::matchAndRewriteChecked(
    high::HistogramNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // We can safely cast here, as the pattern checks for the correct type of the
  // enclosing query beforehand.
  auto supportMarginal =
      cast<JointQuery>(op.getEnclosingQuery()).supportMarginal();
  rewriter.replaceOpWithNewOp<low::SPNHistogramLeaf>(
      op, typeConverter->convertType(op.getType()), op.index(), op.buckets(),
      op.bucketCount(), supportMarginal);
  return success();
}

LogicalResult CategoricalNodeLowering::matchAndRewriteChecked(
    high::CategoricalNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // We can safely cast here, as the pattern checks for the correct type of the
  // enclosing query beforehand.
  auto supportMarginal =
      cast<JointQuery>(op.getEnclosingQuery()).supportMarginal();
  rewriter.replaceOpWithNewOp<low::SPNCategoricalLeaf>(
      op, typeConverter->convertType(op.getType()), op.index(),
      op.probabilities(), supportMarginal);
  return success();
}

LogicalResult GaussianNodeLowering::matchAndRewriteChecked(
    high::GaussianNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // We can safely cast here, as the pattern checks for the correct type of the
  // enclosing query beforehand.
  auto supportMarginal =
      cast<JointQuery>(op.getEnclosingQuery()).supportMarginal();
  rewriter.replaceOpWithNewOp<low::SPNGaussianLeaf>(
      op, typeConverter->convertType(op.getType()), op.index(), op.mean(),
      op.stddev(), supportMarginal);
  return success();
}

namespace {

bool isLogType(Type type) { return type.isa<low::LogType>(); }

} // namespace

LogicalResult RootNodeLowering::matchAndRewriteChecked(
    high::RootNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  assert(operands.size() == 1 &&
         "Expecting only a single result for a JointQuery");
  Value result = operands[0];
  if (!isLogType(result.getType())) {
    // Insert a conversion to log before returning the result.
    // Currently always uses F64 type to represent log results.
    result = rewriter.create<low::SPNLog>(op->getLoc(), operands[0].getType(),
                                          result);
  }
  rewriter.replaceOpWithNewOp<low::SPNYield>(op, result);
  return success();
}
