#include "LoSPNBufferizationPatterns.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

mlir::LogicalResult mlir::spn::low::TaskBufferize::matchAndRewrite(
    SPNTask op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto inputMemRefTy = operands[0].getType().dyn_cast<MemRefType>();
  SmallVector<Value, 10> inputs;

  for (auto operand : operands) {
    inputs.push_back(operand);
  }

  auto index = (inputMemRefTy.isDynamicDim(0)) ? 0 : 1;
  auto batchDim =
      rewriter.create<mlir::memref::DimOp>(op->getLoc(), operands[0], index);
  SmallVector<Value, 10> dynSizes;
  dynSizes.push_back(batchDim);

  SmallVector<Value, 2> allocations;

  for (auto r : op.getResults()) {
    auto memRefType = typeConverter->convertType(r.getType());
    auto alloc = rewriter.create<mlir::memref::AllocOp>(
        op->getLoc(), memRefType, dynSizes, ValueRange{}, IntegerAttr());

    inputs.push_back(alloc);
    allocations.push_back(alloc);
  }
  auto newTask = rewriter.create<mlir::spn::low::SPNTask>(
      op.getLoc(), TypeRange{}, inputs, op.batchSize());

  auto newTaskBlock = newTask.addEntryBlock();
  rewriter.setInsertionPointToStart(newTaskBlock);

  SmallVector<Value, 2> inArgs;
  SmallVector<Value, 2> outArgs;

  auto batchIndex = newTask.getBatchIndex();
  inArgs.push_back(batchIndex);

  for (unsigned i = 0; i < inputs.size(); i++) {
    if (i < operands.size()) {
      inArgs.push_back(newTaskBlock->getArgument(i + 1));
    } else {
      outArgs.push_back(newTaskBlock->getArgument(i + 1));
    }
  }

  rewriter.mergeBlocks(&op.body().front(), newTaskBlock, inArgs);

  rewriter.setInsertionPoint(newTaskBlock->getTerminator());

  auto ret = dyn_cast<SPNReturn>(newTaskBlock->getTerminator());

  for (auto collectArgs : llvm::zip(ret.returnValues(), outArgs)) {
    auto collect =
        dyn_cast<SPNBatchCollect>(std::get<0>(collectArgs).getDefiningOp());
    auto outArg = std::get<1>(collectArgs);
    rewriter.create<low::SPNBatchWrite>(collect->getLoc(), outArg, batchIndex,
                                        collect.resultValues(),
                                        collect.transposedAttr());
    rewriter.eraseOp(collect);
  }

  rewriter.eraseOp(ret);
  rewriter.create<low::SPNReturn>(op.getLoc(), ValueRange{});
  rewriter.replaceOp(op, allocations);

  return success();
}

mlir::LogicalResult mlir::spn::low::BatchExtractBufferize::matchAndRewrite(
    SPNBatchExtract op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<SPNBatchRead>(op, operands[0], operands[1],
                                            op.staticIndex(), op.transposed());

  return success();
}

mlir::LogicalResult mlir::spn::low::KernelBufferize::matchAndRewrite(
    SPNKernel op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<Type> newInputTypes;
  unsigned numInputs = 0;
  for (auto inTy : op.getType().getInputs()) {
    newInputTypes.push_back(typeConverter->convertType(inTy));
    numInputs++;
  }

  for (auto outTy : op.getType().getResults()) {
    newInputTypes.push_back(typeConverter->convertType(outTy));
    numInputs++;
  }

  auto newKernelType =
      FunctionType::get(rewriter.getContext(), newInputTypes, TypeRange{});
  auto newKernel =
      rewriter.create<SPNKernel>(op.getLoc(), op.getName(), newKernelType);
  auto newKernelBlock = newKernel.addEntryBlock();
  rewriter.setInsertionPointToStart(newKernelBlock);

  SmallVector<Value, 5> inArgs;
  SmallVector<Value, 5> outArgs;
  unsigned count = 0;

  for (auto arg : newKernelBlock->getArguments()) {

    if (count < numInputs) {
      inArgs.push_back(arg);
    } else {
      outArgs.push_back(arg);
    }
    count++;
  }
  rewriter.mergeBlocks(&op.body().front(), newKernelBlock, inArgs);

  auto ret = dyn_cast<SPNReturn>(newKernelBlock->getTerminator());
  SmallVector<Value, 2> scalarReturns;
  unsigned outCount = 0;
  for (auto retVal : ret.getOperands()) {
    if (!typeConverter->isLegal(retVal.getType()))
      retVal = typeConverter->materializeTargetConversion(
          rewriter, ret->getLoc(), typeConverter->convertType(retVal.getType()),
          retVal);
    if (retVal.getType().isa<MemRefType>()) {
      rewriter.create<low::SPNCopy>(ret.getLoc(), retVal, outArgs[outCount++]);
    } else {
      scalarReturns.push_back(retVal);
    }
  }
  rewriter.create<low::SPNReturn>(op.getLoc(), scalarReturns);
  rewriter.eraseOp(ret);
  rewriter.eraseOp(op);

  return success();
}