/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZEDYNAMISMPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

struct CanonicalizeCustomCallOpPattern : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> refinements;
    if (failed(hlo::getShapeRefinements(op.getLoc(), op, refinements)))
      return rewriter.notifyMatchFailure(op, "expected valid refinements");
    auto indicesAttr =
        cast<DenseIntElementsAttr>(op->getAttr("indices_of_shape_operands"));
    DenseSet<int64_t> indices(indicesAttr.value_begin<int64_t>(),
                              indicesAttr.value_end<int64_t>());

    // Discard the indices_of_shape_operands attribute.
    // We rely on the verification logic implemented in getShapeRefinements to
    // make sure that its value is consistent with the result types.
    // In the future, when we upgrade indices_of_shape_operands from an
    // experiment to a full-fledged StableHLO feature, this logic will be moved
    // to a proper verifier.
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "indices_of_shape_operands") continue;
      if (attr.getName() == "operand_layouts") {
        // Drop the operand_layouts that correspond to indices_of_shape_operands
        ArrayAttr operandLayouts = op.getOperandLayoutsAttr();
        SmallVector<Attribute> newOperandLayouts;
        for (unsigned i = 0; i < operandLayouts.size(); ++i) {
          if (indices.contains(i)) continue;
          newOperandLayouts.push_back(operandLayouts[i]);
        }
        attr = NamedAttribute(attr.getName(),
                              rewriter.getArrayAttr(newOperandLayouts));
      }
      newAttrs.push_back(attr);
    }

    // Discard the operands that correspond to indices_of_shape_operands.
    // We rely on the verification logic implemented in getShapeRefinements to
    // make sure that: 1) these operands are static, 2) the values of these
    // operands are consistent with the result types.
    SmallVector<Value> newOperands;
    auto resultIndex = 0;
    for (auto& operand : op->getOpOperands()) {
      if (indices.contains(operand.getOperandNumber())) {
        auto resultType =
            dyn_cast<ShapedType>(op->getResult(resultIndex).getType());
        if (!resultType || !resultType.hasStaticShape())
          return rewriter.notifyMatchFailure(op,
                                             "expected static result types");
        ++resultIndex;
        continue;
      }
      newOperands.push_back(operand.get());
    }
    rewriter.replaceOpWithNewOp<CustomCallOp>(op, op.getResultTypes(),
                                              newOperands, newAttrs);
    return success();
  }
};

struct CanonicalizeDynamicReduceWindowCustomCallPattern
    : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != "stablehlo.dynamic_reduce_window")
      return rewriter.notifyMatchFailure(
          op, "not a dynamic_reduce_window custom_call");

    // Operand layout (from JAX's reduce_window lowering):
    //   [inputs..., init_values..., window_dims, strides, base_dil, win_dil,
    //   padding]
    // num_inputs == num_init_values == num_results
    unsigned numResults = op.getNumResults();
    unsigned numOperands = op.getNumOperands();
    if (numOperands != 2 * numResults + 5)
      return rewriter.notifyMatchFailure(op, "unexpected operand count");

    SmallVector<int64_t> windowDims, windowStrides, baseDilations,
        windowDilations, paddingFlat;
    if (!succeeded(
            hlo::matchInts(op.getOperands()[2 * numResults], windowDims)) ||
        !succeeded(hlo::matchInts(op.getOperands()[2 * numResults + 1],
                                  windowStrides)) ||
        !succeeded(hlo::matchInts(op.getOperands()[2 * numResults + 2],
                                  baseDilations)) ||
        !succeeded(hlo::matchInts(op.getOperands()[2 * numResults + 3],
                                  windowDilations)) ||
        !succeeded(hlo::matchInts(op.getOperands()[2 * numResults + 4],
                                  paddingFlat)))
      return rewriter.notifyMatchFailure(
          op, "expected static window parameters");

    // Look up the reducer function from called_computations.
    auto calledComps = op.getCalledComputations();
    if (calledComps.empty() || calledComps.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected exactly one called_computation");
    auto reducerSymRef = cast<FlatSymbolRefAttr>(calledComps[0]);
    auto reducerFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, reducerSymRef);
    if (!reducerFunc || reducerFunc.getBody().empty())
      return rewriter.notifyMatchFailure(op, "reducer function not found");

    // Materialize inputs/init_values into SmallVectors to own the values.
    SmallVector<Value> inputs(op.getOperands().slice(0, numResults));
    SmallVector<Value> initValues(
        op.getOperands().slice(numResults, numResults));

    int64_t rank = windowDims.size();
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank, 2}, rewriter.getI64Type()), paddingFlat);

    // Compute static result types from input shapes and window parameters.
    SmallVector<Type> resultTypes;
    for (unsigned i = 0; i < numResults; ++i) {
      auto inputType = cast<RankedTensorType>(inputs[i].getType());
      SmallVector<int64_t> outputShape;
      for (int64_t d = 0; d < rank; ++d) {
        int64_t paddedSize = inputType.getDimSize(d);
        // Apply base dilation: dilated_size = (size - 1) * base_dilation + 1
        paddedSize = (paddedSize - 1) * baseDilations[d] + 1;
        // Apply padding
        paddedSize += paddingFlat[2 * d] + paddingFlat[2 * d + 1];
        // Apply window dilation
        int64_t dilatedWindow =
            (windowDims[d] - 1) * windowDilations[d] + 1;
        // Output dimension
        int64_t outDim = (paddedSize - dilatedWindow) / windowStrides[d] + 1;
        outputShape.push_back(outDim);
      }
      resultTypes.push_back(
          RankedTensorType::get(outputShape, inputType.getElementType()));
    }

    // Create the ReduceWindowOp with explicit result types and empty body.
    auto newOp = ReduceWindowOp::create(
        rewriter, op.getLoc(), resultTypes, inputs, initValues,
        rewriter.getDenseI64ArrayAttr(windowDims),
        rewriter.getDenseI64ArrayAttr(windowStrides),
        rewriter.getDenseI64ArrayAttr(baseDilations),
        rewriter.getDenseI64ArrayAttr(windowDilations), paddingAttr);

    // Populate the body region by cloning the reducer function.
    Region& newBody = newOp.getBody();
    Block& reducerBlock = reducerFunc.getBody().front();

    SmallVector<Type> blockArgTypes;
    SmallVector<Location> blockArgLocs;
    for (auto arg : reducerBlock.getArguments()) {
      blockArgTypes.push_back(arg.getType());
      blockArgLocs.push_back(arg.getLoc());
    }
    Block* newBlock = rewriter.createBlock(&newBody, newBody.end(),
                                           blockArgTypes, blockArgLocs);

    IRMapping mapping;
    for (auto [oldArg, newArg] :
         llvm::zip(reducerBlock.getArguments(), newBlock->getArguments()))
      mapping.map(oldArg, newArg);

    rewriter.setInsertionPointToEnd(newBlock);
    for (auto& reducerOp : reducerBlock.without_terminator())
      rewriter.clone(reducerOp, mapping);

    // Replace func.return with stablehlo.return.
    auto funcReturn = cast<func::ReturnOp>(reducerBlock.getTerminator());
    SmallVector<Value> returnOperands;
    for (auto operand : funcReturn.getOperands())
      returnOperands.push_back(mapping.lookupOrDefault(operand));
    ReturnOp::create(rewriter, funcReturn.getLoc(), returnOperands);

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct CanonicalizeDynamicBroadcastInDimOpPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern discards the output_dimensions operand as well as the
    // known_expanding_dimensions and known_nonexpanding_dimensions attributes.
    // We rely on the verifier to make sure that their values are consistent
    // with the result type.
    if (!op.getOperand().getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static operand type");
    if (!succeeded(hlo::matchInts(op.getOutputDimensions())))
      return rewriter.notifyMatchFailure(op,
                                         "expected static output_dimensions");
    if (!op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op.getType(), op.getOperand(), op.getBroadcastDimensionsAttr());
    return success();
  }
};

struct CanonicalizeDynamicConvOpPattern
    : public OpRewritePattern<DynamicConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicConvOp op,
                                PatternRewriter& rewriter) const override {
    // ConvolutionOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> padding;
    if (!succeeded(hlo::matchInts(op.getPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected static padding");
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(padding.size()) / 2, 2},
                              rewriter.getI64Type()),
        padding);
    rewriter.replaceOpWithNewOp<ConvolutionOp>(
        op, op.getType(), op.getLhs(), op.getRhs(), op.getWindowStridesAttr(),
        paddingAttr, op.getLhsDilationAttr(), op.getRhsDilationAttr(),
        op.getWindowReversalAttr(), op.getDimensionNumbers(),
        op.getFeatureGroupCount(), op.getBatchGroupCount(),
        op.getPrecisionConfigAttr());
    return success();
  }
};

struct CanonicalizeDynamicGatherOpPattern
    : public OpRewritePattern<DynamicGatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicGatherOp op,
                                PatternRewriter& rewriter) const override {
    // GatherOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> sliceSizes;
    if (!succeeded(hlo::matchInts(op.getSliceSizes(), sliceSizes)))
      return rewriter.notifyMatchFailure(op, "expected static slice_sizes");
    rewriter.replaceOpWithNewOp<GatherOp>(
        op, op.getType(), op.getOperand(), op.getStartIndices(),
        op.getDimensionNumbersAttr(), rewriter.getDenseI64ArrayAttr(sliceSizes),
        op.getIndicesAreSortedAttr());
    return success();
  }
};

struct CanonicalizeDynamicIotaOpPattern
    : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicIotaOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern discards the output_shape operand. We rely on the verifier
    // to make sure that its value is consistent with result type.
    SmallVector<int64_t> outputShape;
    if (!succeeded(hlo::matchInts(op.getOutputShape(), outputShape)))
      return rewriter.notifyMatchFailure(op, "expected static output_shape");
    if (!op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<IotaOp>(op, op.getType(),
                                        op.getIotaDimension());
    return success();
  }
};

struct CanonicalizeDynamicPadOpPattern : public OpRewritePattern<DynamicPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicPadOp op,
                                PatternRewriter& rewriter) const override {
    // PadOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> edgePaddingLow, edgePaddingHigh, interiorPadding;
    if (!succeeded(hlo::matchInts(op.getEdgePaddingLow(), edgePaddingLow)))
      return rewriter.notifyMatchFailure(op, "expected static low");
    if (!succeeded(hlo::matchInts(op.getEdgePaddingHigh(), edgePaddingHigh)))
      return rewriter.notifyMatchFailure(op, "expected static high");
    if (!succeeded(hlo::matchInts(op.getInteriorPadding(), interiorPadding)))
      return rewriter.notifyMatchFailure(op, "expected static interior");
    rewriter.replaceOpWithNewOp<PadOp>(op, op.getType(), op.getOperand(),
                                       op.getPaddingValue(), edgePaddingLow,
                                       edgePaddingHigh, interiorPadding);
    return success();
  }
};

struct CanonicalizeDynamicReshapeOpPattern
    : public OpRewritePattern<DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern ignores and discards the output_shape operand. We rely on
    // the verifier to make sure that its value is consistent with result type.
    if (!succeeded(hlo::matchInts(op.getOutputShape())))
      return rewriter.notifyMatchFailure(op, "expected static output_shape");
    if (!op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
    return success();
  }
};

struct CanonicalizeRealDynamicSliceOpToDynamicSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    // DynamicSliceOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.

    // This rewrite only works for unit strides because DynamicSliceOp
    // doesn't support strides (i.e. it implicitly has unit strides).
    SmallVector<int64_t> strides;
    if (!succeeded(hlo::matchInts(op.getStrides(), strides)))
      return rewriter.notifyMatchFailure(op, "expected static strides");
    if (!llvm::all_of(strides, [&](int64_t stride) { return stride == 1; }))
      return rewriter.notifyMatchFailure(op, "expected unit strides");

    // Check that slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    if (!matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) &&
        !matchPattern(op.getLimitIndices(),
                      m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices)))
      return rewriter.notifyMatchFailure(
          op, "expected limit indices equal to start indices plus constant");

    // RealDynamicSliceOp can take tensors of integer or index element types.
    // DynamicSliceOp::slice_sizes only supports i64 element type.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<int64_t> sliceSizes;
    for (auto element : sliceSizesAttr.getValues<APInt>()) {
      sliceSizes.push_back(element.getSExtValue());
    }

    // RealDynamicSliceOp::start_indices is a 1-dimensional tensor.
    // DynamicSliceOp::start_indices is a vararg of 0-dimensional tensors.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<Value> startIndices;
    for (auto i = 0; i < static_cast<int64_t>(sliceSizes.size()); ++i) {
      auto startIndexElementType =
          op.getStartIndices().getType().getElementType();
      auto startIndex1DType = RankedTensorType::get({1}, startIndexElementType);
      auto startIndex1D = SliceOp::create(
          rewriter, op.getLoc(), startIndex1DType, op.getStartIndices(),
          ArrayRef<int64_t>{i}, ArrayRef<int64_t>{i + 1}, ArrayRef<int64_t>{1});
      auto startIndex0DType = RankedTensorType::get({}, startIndexElementType);
      auto startIndex0D = ReshapeOp::create(rewriter, op.getLoc(),
                                            startIndex0DType, startIndex1D);
      startIndices.push_back(startIndex0D);
    }

    rewriter.replaceOpWithNewOp<DynamicSliceOp>(
        op, op.getType(), op.getOperand(), startIndices, sliceSizes);
    return success();
  }
};

struct CanonicalizeRealDynamicSliceOpToSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    // SliceOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> startIndices, limitIndices, strides;
    if (!succeeded(hlo::matchInts(op.getStartIndices(), startIndices)))
      return rewriter.notifyMatchFailure(op, "expected static start");
    if (!succeeded(hlo::matchInts(op.getLimitIndices(), limitIndices)))
      return rewriter.notifyMatchFailure(op, "expected static limit");
    if (!succeeded(hlo::matchInts(op.getStrides(), strides)))
      return rewriter.notifyMatchFailure(op, "expected static strides");
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getType(), op.getOperand(),
                                         startIndices, limitIndices, strides);
    return success();
  }
};

struct StablehloCanonicalizeDynamismPass
    : public impl::StablehloCanonicalizeDynamismPassBase<
          StablehloCanonicalizeDynamismPass> {
  using StablehloCanonicalizeDynamismPassBase::
      StablehloCanonicalizeDynamismPassBase;

  LogicalResult initialize(MLIRContext* context) override {
    config.setUseTopDownTraversal(true)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive)
        .setMaxIterations(2)
        .setMaxNumRewrites(GreedyRewriteConfig::kNoLimit)
        .setStrictness(GreedyRewriteStrictness::AnyOp);

    RewritePatternSet patterns_(context);
    populateStablehloCanonicalizeDynamismPatterns(context, &patterns_);
    patterns = std::move(patterns_);

    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsGreedily(func, patterns, config))) {
      func.emitError("Failed to converge StablehloCanonicalizeDynamism in ")
          << config.getMaxIterations() << " iterations";
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

}  // namespace

void populateStablehloCanonicalizeDynamismPatterns(
    MLIRContext* context, RewritePatternSet* patterns) {
  patterns->add<CanonicalizeCustomCallOpPattern>(context);
  patterns->add<CanonicalizeDynamicReduceWindowCustomCallPattern>(context);
  patterns->add<CanonicalizeDynamicBroadcastInDimOpPattern>(context);
  patterns->add<CanonicalizeDynamicConvOpPattern>(context);
  patterns->add<CanonicalizeDynamicGatherOpPattern>(context);
  patterns->add<CanonicalizeDynamicIotaOpPattern>(context);
  patterns->add<CanonicalizeDynamicPadOpPattern>(context);
  patterns->add<CanonicalizeDynamicReshapeOpPattern>(context);
  patterns->add<CanonicalizeRealDynamicSliceOpToDynamicSliceOpPattern>(context);
  patterns->add<CanonicalizeRealDynamicSliceOpToSliceOpPattern>(context);
}

}  // namespace stablehlo
}  // namespace mlir
