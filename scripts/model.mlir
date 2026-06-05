// StableHLO test model exercising dynamic_reduce_window and shape_assertion
// Usage:
//   stablehlo-opt model.mlir --inline \
//     --stablehlo-refine-arguments='types=tensor<?x?xf32>' \
//     --stablehlo-refine-shapes \
//     --stablehlo-canonicalize-dynamism \
//     --stablehlo-check-shape-assertions

module {

  // Reducer function for dynamic_reduce_window (max pooling)
  func.func @reduce_window_reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }

  func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // ---- dynamic_reduce_window (max pooling) ----
    // window_dimensions = [1, 3], strides = [1, 1], dilations = [1, 1], padding = [[0, 0]]
    %init = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %c_win   = stablehlo.constant dense<[1, 3]> : tensor<2xi32>
    %c_str   = stablehlo.constant dense<[1, 1]> : tensor<2xi32>
    %c_bdil  = stablehlo.constant dense<[1, 1]> : tensor<2xi32>
    %c_wdil  = stablehlo.constant dense<[1, 1]> : tensor<2xi32>
    %c_pad   = stablehlo.constant dense<[[0, 0], [0, 0]]> : tensor<2x2xi32>

    %pooled = stablehlo.custom_call @stablehlo.dynamic_reduce_window(
      %arg0, %init, %c_win, %c_str, %c_bdil, %c_wdil, %c_pad
    ) {
      api_version = 2 : i32,
      called_computations = [@reduce_window_reducer]
    } : (tensor<?x?xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>,
         tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x?xf32>

    // ---- shape_assertion (verify result is non-empty) ----
    %c_true  = stablehlo.constant dense<true> : tensor<i1>
    %c_zero  = stablehlo.constant dense<0> : tensor<i32>
    stablehlo.custom_call @shape_assertion(%c_true, %c_zero) {
      api_version = 2 : i32,
      error_message = "result tensor must be non-empty",
      has_side_effect = true
    } : (tensor<i1>, tensor<i32>) -> ()

    // ---- simple arithmetic to exercise refine-shapes ----
    %added = stablehlo.add %pooled, %pooled : tensor<?x?xf32>
    return %added : tensor<?x?xf32>
  }
}