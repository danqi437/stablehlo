// StableHLO test model derived from JAX-compiled program
//
// This model exercises:
//   - stablehlo.custom_call @stablehlo.dynamic_reduce_window
//   - stablehlo.reduce, stablehlo.sort, stablehlo.gather
//   - Dynamic shape refinement passes
//
// Usage:
//   stablehlo-opt model.mlir --inline \
//     --stablehlo-refine-arguments='types=tensor<8xi64>,tensor<8xf64>' \
//     --stablehlo-refine-shapes \
//     --stablehlo-canonicalize-dynamism \
//     --stablehlo-check-shape-assertions

module @jit_wrapped_fn attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8xi64>, %arg1: tensor<8xf64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<1xi64>
    %c_1 = stablehlo.constant dense<[[7, 0]]> : tensor<1x2xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %c_3 = stablehlo.constant dense<8> : tensor<1xi32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<1.000000e-05> : tensor<f64>
    %cst_5 = stablehlo.constant dense<1.000000e-08> : tensor<f64>
    %cst_6 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %c_10 = stablehlo.constant dense<1> : tensor<1xi32>
    %cst_11 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %cst_12 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %c_13 = stablehlo.constant dense<0> : tensor<i64>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
    %c_15 = stablehlo.constant dense<8> : tensor<i64>
    %c_16 = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<8xi64>
    %1 = stablehlo.compare EQ, %arg0, %0, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    %2 = stablehlo.convert %1 : (tensor<8xi1>) -> tensor<8xi32>
    %3 = stablehlo.convert %2 : (tensor<8xi32>) -> tensor<8xi64>
    %4 = stablehlo.reduce(%3 init: %c_13) applies stablehlo.add across dimensions = [0] : (tensor<8xi64>, tensor<i64>) -> tensor<i64>
    %5 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<8xi64>
    %6 = stablehlo.compare EQ, %arg0, %5, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    %7 = stablehlo.convert %6 : (tensor<8xi1>) -> tensor<8xi32>
    %8 = stablehlo.convert %7 : (tensor<8xi32>) -> tensor<8xi64>
    %9 = stablehlo.reduce(%8 init: %c_13) applies stablehlo.add across dimensions = [0] : (tensor<8xi64>, tensor<i64>) -> tensor<i64>
    %10 = stablehlo.compare GT, %4, %c_13, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %11 = stablehlo.compare GT, %9, %c_13, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %12 = stablehlo.and %10, %11 : tensor<i1>
    %13 = stablehlo.reduce(%arg1 init: %cst_12) applies stablehlo.minimum across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
    %14 = stablehlo.reduce(%arg1 init: %cst_11) applies stablehlo.maximum across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
    %15 = stablehlo.is_finite %14 : (tensor<f64>) -> tensor<i1>
    %16 = stablehlo.subtract %13, %14 : tensor<f64>
    %17 = stablehlo.abs %16 : tensor<f64>
    %18 = stablehlo.abs %14 : tensor<f64>
    %19 = stablehlo.multiply %cst, %18 : tensor<f64>
    %20 = stablehlo.add %cst_5, %19 : tensor<f64>
    %21 = stablehlo.compare LE, %17, %20, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %22 = stablehlo.compare EQ, %13, %14, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %23 = stablehlo.and %15, %21 : tensor<i1>
    %24 = stablehlo.or %22, %23 : tensor<i1>
    %25 = stablehlo.negate %arg1 : tensor<8xf64>
    %26 = stablehlo.iota dim = 0 : tensor<8xi64>
    %27:2 = "stablehlo.sort"(%25, %26) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<i64>, %arg5: tensor<i64>):
      %69 = stablehlo.compare LT, %arg2, %arg3, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %69 : tensor<i1>
    }) : (tensor<8xf64>, tensor<8xi64>) -> (tensor<8xf64>, tensor<8xi64>)
    %28 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<8xi64>
    %29 = stablehlo.compare LT, %27#1, %28, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    %30 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i64>) -> tensor<8xi64>
    %31 = stablehlo.add %27#1, %30 : tensor<8xi64>
    %32 = stablehlo.select %29, %31, %27#1 : tensor<8xi1>, tensor<8xi64>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0] : (tensor<8xi64>) -> tensor<8x1xi64>
    %34 = "stablehlo.gather"(%arg0, %33) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<8xi64>, tensor<8x1xi64>) -> tensor<8xi64>
    %35 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<8xi64>
    %36 = stablehlo.compare EQ, %34, %35, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    %37 = stablehlo.convert %36 : (tensor<8xi1>) -> tensor<8xi32>
    %38 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%37, %c_2, %c_3, %c_10, %c_10, %c_10, %c_1) {api_version = 2 : i32, called_computations = [@reduce_window_int32_reducer]} : (tensor<8xi32>, tensor<i32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<?xi32>
    %39 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<8xi64>
    %40 = stablehlo.compare EQ, %34, %39, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    %41 = stablehlo.convert %40 : (tensor<8xi1>) -> tensor<8xi32>
    %42 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%41, %c_2, %c_3, %c_10, %c_10, %c_10, %c_1) {api_version = 2 : i32, called_computations = [@reduce_window_int32_reducer]} : (tensor<8xi32>, tensor<i32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<?xi32>
    %43 = stablehlo.convert %38 : (tensor<?xi32>) -> tensor<?xf64>
    %44 = stablehlo.convert %4 : (tensor<i64>) -> tensor<f64>
    %45 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<f64>) -> tensor<8xf64>
    %46 = stablehlo.divide %43, %45 : (tensor<?xf64>, tensor<8xf64>) -> tensor<8xf64>
    %47 = stablehlo.convert %42 : (tensor<?xi32>) -> tensor<?xf64>
    %48 = stablehlo.convert %9 : (tensor<i64>) -> tensor<f64>
    %49 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<f64>) -> tensor<8xf64>
    %50 = stablehlo.divide %47, %49 : (tensor<?xf64>, tensor<8xf64>) -> tensor<8xf64>
    %51 = stablehlo.concatenate %cst_14, %50, dim = 0 : (tensor<1xf64>, tensor<8xf64>) -> tensor<9xf64>
    %52 = stablehlo.concatenate %cst_14, %46, dim = 0 : (tensor<1xf64>, tensor<8xf64>) -> tensor<9xf64>
    %53 = "stablehlo.gather"(%51, %c_0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 8>}> : (tensor<9xf64>, tensor<1xi64>) -> tensor<8xf64>
    %54 = "stablehlo.gather"(%51, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 8>}> : (tensor<9xf64>, tensor<1xi64>) -> tensor<8xf64>
    %55 = stablehlo.subtract %53, %54 : tensor<8xf64>
    %56 = "stablehlo.gather"(%52, %c_0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 8>}> : (tensor<9xf64>, tensor<1xi64>) -> tensor<8xf64>
    %57 = "stablehlo.gather"(%52, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 8>}> : (tensor<9xf64>, tensor<1xi64>) -> tensor<8xf64>
    %58 = stablehlo.add %56, %57 : tensor<8xf64>
    %59 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f64>) -> tensor<8xf64>
    %60 = stablehlo.divide %58, %59 : tensor<8xf64>
    %61 = stablehlo.multiply %55, %60 : tensor<8xf64>
    %62 = stablehlo.reduce(%61 init: %cst_8) applies stablehlo.add across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
    %63 = stablehlo.convert %cst_8 : tensor<f64>
    %64 = stablehlo.maximum %63, %62 : tensor<f64>
    %65 = stablehlo.convert %cst_7 : tensor<f64>
    %66 = stablehlo.minimum %65, %64 : tensor<f64>
    %67 = stablehlo.select %24, %cst_6, %66 : tensor<i1>, tensor<f64>
    %68 = stablehlo.select %12, %67, %cst_8 : tensor<i1>, tensor<f64>
    return %68 : tensor<f64>
  }
  func.func @reduce_window_int32_reducer(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}