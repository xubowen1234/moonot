///|
test "barycenter scaling state keeps measure rows isolated" {
  let state = barycenter_scaling_state(
    [[1.0, 1.0], [1.0, 1.0]],
    [0.5, 0.5],
    BarycenterCostMatrices::Shared([[0.0, -1.0], [-2.0, 0.0]]),
    [[0.0, 0.0], [10.0, 0.0]],
    2.0,
  )
  assert_true(state.log_kernel_products[0][0] < state.log_kernel_products[1][0])
  let second_measure_value = state.log_kernel_products[1][0]
  state.log_kernel_products[0][0] = -123.0
  assert_eq(state.log_kernel_products[1][0], second_measure_value)
}

///|
test "barycenter scaling state selects each heterogeneous kernel" {
  let state = barycenter_scaling_state(
    [[1.0, 1.0], [1.0, 1.0]],
    [0.5, 0.5],
    BarycenterCostMatrices::PerMeasure([
      [[0.0, -1.0], [-2.0, 0.0]],
      [[-4.0, 0.0], [0.0, -3.0]],
    ]),
    [[0.0, 0.0], [0.0, 0.0]],
    2.0,
  )
  assert_true(
    state.log_kernel_products[0][0] != state.log_kernel_products[1][0],
  )
}
