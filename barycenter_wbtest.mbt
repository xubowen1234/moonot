///|
test "barycenter scaling state keeps measure rows isolated" {
  let state = barycenter_scaling_state(
    [[1.0, 1.0], [1.0, 1.0]],
    [0.5, 0.5],
    [[0.0, -1.0], [-2.0, 0.0]],
    [[0.0, 0.0], [10.0, 0.0]],
    2.0,
  )
  assert_true(state.log_kernel_products[0][0] < state.log_kernel_products[1][0])
  let second_measure_value = state.log_kernel_products[1][0]
  state.log_kernel_products[0][0] = -123.0
  assert_eq(state.log_kernel_products[1][0], second_measure_value)
}
