///|
test "unbalanced transport product retains a subnormal cost weight" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.0,
    rho_source: 0.0,
    rho_target: 0.0,
    cost_weight: 5.0e-324,
    tolerance: 1.0e-12,
    max_iterations: 100,
  }
  match @moonot.unbalanced_sinkhorn([1.0e150], [1.0e150], [[0.75]], options) {
    Err(_) => assert_true(false)
    Ok(solution) => {
      let expected = solution.coupling[0][0] * 5.0e-324 * 0.75
      let ratio = solution.transport_cost / expected
      assert_true(ratio > 0.99999999999999 && ratio < 1.00000000000001)
    }
  }
}

///|
test "unbalanced tiny factors recover a normal objective term" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.0,
    rho_source: 0.0,
    rho_target: 0.0,
    cost_weight: 1.0e-200,
    tolerance: 1.0e-12,
    max_iterations: 100,
  }
  match
    @moonot.unbalanced_sinkhorn([1.0e150], [1.0e150], [[1.0e-200]], options) {
    Err(_) => assert_true(false)
    Ok(solution) => {
      let expected = solution.coupling[0][0] * 1.0e-200 * 1.0e-200
      let ratio = solution.transport_cost / expected
      assert_true(ratio > 0.99999999999999 && ratio < 1.00000000000001)
    }
  }
}
