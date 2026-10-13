///|
fn divergence_test_options() -> @moonot.SinkhornOptions {
  { epsilon: 0.6, tolerance: 1.0e-12, max_iterations: 1000, }
}

///|
fn divergence_approximately_equal(
  actual : Double,
  expected : Double,
  tolerance : Double,
) -> Bool {
  let difference = if actual > expected {
    actual - expected
  } else {
    expected - actual
  }
  difference <= tolerance
}

///|
test "value is exactly the specified combination of balanced objectives" {
  let source = [0.4, 0.6]
  let target = [0.7, 0.3]
  let cross_cost = [[0.0, 1.0], [1.0, 0.0]]
  let source_self_cost = [[0.0, 0.5], [0.5, 0.0]]
  let target_self_cost = [[0.0, 0.25], [0.25, 0.0]]
  let options = divergence_test_options()
  let cross = @moonot.balanced_sinkhorn(source, target, cross_cost, options)
  let source_self = @moonot.balanced_sinkhorn(
    source, source, source_self_cost, options,
  )
  let target_self = @moonot.balanced_sinkhorn(
    target, target, target_self_cost, options,
  )
  let expected = match (cross, source_self, target_self) {
    (Ok(xy), Ok(xx), Ok(yy)) =>
      xy.regularized_objective -
      0.5 * xx.regularized_objective -
      0.5 * yy.regularized_objective
    _ => {
      assert_true(false)
      0.0
    }
  }
  match
    @moonot.balanced_sinkhorn_divergence(
      source, target, cross_cost, source_self_cost, target_self_cost, options,
    ) {
    Err(_) => assert_true(false)
    Ok(result) => {
      assert_true(
        divergence_approximately_equal(result.value, expected, 1.0e-12),
      )
      match cross {
        Err(_) => assert_true(false)
        Ok(expected_cross) =>
          assert_eq(
            result.cross.regularized_objective,
            expected_cross.regularized_objective,
          )
      }
      match source_self {
        Err(_) => assert_true(false)
        Ok(expected_source_self) =>
          assert_eq(
            result.source_self.regularized_objective,
            expected_source_self.regularized_objective,
          )
      }
      match target_self {
        Err(_) => assert_true(false)
        Ok(expected_target_self) =>
          assert_eq(
            result.target_self.regularized_objective,
            expected_target_self.regularized_objective,
          )
      }
    }
  }
}

///|
test "identical inputs and consistent symmetric costs have zero divergence" {
  let marginal = [0.25, 0.75]
  let cost = [[0.0, 2.0], [2.0, 0.0]]
  match
    @moonot.balanced_sinkhorn_divergence(
      marginal,
      marginal,
      cost,
      cost,
      cost,
      divergence_test_options(),
    ) {
    Err(_) => assert_true(false)
    Ok(result) =>
      assert_true(divergence_approximately_equal(result.value, 0.0, 1.0e-12))
  }
}

///|
test "arbitrary symmetric cost matrices are not promised nonnegative divergence" {
  let result = @moonot.balanced_sinkhorn_divergence(
    [1.0],
    [1.0],
    [[0.0]],
    [[4.0]],
    [[4.0]],
    { epsilon: 1.0, tolerance: 1.0e-12, max_iterations: 10, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => assert_eq(solution.value, -4.0)
  }
}

///|
test "self-cost matrices must have the corresponding square dimensions" {
  let source = [0.5, 0.5]
  let target = [1.0]
  let options = divergence_test_options()
  let cross = [[0.0], [1.0]]
  let valid_target_self = [[0.0]]
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      source,
      target,
      cross,
      [[0.0]],
      valid_target_self,
      options,
    )
    is Err(@moonot.SinkhornDivergenceError::SourceSelfCostRowCountMismatch),
  )
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      source,
      target,
      cross,
      [[0.0, 0.0], [0.0]],
      valid_target_self,
      options,
    )
    is Err(
      @moonot.SinkhornDivergenceError::SourceSelfCostColumnCountMismatch(1)
    ),
  )
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [1.0],
      [0.5, 0.5],
      [[0.0, 1.0]],
      [[0.0]],
      [[0.0]],
      options,
    )
    is Err(@moonot.SinkhornDivergenceError::TargetSelfCostRowCountMismatch),
  )
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [1.0],
      [0.5, 0.5],
      [[0.0, 1.0]],
      [[0.0]],
      [[0.0, 0.0], [0.0]],
      options,
    )
    is Err(
      @moonot.SinkhornDivergenceError::TargetSelfCostColumnCountMismatch(1)
    ),
  )
}

///|
test "self-cost matrices must be symmetric within the documented tolerance" {
  let options = divergence_test_options()
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [0.5, 0.5],
      [1.0],
      [[0.0], [1.0]],
      [[0.0, 1.0], [1.1, 0.0]],
      [[0.0]],
      options,
    )
    is Err(@moonot.SinkhornDivergenceError::AsymmetricSourceSelfCost(0, 1)),
  )
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [1.0],
      [0.5, 0.5],
      [[0.0, 1.0]],
      [[0.0]],
      [[0.0, 1.0], [0.9, 0.0]],
      options,
    )
    is Err(@moonot.SinkhornDivergenceError::AsymmetricTargetSelfCost(0, 1)),
  )
  match
    @moonot.balanced_sinkhorn_divergence(
      [0.5, 0.5],
      [1.0],
      [[0.0], [1.0]],
      [[0.0, 1.0 + 5.0e-13], [1.0, 0.0]],
      [[0.0]],
      options,
    ) {
    Err(_) => assert_true(false)
    Ok(_) => assert_true(true)
  }
}

///|
test "errors from each underlying solve retain their solve context" {
  let options = divergence_test_options()
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [1.0],
      [1.0],
      [[-1.0]],
      [[0.0]],
      [[0.0]],
      options,
    )
    is Err(
      @moonot.SinkhornDivergenceError::CrossTransport(
        @moonot.SinkhornError::NegativeCost(0, 0)
      )
    ),
  )
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [1.0],
      [1.0],
      [[0.0]],
      [[-1.0]],
      [[0.0]],
      options,
    )
    is Err(
      @moonot.SinkhornDivergenceError::SourceSelfTransport(
        @moonot.SinkhornError::NegativeCost(0, 0)
      )
    ),
  )
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [1.0],
      [1.0],
      [[0.0]],
      [[0.0]],
      [[-1.0]],
      options,
    )
    is Err(
      @moonot.SinkhornDivergenceError::TargetSelfTransport(
        @moonot.SinkhornError::NegativeCost(0, 0)
      )
    ),
  )
  assert_true(
    @moonot.balanced_sinkhorn_divergence(
      [1.0],
      [2.0],
      [[0.0]],
      [[0.0]],
      [[0.0]],
      options,
    )
    is Err(
      @moonot.SinkhornDivergenceError::CrossTransport(
        @moonot.SinkhornError::UnequalTotalMass
      )
    ),
  )
}

///|
test "each component retains iteration-budget diagnostics" {
  let marginal = [0.5, 0.5]
  let cross_cost = [[0.0, 0.0], [0.0, 10.0]]
  let self_cost = [[0.0, 1.0], [1.0, 0.0]]
  match
    @moonot.balanced_sinkhorn_divergence(
      marginal,
      marginal,
      cross_cost,
      self_cost,
      self_cost,
      { epsilon: 1.0, tolerance: 1.0e-15, max_iterations: 1, },
    ) {
    Err(_) => assert_true(false)
    Ok(result) => {
      assert_true(
        result.cross.termination is @moonot.SinkhornTermination::IterationLimit,
      )
      assert_eq(result.cross.iterations, 1)
      assert_true(result.source_self.iterations <= 1)
      assert_true(result.target_self.iterations <= 1)
    }
  }
}
