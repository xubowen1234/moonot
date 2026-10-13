///|
fn barycenter_test_close(
  actual : Double,
  expected : Double,
  tolerance : Double,
) -> Bool {
  let delta = if actual > expected {
    actual - expected
  } else {
    expected - actual
  }
  delta <= tolerance
}

///|
test "symmetric two-measure barycenter lies at the midpoint" {
  let result = @moonot.regularized_barycenter(
    [[1.0, 0.0], [0.0, 1.0]],
    [1.0, 1.0],
    [[0.0, 1.0], [1.0, 0.0]],
    { epsilon: 0.2, tolerance: 1.0e-10, max_iterations: 3000, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.BarycenterTermination::Converged,
      )
      assert_true(barycenter_test_close(solution.barycenter[0], 0.5, 1.0e-8))
      assert_true(barycenter_test_close(solution.barycenter[1], 0.5, 1.0e-8))
      assert_true(solution.log_barycenter_residual <= 1.0e-10)
      assert_true(solution.maximum_transport_residual <= 1.0e-10)
      assert_true(solution.weighted_objective < 1.0e100)
    }
  }
}

///|
test "measure order does not change the weighted barycenter" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.5,
    tolerance: 1.0e-9,
    max_iterations: 3000,
  }
  let cost = [[0.0, 0.5, 2.0], [0.5, 0.0, 0.5], [2.0, 0.5, 0.0]]
  let forward = @moonot.regularized_barycenter(
    [[0.6, 0.3, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]],
    [1.0, 2.0, 3.0],
    cost,
    options,
  )
  let reversed = @moonot.regularized_barycenter(
    [[0.2, 0.5, 0.3], [0.1, 0.3, 0.6], [0.6, 0.3, 0.1]],
    [3.0, 2.0, 1.0],
    cost,
    options,
  )
  match (forward, reversed) {
    (Ok(left), Ok(right)) => {
      for i in 0..<left.barycenter.length() {
        assert_true(
          barycenter_test_close(left.barycenter[i], right.barycenter[i], 1.0e-8),
        )
      }
      assert_true(
        barycenter_test_close(
          left.weighted_objective,
          right.weighted_objective,
          1.0e-8,
        ),
      )
    }
    _ => assert_true(false)
  }
}

///|
test "zero-mass measures return a zero barycenter without iterations" {
  match
    @moonot.regularized_barycenter(
      [[0.0, 0.0], [0.0, 0.0]],
      [0.25, 0.75],
      [[0.0, 1.0], [1.0, 0.0]],
      { epsilon: 1.0, tolerance: 1.0e-8, max_iterations: 10, },
    ) {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.barycenter[0], 0.0)
      assert_eq(solution.barycenter[1], 0.0)
      assert_eq(solution.weighted_objective, 0.0)
      assert_eq(solution.iterations, 0)
      assert_true(
        solution.termination is @moonot.BarycenterTermination::Converged,
      )
    }
  }
}

///|
test "barycenter validation rejects unequal masses, dimensions, and weights" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  assert_true(
    @moonot.regularized_barycenter(
      [[0.5, 0.5], [0.2, 0.2]],
      [1.0, 1.0],
      [[0.0, 1.0], [1.0, 0.0]],
      options,
    )
    is Err(@moonot.BarycenterError::UnequalMeasureTotal(1)),
  )
  assert_true(
    @moonot.regularized_barycenter([[1.0]], [1.0], [[0.0, 1.0]], options)
    is Err(@moonot.BarycenterError::CostColumnCountMismatch(0)),
  )
  assert_true(
    @moonot.regularized_barycenter([[1.0]], [0.0], [[0.0]], options)
    is Err(@moonot.BarycenterError::NonPositiveWeight(0)),
  )
  assert_true(
    @moonot.regularized_barycenter([[1.0]], [1.0, 1.0], [[0.0]], options)
    is Err(@moonot.BarycenterError::WeightCountMismatch),
  )
}

///|
test "empty measure collections, supports, and cost rows are rejected" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  assert_true(
    @moonot.regularized_barycenter([], [], [], options)
    is Err(@moonot.BarycenterError::EmptyMeasures),
  )
  assert_true(
    @moonot.regularized_barycenter([[]], [1.0], [], options)
    is Err(@moonot.BarycenterError::EmptySupport),
  )
  assert_true(
    @moonot.regularized_barycenter([[0.5, 0.5]], [1.0], [[0.0, 1.0]], options)
    is Err(@moonot.BarycenterError::CostRowCountMismatch),
  )
}

///|
test "barycenter rejects non-finite and negative costs" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  assert_true(
    @moonot.regularized_barycenter(
      [[0.5, 0.5]],
      [1.0],
      [[0.0, nan], [1.0, 0.0]],
      options,
    )
    is Err(@moonot.BarycenterError::NonFiniteCost(0, 1)),
  )
  assert_true(
    @moonot.regularized_barycenter(
      [[0.5, 0.5]],
      [1.0],
      [[0.0, -0.01], [1.0, 0.0]],
      options,
    )
    is Err(@moonot.BarycenterError::NegativeCost(0, 1)),
  )
}

///|
test "barycenter validates finite positive solver options and iteration limits" {
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  let measures = [[1.0]]
  let weights = [1.0]
  let cost = [[0.0]]
  assert_true(
    @moonot.regularized_barycenter(measures, weights, cost, {
      epsilon: 0.0,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.BarycenterError::InvalidEpsilon),
  )
  assert_true(
    @moonot.regularized_barycenter(measures, weights, cost, {
      epsilon: infinity,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.BarycenterError::InvalidEpsilon),
  )
  assert_true(
    @moonot.regularized_barycenter(measures, weights, cost, {
      epsilon: 1.0,
      tolerance: -1.0,
      max_iterations: 10,
    })
    is Err(@moonot.BarycenterError::InvalidTolerance),
  )
  assert_true(
    @moonot.regularized_barycenter(measures, weights, cost, {
      epsilon: 1.0,
      tolerance: nan,
      max_iterations: 10,
    })
    is Err(@moonot.BarycenterError::InvalidTolerance),
  )
  assert_true(
    @moonot.regularized_barycenter(measures, weights, cost, {
      epsilon: 1.0,
      tolerance: 1.0e-8,
      max_iterations: 0,
    })
    is Err(@moonot.BarycenterError::InvalidIterationLimit),
  )
}

///|
test "barycenter rejects invalid weights and masses including overflowed totals" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  assert_true(
    @moonot.regularized_barycenter([[1.0]], [nan], [[0.0]], options)
    is Err(@moonot.BarycenterError::NonFiniteWeight(0)),
  )
  assert_true(
    @moonot.regularized_barycenter([[1.0]], [-1.0], [[0.0]], options)
    is Err(@moonot.BarycenterError::NonPositiveWeight(0)),
  )
  assert_true(
    @moonot.regularized_barycenter([[infinity]], [1.0], [[0.0]], options)
    is Err(@moonot.BarycenterError::NonFiniteMass(0, 0)),
  )
  assert_true(
    @moonot.regularized_barycenter([[-0.1]], [1.0], [[0.0]], options)
    is Err(@moonot.BarycenterError::NegativeMass(0, 0)),
  )
  let large_finite_mass = @math.exp(709.5)
  assert_true(
    @moonot.regularized_barycenter(
      [[large_finite_mass, large_finite_mass]],
      [1.0],
      [[0.0, 1.0], [1.0, 0.0]],
      options,
    )
    is Err(@moonot.BarycenterError::NonFiniteMeasureTotal(0)),
  )
}

///|
test "barycenter requires every measure to use the same support length" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  assert_true(
    @moonot.regularized_barycenter(
      [[0.5, 0.5], []],
      [1.0, 1.0],
      [[0.0, 1.0], [1.0, 0.0]],
      options,
    )
    is Err(@moonot.BarycenterError::EmptyMeasure(1)),
  )
  assert_true(
    @moonot.regularized_barycenter(
      [[0.5, 0.5], [0.5, 0.5, 0.0]],
      [1.0, 1.0],
      [[0.0, 1.0], [1.0, 0.0]],
      options,
    )
    is Err(@moonot.BarycenterError::EmptyMeasure(1)),
  )
}

///|
test "single-measure barycenter preserves a non-unit total mass" {
  let result = @moonot.regularized_barycenter(
    [[2.0, 0.0]],
    [7.0],
    [[0.0, 0.7], [0.7, 0.0]],
    { epsilon: 0.4, tolerance: 1.0e-10, max_iterations: 3000, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.barycenter.length(), 2)
      assert_true(
        barycenter_test_close(
          solution.barycenter[0] + solution.barycenter[1],
          2.0,
          1.0e-10,
        ),
      )
      assert_true(solution.barycenter[0] >= 0.0)
      assert_true(solution.barycenter[1] >= 0.0)
      assert_true(solution.weighted_objective.is_nan() is false)
      assert_true(solution.weighted_objective.is_inf() is false)
    }
  }
}

///|
test "scaling all positive barycenter weights leaves the solution unchanged" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.6,
    tolerance: 1.0e-10,
    max_iterations: 3000,
  }
  let measures = [[0.7, 0.3], [0.2, 0.8]]
  let cost = [[0.0, 0.9], [0.9, 0.0]]
  let base = @moonot.regularized_barycenter(measures, [1.0, 3.0], cost, options)
  let scaled = @moonot.regularized_barycenter(
    measures,
    [10.0, 30.0],
    cost,
    options,
  )
  match (base, scaled) {
    (Ok(left), Ok(right)) => {
      for i in 0..<left.barycenter.length() {
        assert_true(
          barycenter_test_close(left.barycenter[i], right.barycenter[i], 1.0e-9),
        )
      }
      assert_true(
        barycenter_test_close(
          left.weighted_objective,
          right.weighted_objective,
          1.0e-9,
        ),
      )
    }
    _ => assert_true(false)
  }
}

///|
test "outer barycenter iteration limit is reported without a convergence claim" {
  let result = @moonot.regularized_barycenter(
    [[0.8, 0.2], [0.3, 0.7]],
    [1.0, 1.0],
    [[0.0, 1.0], [1.0, 0.0]],
    { epsilon: 0.4, tolerance: 1.0e-14, max_iterations: 1, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.iterations, 1)
      assert_true(
        solution.termination is @moonot.BarycenterTermination::IterationLimit,
      )
      assert_true(solution.log_barycenter_residual > 1.0e-14)
      assert_true(
        barycenter_test_close(
          solution.barycenter[0] + solution.barycenter[1],
          1.0,
          1.0e-10,
        ),
      )
    }
  }
}
