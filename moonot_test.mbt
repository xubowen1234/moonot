///|
fn approximately_equal(
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
test "symmetric balanced transport converges and reports objective" {
  let result = @moonot.balanced_sinkhorn(
    [0.5, 0.5],
    [0.5, 0.5],
    [[0.0, 1.0], [1.0, 0.0]],
    { epsilon: 1.0, tolerance: 1.0e-12, max_iterations: 100, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      let diagonal = 0.5 / (1.0 + @math.exp(-1.0))
      let off_diagonal = 0.5 - diagonal
      assert_true(
        solution.termination is @moonot.SinkhornTermination::Converged,
      )
      assert_true(solution.marginal_residual <= 1.0e-12)
      assert_true(solution.iterations > 0)
      assert_true(
        approximately_equal(solution.coupling[0][0], diagonal, 1.0e-12),
      )
      assert_true(
        approximately_equal(solution.coupling[0][1], off_diagonal, 1.0e-12),
      )
      assert_true(
        approximately_equal(solution.coupling[1][0], off_diagonal, 1.0e-12),
      )
      assert_true(
        approximately_equal(solution.coupling[1][1], diagonal, 1.0e-12),
      )
      let expected_objective = 2.0 * diagonal * (@math.ln(diagonal) - 1.0) +
        2.0 * off_diagonal * (@math.ln(off_diagonal) - 1.0) +
        2.0 * off_diagonal
      assert_true(
        approximately_equal(
          solution.regularized_objective,
          expected_objective,
          1.0e-12,
        ),
      )
    }
  }
}

///|
test "zero-mass support yields exact zero rows and columns" {
  let result = @moonot.balanced_sinkhorn(
    [0.0, 0.25, 0.75],
    [0.25, 0.0, 0.75],
    [[0.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 2.0, 0.0]],
    { epsilon: 0.5, tolerance: 1.0e-12, max_iterations: 100, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.coupling[0][0], 0.0)
      assert_eq(solution.coupling[0][1], 0.0)
      assert_eq(solution.coupling[0][2], 0.0)
      assert_eq(solution.coupling[1][1], 0.0)
      assert_eq(solution.coupling[2][1], 0.0)
      assert_true(solution.marginal_residual <= 1.0e-12)
    }
  }
}

///|
test "all-zero marginals return a zero plan without iterations" {
  let result = @moonot.balanced_sinkhorn(
    [0.0, 0.0],
    [0.0, 0.0],
    [[0.0, 1.0], [1.0, 0.0]],
    { epsilon: 1.0, tolerance: 1.0e-12, max_iterations: 10, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.iterations, 0)
      assert_eq(solution.marginal_residual, 0.0)
      assert_eq(solution.regularized_objective, 0.0)
      assert_true(
        solution.termination is @moonot.SinkhornTermination::Converged,
      )
    }
  }
}

///|
test "log-domain scaling handles extreme cost-to-epsilon ratios" {
  let result = @moonot.balanced_sinkhorn(
    [0.5, 0.5],
    [0.5, 0.5],
    [[10000.0, 0.0], [0.0, 10000.0]],
    { epsilon: 0.01, tolerance: 1.0e-12, max_iterations: 20, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.SinkhornTermination::Converged,
      )
      assert_true(approximately_equal(solution.coupling[0][1], 0.5, 1.0e-12))
      assert_true(approximately_equal(solution.coupling[1][0], 0.5, 1.0e-12))
      assert_eq(solution.coupling[0][0], 0.0)
      assert_eq(solution.coupling[1][1], 0.0)
      assert_true(solution.marginal_residual <= 1.0e-12)
    }
  }
}

///|
test "iteration limit is reported with the best available diagnostics" {
  let result = @moonot.balanced_sinkhorn(
    [0.5, 0.5],
    [0.5, 0.5],
    [[0.0, 0.0], [0.0, 10.0]],
    { epsilon: 1.0, tolerance: 1.0e-15, max_iterations: 1, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.SinkhornTermination::IterationLimit,
      )
      assert_eq(solution.iterations, 1)
      assert_true(solution.marginal_residual > 1.0e-15)
    }
  }
}

///|
test "validation rejects unequal masses and malformed dimensions" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 100,
  }
  let unequal = @moonot.balanced_sinkhorn(
    [1.0, 0.0],
    [0.25, 0.25],
    [[0.0, 1.0], [1.0, 0.0]],
    options,
  )
  assert_true(unequal is Err(@moonot.SinkhornError::UnequalTotalMass))
  let bad_rows = @moonot.balanced_sinkhorn([1.0], [1.0], [], options)
  assert_true(bad_rows is Err(@moonot.SinkhornError::CostRowCountMismatch))
  let ragged = @moonot.balanced_sinkhorn(
    [0.5, 0.5],
    [0.5, 0.5],
    [[0.0], [1.0, 0.0]],
    options,
  )
  assert_true(ragged is Err(@moonot.SinkhornError::CostColumnCountMismatch(0)))
}

///|
test "validation rejects invalid masses costs and solver parameters" {
  let good_cost = [[0.0]]
  let good_options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  assert_true(
    @moonot.balanced_sinkhorn([-1.0], [1.0], good_cost, good_options)
    is Err(@moonot.SinkhornError::NegativeSourceMass(0)),
  )
  assert_true(
    @moonot.balanced_sinkhorn([1.0], [-1.0], good_cost, good_options)
    is Err(@moonot.SinkhornError::NegativeTargetMass(0)),
  )
  assert_true(
    @moonot.balanced_sinkhorn([1.0], [1.0], [[-1.0]], good_options)
    is Err(@moonot.SinkhornError::NegativeCost(0, 0)),
  )
  assert_true(
    @moonot.balanced_sinkhorn([1.0], [1.0], good_cost, {
      epsilon: 0.0,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.SinkhornError::InvalidEpsilon),
  )
  assert_true(
    @moonot.balanced_sinkhorn([1.0], [1.0], good_cost, {
      epsilon: 1.0,
      tolerance: 0.0,
      max_iterations: 10,
    })
    is Err(@moonot.SinkhornError::InvalidTolerance),
  )
  assert_true(
    @moonot.balanced_sinkhorn([1.0], [1.0], good_cost, {
      epsilon: 1.0,
      tolerance: 1.0e-8,
      max_iterations: 0,
    })
    is Err(@moonot.SinkhornError::InvalidIterationLimit),
  )
}

///|
test "rectangular transport preserves unequal support dimensions" {
  let result = @moonot.balanced_sinkhorn(
    [0.25, 0.75],
    [0.2, 0.3, 0.5],
    [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
    { epsilon: 0.75, tolerance: 1.0e-10, max_iterations: 1000, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.coupling.length(), 2)
      assert_eq(solution.coupling[0].length(), 3)
      assert_true(solution.marginal_residual <= 1.0e-10)
    }
  }
}

///|
test "large finite masses remain finite in the log-domain solver" {
  let result = @moonot.balanced_sinkhorn(
    [1.0e150, 1.0e150],
    [1.0e150, 1.0e150],
    [[0.0, 0.0], [0.0, 0.0]],
    { epsilon: 1.0, tolerance: 1.0e138, max_iterations: 10, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.SinkhornTermination::Converged,
      )
      assert_true(solution.coupling[0][0] > 0.0)
      assert_true(solution.coupling[0][0] < 1.0e150)
      assert_true(solution.marginal_residual <= 1.0e138)
    }
  }
}
