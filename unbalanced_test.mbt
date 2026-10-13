///|
fn unbalanced_approximately_equal(
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
fn unbalanced_test_options() -> @moonot.UnbalancedSinkhornOptions {
  {
    epsilon: 2.0,
    rho_source: 1.0,
    rho_target: 3.0,
    cost_weight: 0.5,
    tolerance: 1.0e-12,
    max_iterations: 1000,
  }
}

///|
test "one cell matches the closed-form asymmetric KL-unbalanced minimizer" {
  let source = 2.0
  let target = 3.0
  let weighted_cost = 2.0
  let options = unbalanced_test_options()
  let expected_mass = @math.exp(
    (
      -weighted_cost +
      options.epsilon * @math.ln(source) +
      options.epsilon * @math.ln(target) +
      options.rho_source * @math.ln(source) +
      options.rho_target * @math.ln(target)
    ) /
    (options.epsilon + options.rho_source + options.rho_target),
  )
  let result = @moonot.unbalanced_sinkhorn([source], [target], [[4.0]], options)
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      let expected_reference_kl = expected_mass *
        (@math.ln(expected_mass / (source * target)) - 1.0) +
        source * target
      let expected_source_kl = expected_mass *
        (@math.ln(expected_mass / source) - 1.0) +
        source
      let expected_target_kl = expected_mass *
        (@math.ln(expected_mass / target) - 1.0) +
        target
      let expected_objective = weighted_cost * expected_mass +
        options.epsilon * expected_reference_kl +
        options.rho_source * expected_source_kl +
        options.rho_target * expected_target_kl
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.coupling[0][0],
          expected_mass,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.transport_cost,
          weighted_cost * expected_mass,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.reference_kl,
          expected_reference_kl,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.source_marginal_kl,
          expected_source_kl,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.target_marginal_kl,
          expected_target_kl,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.objective,
          expected_objective,
          1.0e-11,
        ),
      )
      assert_true(solution.scaling_residual <= options.tolerance)
    }
  }
}

///|
test "cost weight zero removes cost and asymmetric penalties remain distinct" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.0,
    rho_source: 0.25,
    rho_target: 4.0,
    cost_weight: 0.0,
    tolerance: 1.0e-12,
    max_iterations: 1000,
  }
  let first = @moonot.unbalanced_sinkhorn([2.0], [5.0], [[0.0]], options)
  let second = @moonot.unbalanced_sinkhorn([2.0], [5.0], [[1.0e100]], options)
  match (first, second) {
    (Ok(left), Ok(right)) => {
      assert_true(
        unbalanced_approximately_equal(
          left.coupling[0][0],
          right.coupling[0][0],
          1.0e-12,
        ),
      )
      assert_eq(left.transport_cost, 0.0)
      assert_eq(right.transport_cost, 0.0)
      let expected_mass = @math.exp(
        (
          options.epsilon * (@math.ln(2.0) + @math.ln(5.0)) +
          options.rho_source * @math.ln(2.0) +
          options.rho_target * @math.ln(5.0)
        ) /
        (options.epsilon + options.rho_source + options.rho_target),
      )
      assert_true(
        unbalanced_approximately_equal(
          left.coupling[0][0],
          expected_mass,
          1.0e-12,
        ),
      )
    }
    _ => assert_true(false)
  }
}

///|
test "zero supports stay exactly zero in a sparse rectangular plan" {
  let result = @moonot.unbalanced_sinkhorn(
    [0.0, 0.4, 0.6],
    [0.7, 0.0],
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]],
    {
      epsilon: 0.5,
      rho_source: 2.0,
      rho_target: 0.75,
      cost_weight: 1.5,
      tolerance: 1.0e-11,
      max_iterations: 1000,
    },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.coupling[0][0], 0.0)
      assert_eq(solution.coupling[0][1], 0.0)
      assert_eq(solution.coupling[1][1], 0.0)
      assert_eq(solution.coupling[2][1], 0.0)
      assert_true(solution.coupling[1][0] > 0.0)
      assert_true(solution.coupling[2][0] > 0.0)
      assert_true(solution.source_marginal_residual >= 0.0)
      assert_true(solution.target_marginal_residual >= 0.0)
      assert_true(solution.objective >= 0.0)
    }
  }
}

///|
test "a zero source produces zero coupling and KL zero-support constants" {
  let result = @moonot.unbalanced_sinkhorn(
    [0.0, 0.0],
    [2.0, 1.0],
    [[0.0, 1.0], [1.0, 0.0]],
    {
      epsilon: 1.0,
      rho_source: 2.0,
      rho_target: 0.5,
      cost_weight: 1.0,
      tolerance: 1.0e-10,
      max_iterations: 10,
    },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.iterations, 0)
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
      assert_eq(solution.coupling[0][0], 0.0)
      assert_eq(solution.coupling[1][1], 0.0)
      assert_eq(solution.reference_kl, 0.0)
      assert_eq(solution.source_marginal_kl, 0.0)
      assert_eq(solution.target_marginal_kl, 3.0)
      assert_eq(solution.objective, 1.5)
    }
  }
}

///|
test "iteration budget exhaustion returns the last plan and scaling residual" {
  let result = @moonot.unbalanced_sinkhorn(
    [1.0, 1.0],
    [1.0, 1.0],
    [[0.0, 10.0], [10.0, 0.0]],
    {
      epsilon: 1.0,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: 1.0,
      tolerance: 1.0e-15,
      max_iterations: 1,
    },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination
        is @moonot.UnbalancedSinkhornTermination::IterationLimit,
      )
      assert_eq(solution.iterations, 1)
      assert_true(solution.scaling_residual > 1.0e-15)
      assert_eq(solution.coupling.length(), 2)
      assert_true(solution.coupling[0][0] > 0.0)
    }
  }
}

///|
test "large cost ratios stay finite and suppress negligible transport" {
  let result = @moonot.unbalanced_sinkhorn(
    [1.0, 1.0],
    [1.0, 1.0],
    [[0.0, 1000.0], [1000.0, 0.0]],
    {
      epsilon: 0.1,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: 1.0,
      tolerance: 1.0e-12,
      max_iterations: 100,
    },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
      assert_eq(solution.coupling[0][1], 0.0)
      assert_eq(solution.coupling[1][0], 0.0)
      assert_true(solution.coupling[0][0] > 0.0)
      assert_true(solution.coupling[1][1] > 0.0)
      assert_true(solution.objective.is_nan() is false)
      assert_true(solution.objective.is_inf() is false)
    }
  }
}

///|
test "large finite masses use logarithmic scaling without overflow" {
  let result = @moonot.unbalanced_sinkhorn([1.0e150], [1.0e150], [[0.0]], {
    epsilon: 1.0,
    rho_source: 1.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-12,
    max_iterations: 100,
  })
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        unbalanced_approximately_equal(
          solution.coupling[0][0],
          1.0e200,
          1.0e188,
        ),
      )
      assert_true(solution.objective > 0.0)
      assert_true(solution.objective.is_inf() is false)
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
    }
  }
}

///|
test "unbalanced validation reports boundary and shape errors" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.0,
    rho_source: 0.0,
    rho_target: 0.0,
    cost_weight: 0.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  assert_true(
    @moonot.unbalanced_sinkhorn([], [1.0], [], options)
    is Err(@moonot.UnbalancedSinkhornError::EmptySourceMarginal),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [], options)
    is Err(@moonot.UnbalancedSinkhornError::CostRowCountMismatch),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0, 2.0], [[0.0]], options)
    is Err(@moonot.UnbalancedSinkhornError::CostColumnCountMismatch(0)),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([-1.0], [1.0], [[0.0]], options)
    is Err(@moonot.UnbalancedSinkhornError::NegativeSourceMass(0)),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[-1.0]], options)
    is Err(@moonot.UnbalancedSinkhornError::NegativeCost(0, 0)),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 1.0,
      rho_source: -1.0,
      rho_target: 0.0,
      cost_weight: 1.0,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidSourcePenalty),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 1.0,
      rho_source: 0.0,
      rho_target: 0.0,
      cost_weight: -1.0,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidCostWeight),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 0.0,
      rho_source: 0.0,
      rho_target: 0.0,
      cost_weight: 0.0,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidEpsilon),
  )
}

///|
test "two-sided scaling satisfies the unbalanced stationarity equations" {
  let source = [0.4, 1.2]
  let target = [0.7, 0.9, 0.2]
  let cost = [[0.2, 1.1, 0.7], [1.3, 0.1, 0.4]]
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.8,
    rho_source: 0.5,
    rho_target: 1.7,
    cost_weight: 1.3,
    tolerance: 1.0e-13,
    max_iterations: 3000,
  }
  let result = @moonot.unbalanced_sinkhorn(source, target, cost, options)
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
      let row0 = solution.coupling[0][0] +
        solution.coupling[0][1] +
        solution.coupling[0][2]
      let row1 = solution.coupling[1][0] +
        solution.coupling[1][1] +
        solution.coupling[1][2]
      let col0 = solution.coupling[0][0] + solution.coupling[1][0]
      let col1 = solution.coupling[0][1] + solution.coupling[1][1]
      let col2 = solution.coupling[0][2] + solution.coupling[1][2]
      let row_sums = [row0, row1]
      let col_sums = [col0, col1, col2]
      let mut cost_value = 0.0
      let mut reference_kl = 0.0
      for i in 0..<source.length() {
        for j in 0..<target.length() {
          let mass = solution.coupling[i][j]
          let reference = source[i] * target[j]
          let scaled_cost = options.cost_weight * cost[i][j]
          let gradient = scaled_cost +
            options.epsilon * @math.ln(mass / reference) +
            options.rho_source * @math.ln(row_sums[i] / source[i]) +
            options.rho_target * @math.ln(col_sums[j] / target[j])
          assert_true(gradient <= 1.0e-10 && gradient >= -1.0e-10)
          cost_value = cost_value + scaled_cost * mass
          reference_kl = reference_kl +
            mass * @math.ln(mass / reference) -
            mass +
            reference
        }
      }
      let source_kl = row_sums[0] * @math.ln(row_sums[0] / source[0]) -
        row_sums[0] +
        source[0] +
        row_sums[1] * @math.ln(row_sums[1] / source[1]) -
        row_sums[1] +
        source[1]
      let target_kl = col_sums[0] * @math.ln(col_sums[0] / target[0]) -
        col_sums[0] +
        target[0] +
        col_sums[1] * @math.ln(col_sums[1] / target[1]) -
        col_sums[1] +
        target[1] +
        col_sums[2] * @math.ln(col_sums[2] / target[2]) -
        col_sums[2] +
        target[2]
      let objective = cost_value +
        options.epsilon * reference_kl +
        options.rho_source * source_kl +
        options.rho_target * target_kl
      assert_true(
        unbalanced_approximately_equal(
          solution.transport_cost,
          cost_value,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.reference_kl,
          reference_kl,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.source_marginal_kl,
          source_kl,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(
          solution.target_marginal_kl,
          target_kl,
          1.0e-12,
        ),
      )
      assert_true(
        unbalanced_approximately_equal(solution.objective, objective, 1.0e-12),
      )
    }
  }
}

///|
test "all-zero support follows the zero-reference KL convention" {
  let result = @moonot.unbalanced_sinkhorn([0.0], [0.0, 0.0], [[0.0, 2.0]], {
    epsilon: 1.0,
    rho_source: 1.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-10,
    max_iterations: 10,
  })
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.iterations, 0)
      assert_eq(solution.coupling.length(), 1)
      assert_eq(solution.coupling[0].length(), 2)
      assert_eq(solution.objective, 0.0)
      assert_eq(solution.reference_kl, 0.0)
      assert_eq(solution.source_marginal_kl, 0.0)
      assert_eq(solution.target_marginal_kl, 0.0)
    }
  }
}

///|
test "non-finite data is returned as a typed validation error" {
  let options = unbalanced_test_options()
  let infinity = @math.exp(1000.0)
  assert_true(
    @moonot.unbalanced_sinkhorn([infinity], [1.0], [[0.0]], options)
    is Err(@moonot.UnbalancedSinkhornError::NonFiniteSourceMass(0)),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[infinity]], options)
    is Err(@moonot.UnbalancedSinkhornError::NonFiniteCost(0, 0)),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 1.0,
      rho_source: infinity,
      rho_target: 1.0,
      cost_weight: 1.0,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidSourcePenalty),
  )
}

///|
test "a zero target produces the zero plan and source-side I-divergence" {
  let result = @moonot.unbalanced_sinkhorn(
    [2.0, 1.0],
    [0.0, 0.0],
    [[0.0, 2.0], [3.0, 1.0]],
    {
      epsilon: 1.0,
      rho_source: 0.5,
      rho_target: 2.0,
      cost_weight: 3.0,
      tolerance: 1.0e-10,
      max_iterations: 10,
    },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.iterations, 0)
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
      assert_eq(solution.coupling[0][0], 0.0)
      assert_eq(solution.coupling[1][1], 0.0)
      assert_eq(solution.reference_kl, 0.0)
      assert_eq(solution.source_marginal_kl, 3.0)
      assert_eq(solution.target_marginal_kl, 0.0)
      assert_eq(solution.source_marginal_residual, 2.0)
      assert_eq(solution.target_marginal_residual, 0.0)
      assert_eq(solution.transport_cost, 0.0)
      assert_eq(solution.objective, 1.5)
    }
  }
}

///|
test "zero marginal penalties reduce to the Gibbs plan over the product reference" {
  let source = [2.0, 3.0]
  let target = [4.0, 5.0]
  let cost = [[0.0, 0.5], [1.0, 0.25]]
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.5,
    rho_source: 0.0,
    rho_target: 0.0,
    cost_weight: 2.0,
    tolerance: 1.0e-12,
    max_iterations: 1,
  }
  match @moonot.unbalanced_sinkhorn(source, target, cost, options) {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
      assert_eq(solution.iterations, 1)
      for i in 0..<source.length() {
        for j in 0..<target.length() {
          let expected = source[i] *
            target[j] *
            @math.exp(-options.cost_weight * cost[i][j] / options.epsilon)
          assert_true(
            unbalanced_approximately_equal(
              solution.coupling[i][j],
              expected,
              1.0e-12,
            ),
          )
        }
      }
    }
  }
}

///|
test "unbalanced validation covers target bounds, scalar limits, and numeric range" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.0,
    rho_source: 1.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  let infinity = @math.exp(1000.0)
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [], [[]], options)
    is Err(@moonot.UnbalancedSinkhornError::EmptyTargetMarginal),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [-1.0], [[0.0]], options)
    is Err(@moonot.UnbalancedSinkhornError::NegativeTargetMass(0)),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [infinity], [[0.0]], options)
    is Err(@moonot.UnbalancedSinkhornError::NonFiniteTargetMass(0)),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 1.0,
      rho_source: 1.0,
      rho_target: -1.0,
      cost_weight: 1.0,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidTargetPenalty),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 1.0,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: infinity,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidCostWeight),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 1.0,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: 1.0,
      tolerance: infinity,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidTolerance),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[0.0]], {
      epsilon: 1.0,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: 1.0,
      tolerance: 1.0e-8,
      max_iterations: 0,
    })
    is Err(@moonot.UnbalancedSinkhornError::InvalidIterationLimit),
  )
  assert_true(
    @moonot.unbalanced_sinkhorn([1.0], [1.0], [[1.0e308]], {
      epsilon: 1.0e-308,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: 1.0e308,
      tolerance: 1.0e-8,
      max_iterations: 10,
    })
    is Err(@moonot.UnbalancedSinkhornError::NumericalFailure),
  )
}
