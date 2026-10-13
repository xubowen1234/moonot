///|
fn barycenter_costs_test_close(
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

///|
fn barycenter_costs_reference_entropy(mass : Double) -> Double {
  if mass <= 0.0 {
    0.0
  } else {
    mass * (@math.ln(mass) - 1.0)
  }
}

///|
fn barycenter_costs_reference_transport(
  first_mass : Double,
  second_mass : Double,
  cost : Array[Array[Double]],
  epsilon : Double,
) -> Double {
  let mut lower = 0.0
  if first_mass + second_mass > 1.0 {
    lower = first_mass + second_mass - 1.0
  }
  let mut upper = first_mass
  if second_mass < upper {
    upper = second_mass
  }
  let cost_delta = cost[0][0] - cost[0][1] - cost[1][0] + cost[1][1]
  for _ in 0..<100 {
    let t = (lower + upper) / 2.0
    let bottom_right = 1.0 - first_mass - second_mass + t
    let derivative = cost_delta +
      epsilon *
      @math.ln(t * bottom_right / ((first_mass - t) * (second_mass - t)))
    if derivative < 0.0 {
      lower = t
    } else {
      upper = t
    }
  }
  let top_left = (lower + upper) / 2.0
  let top_right = first_mass - top_left
  let bottom_left = second_mass - top_left
  let bottom_right = 1.0 - first_mass - second_mass + top_left
  cost[0][0] * top_left +
  cost[0][1] * top_right +
  cost[1][0] * bottom_left +
  cost[1][1] * bottom_right +
  epsilon *
  (
    barycenter_costs_reference_entropy(top_left) +
    barycenter_costs_reference_entropy(top_right) +
    barycenter_costs_reference_entropy(bottom_left) +
    barycenter_costs_reference_entropy(bottom_right)
  )
}

///|
fn barycenter_costs_reference_objective(
  first_mass : Double,
  measures : Array[Array[Double]],
  weights : Array[Double],
  costs : Array[Array[Array[Double]]],
  epsilon : Double,
) -> Double {
  weights[0] /
  (weights[0] + weights[1]) *
  barycenter_costs_reference_transport(
    first_mass,
    measures[0][0],
    costs[0],
    epsilon,
  ) +
  weights[1] /
  (weights[0] + weights[1]) *
  barycenter_costs_reference_transport(
    first_mass,
    measures[1][0],
    costs[1],
    epsilon,
  )
}

///|
fn barycenter_costs_reference_minimum(
  measures : Array[Array[Double]],
  weights : Array[Double],
  costs : Array[Array[Array[Double]]],
  epsilon : Double,
) -> (Double, Double) {
  let golden_ratio = 0.6180339887498949
  let mut lower = 0.0
  let mut upper = 1.0
  let mut left = upper - golden_ratio * (upper - lower)
  let mut right = lower + golden_ratio * (upper - lower)
  let mut left_value = barycenter_costs_reference_objective(
    left, measures, weights, costs, epsilon,
  )
  let mut right_value = barycenter_costs_reference_objective(
    right, measures, weights, costs, epsilon,
  )
  for _ in 0..<120 {
    if left_value < right_value {
      upper = right
      right = left
      right_value = left_value
      left = upper - golden_ratio * (upper - lower)
      left_value = barycenter_costs_reference_objective(
        left, measures, weights, costs, epsilon,
      )
    } else {
      lower = left
      left = right
      left_value = right_value
      right = lower + golden_ratio * (upper - lower)
      right_value = barycenter_costs_reference_objective(
        right, measures, weights, costs, epsilon,
      )
    }
  }
  let minimum_mass = (lower + upper) / 2.0
  (
    minimum_mass,
    barycenter_costs_reference_objective(
      minimum_mass, measures, weights, costs, epsilon,
    ),
  )
}

///|
test "heterogeneous asymmetric Dirac barycenter matches analytic solution and objective" {
  let epsilon = 1.0
  let costs = [[[0.0, 9.0], [2.0, 0.0]], [[0.0, 4.0], [3.0, 0.0]]]
  let result = @moonot.regularized_barycenter_with_costs(
    [[1.0, 0.0], [0.0, 1.0]],
    [1.0, 1.0],
    costs,
    { epsilon, tolerance: 1.0e-12, max_iterations: 100, },
  )
  let exp_one = @math.exp(1.0)
  let expected = [1.0 / (1.0 + exp_one), exp_one / (1.0 + exp_one)]
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.BarycenterTermination::Converged,
      )
      assert_true(
        barycenter_costs_test_close(
          solution.barycenter[0],
          expected[0],
          1.0e-10,
        ),
      )
      assert_true(
        barycenter_costs_test_close(
          solution.barycenter[1],
          expected[1],
          1.0e-10,
        ),
      )
      let expected_objective = expected[0] * 2.0 +
        expected[1] * 1.0 +
        epsilon * (expected[0] * @math.ln(expected[0]) - expected[0]) +
        epsilon * (expected[1] * @math.ln(expected[1]) - expected[1])
      assert_true(
        barycenter_costs_test_close(
          solution.weighted_objective,
          expected_objective,
          1.0e-9,
        ),
      )
    }
  }
}

///|
test "repeating a shared cost preserves legacy barycenter and diagnostics" {
  let measures = [[0.72, 0.28], [0.16, 0.84], [0.45, 0.55]]
  let weights = [1.0, 2.0, 1.5]
  let cost = [[0.0, 0.8], [0.8, 0.0]]
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.35,
    tolerance: 1.0e-11,
    max_iterations: 3000,
  }
  let legacy = @moonot.regularized_barycenter(measures, weights, cost, options)
  let with_costs = @moonot.regularized_barycenter_with_costs(
    measures,
    weights,
    [cost, cost, cost],
    options,
  )
  match (legacy, with_costs) {
    (Ok(left), Ok(right)) => {
      for i in 0..<left.barycenter.length() {
        assert_true(
          barycenter_costs_test_close(
            left.barycenter[i],
            right.barycenter[i],
            1.0e-12,
          ),
        )
      }
      assert_true(
        barycenter_costs_test_close(
          left.weighted_objective,
          right.weighted_objective,
          1.0e-11,
        ),
      )
      assert_true(
        barycenter_costs_test_close(
          left.maximum_transport_residual,
          right.maximum_transport_residual,
          1.0e-12,
        ),
      )
      assert_eq(left.iterations, right.iterations)
      match left.termination {
        @moonot.BarycenterTermination::Converged =>
          assert_true(
            right.termination is @moonot.BarycenterTermination::Converged,
          )
        @moonot.BarycenterTermination::IterationLimit =>
          assert_true(
            right.termination is @moonot.BarycenterTermination::IterationLimit,
          )
      }
    }
    _ => assert_true(false)
  }
}

///|
test "measure-weight-cost triples may be permuted together" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.45,
    tolerance: 1.0e-10,
    max_iterations: 4000,
  }
  let measures = [[0.7, 0.2, 0.1], [0.15, 0.65, 0.2], [0.2, 0.25, 0.55]]
  let weights = [1.0, 2.0, 3.0]
  let costs = [
    [[0.0, 0.5, 1.4], [0.8, 0.0, 0.7], [1.2, 0.4, 0.0]],
    [[0.0, 1.0, 0.2], [0.3, 0.0, 0.9], [0.5, 0.6, 0.0]],
    [[0.0, 0.4, 0.9], [0.7, 0.0, 0.6], [1.1, 0.2, 0.0]],
  ]
  let forward = @moonot.regularized_barycenter_with_costs(
    measures, weights, costs, options,
  )
  let permuted = @moonot.regularized_barycenter_with_costs(
    [measures[2], measures[0], measures[1]],
    [weights[2], weights[0], weights[1]],
    [costs[2], costs[0], costs[1]],
    options,
  )
  match (forward, permuted) {
    (Ok(left), Ok(right)) =>
      for i in 0..<left.barycenter.length() {
        assert_true(
          barycenter_costs_test_close(
            left.barycenter[i],
            right.barycenter[i],
            1.0e-9,
          ),
        )
      }
    _ => assert_true(false)
  }
}

///|
test "adding a constant to each cost shifts objective but not barycenter" {
  let measures = [[0.65, 0.35], [0.25, 0.75]]
  let weights = [1.0, 3.0]
  let costs = [[[0.0, 0.7], [1.1, 0.0]], [[0.4, 1.2], [0.3, 0.5]]]
  let shifted = [[[2.5, 3.2], [3.6, 2.5]], [[1.4, 2.2], [1.3, 1.5]]]
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.3,
    tolerance: 1.0e-11,
    max_iterations: 3000,
  }
  let original = @moonot.regularized_barycenter_with_costs(
    measures, weights, costs, options,
  )
  let translated = @moonot.regularized_barycenter_with_costs(
    measures, weights, shifted, options,
  )
  match (original, translated) {
    (Ok(left), Ok(right)) => {
      for i in 0..<left.barycenter.length() {
        assert_true(
          barycenter_costs_test_close(
            left.barycenter[i],
            right.barycenter[i],
            1.0e-9,
          ),
        )
      }
      let expected_shift = 2.5 * 0.25 + 1.0 * 0.75
      assert_true(
        barycenter_costs_test_close(
          right.weighted_objective - left.weighted_objective,
          expected_shift,
          1.0e-9,
        ),
      )
    }
    _ => assert_true(false)
  }
}

///|

///|
test "heterogeneous positive 2x2 barycenter matches independent scalar reference" {
  let measures = [[0.7, 0.3], [0.25, 0.75]]
  let weights = [2.0, 3.0]
  let costs = [[[0.2, 1.3], [0.9, 0.4]], [[0.8, 0.3], [1.5, 0.1]]]
  let epsilon = 0.4
  let (expected_first_mass, expected_objective) = barycenter_costs_reference_minimum(
    measures, weights, costs, epsilon,
  )
  let result = @moonot.regularized_barycenter_with_costs(
    measures,
    weights,
    costs,
    { epsilon, tolerance: 1.0e-12, max_iterations: 10000, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_true(
        solution.termination is @moonot.BarycenterTermination::Converged,
      )
      assert_true(
        barycenter_costs_test_close(
          solution.barycenter[0],
          expected_first_mass,
          1.0e-7,
        ),
      )
      assert_true(
        barycenter_costs_test_close(
          solution.weighted_objective,
          expected_objective,
          1.0e-7,
        ),
      )
    }
  }
}

///|
test "heterogeneous full-support costs preserve mass and report limits" {
  let result = @moonot.regularized_barycenter_with_costs(
    [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
    [2.0, 1.0],
    [
      [[0.0, 0.8, 1.7], [0.5, 0.0, 0.9], [1.2, 0.6, 0.0]],
      [[0.0, 1.3, 0.4], [0.7, 0.0, 1.1], [0.2, 0.9, 0.0]],
    ],
    { epsilon: 0.35, tolerance: 1.0e-14, max_iterations: 1, },
  )
  match result {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.iterations, 1)
      assert_true(
        solution.termination is @moonot.BarycenterTermination::IterationLimit,
      )
      let total = solution.barycenter[0] +
        solution.barycenter[1] +
        solution.barycenter[2]
      assert_true(barycenter_costs_test_close(total, 1.0, 1.0e-12))
    }
  }
}

///|
test "cost-set validation identifies count shape and invalid entries" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  let measures = [[0.5, 0.5], [0.4, 0.6]]
  let valid = [[0.0, 1.0], [1.0, 0.0]]
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      measures,
      [1.0, 1.0],
      [valid],
      options,
    )
    is Err(@moonot.BarycenterCostsError::CostCountMismatch),
  )
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      measures,
      [1.0, 1.0],
      [valid, [[0.0, 1.0]]],
      options,
    )
    is Err(@moonot.BarycenterCostsError::CostRowCountMismatch(1)),
  )
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      measures,
      [1.0, 1.0],
      [valid, [[0.0, 1.0], [1.0]]],
      options,
    )
    is Err(@moonot.BarycenterCostsError::CostColumnCountMismatch(1, 1)),
  )
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      measures,
      [1.0, 1.0],
      [valid, [[0.0, nan], [1.0, 0.0]]],
      options,
    )
    is Err(@moonot.BarycenterCostsError::NonFiniteCost(1, 0, 1)),
  )
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      measures,
      [1.0, 1.0],
      [valid, [[0.0, -0.1], [1.0, 0.0]]],
      options,
    )
    is Err(@moonot.BarycenterCostsError::NegativeCost(1, 0, 1)),
  )
}

///|
test "all-zero measures still validate every heterogeneous cost matrix" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  let zero_measures = [[0.0, 0.0], [0.0, 0.0]]
  let valid = [[0.0, 1.0], [1.0, 0.0]]
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      zero_measures,
      [1.0, 1.0],
      [valid, [[0.0], [1.0, 0.0]]],
      options,
    )
    is Err(@moonot.BarycenterCostsError::CostColumnCountMismatch(1, 0)),
  )
  match
    @moonot.regularized_barycenter_with_costs(
      zero_measures,
      [1.0, 1.0],
      [valid, valid],
      options,
    ) {
    Err(_) => assert_true(false)
    Ok(solution) => {
      assert_eq(solution.iterations, 0)
      assert_true(
        solution.termination is @moonot.BarycenterTermination::Converged,
      )
      assert_eq(solution.weighted_objective, 0.0)
      assert_eq(solution.barycenter[0], 0.0)
      assert_eq(solution.barycenter[1], 0.0)
    }
  }
}

///|
test "heterogeneous solver never mutates caller-owned inputs" {
  let measures = [[0.8, 0.2], [0.3, 0.7]]
  let weights = [2.0, 1.0]
  let costs = [[[0.0, 0.4], [1.3, 0.0]], [[0.2, 0.8], [0.5, 0.1]]]
  let _ = @moonot.regularized_barycenter_with_costs(measures, weights, costs, {
    epsilon: 0.4,
    tolerance: 1.0e-10,
    max_iterations: 2000,
  })
  assert_eq(measures[0][0], 0.8)
  assert_eq(measures[1][1], 0.7)
  assert_eq(weights[0], 2.0)
  assert_eq(costs[0][1][0], 1.3)
  assert_eq(costs[1][0][1], 0.8)
}

///|
test "heterogeneous entry wraps shared validation and rejects kernel overflow" {
  let valid = [[0.0, 1.0], [1.0, 0.0]]
  let base_options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-8,
    max_iterations: 10,
  }
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      [[0.5, 0.5], [0.5, 0.5]],
      [1.0, 1.0],
      [valid, valid],
      { epsilon: 0.0, tolerance: base_options.tolerance, max_iterations: 10, },
    )
    is Err(
      @moonot.BarycenterCostsError::InputError(
        @moonot.BarycenterError::InvalidEpsilon
      )
    ),
  )
  let huge_cost = @math.exp(700.0)
  assert_true(
    @moonot.regularized_barycenter_with_costs(
      [[1.0, 0.0], [0.0, 1.0]],
      [1.0, 1.0],
      [[[0.0, huge_cost], [0.0, 0.0]], valid],
      { epsilon: @math.exp(-700.0), tolerance: 1.0e-8, max_iterations: 10, },
    )
    is Err(@moonot.BarycenterCostsError::NumericalFailure),
  )
}
