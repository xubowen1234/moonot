///|
test "coupling marginals preserve row and column orientation" {
  match @moonot.coupling_marginals([[0.25, 0.25], [0.1, 0.4]]) {
    Err(_) => assert_true(false)
    Ok(marginals) => {
      assert_eq(marginals.source, [0.5, 0.5])
      assert_eq(marginals.target, [0.35, 0.65])
    }
  }
}

///|
test "coupling transport cost reports the unregularized inner product" {
  match
    @moonot.coupling_transport_cost([[0.25, 0.25], [0.1, 0.4]], [
      [0.0, 2.0],
      [3.0, 4.0],
    ]) {
    Err(_) => assert_true(false)
    Ok(value) => assert_true(value > 2.399999999999 && value < 2.400000000001)
  }
}

///|
test "coupling utilities reject empty and ragged matrices" {
  assert_true(
    @moonot.coupling_marginals([]) is Err(@moonot.CouplingError::EmptyCoupling),
  )
  assert_true(
    @moonot.coupling_transport_cost([[]], [[]])
    is Err(@moonot.CouplingError::EmptyCouplingRow(0)),
  )
  assert_true(
    @moonot.coupling_marginals([[0.2], [0.3, 0.4]])
    is Err(@moonot.CouplingError::RaggedCouplingRow(1, 1, 2)),
  )
}

///|
test "coupling utilities reject non-finite and negative entries" {
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  assert_true(
    @moonot.coupling_marginals([[nan]])
    is Err(@moonot.CouplingError::NonFiniteCouplingEntry(0, 0)),
  )
  assert_true(
    @moonot.coupling_marginals([[-1.0]])
    is Err(@moonot.CouplingError::NegativeCouplingEntry(0, 0)),
  )
  assert_true(
    @moonot.coupling_transport_cost([[1.0]], [[infinity]])
    is Err(@moonot.CouplingError::NonFiniteCost(0, 0)),
  )
  assert_true(
    @moonot.coupling_transport_cost([[1.0]], [[-1.0]])
    is Err(@moonot.CouplingError::NegativeCost(0, 0)),
  )
}

///|
test "coupling transport cost validates shape and numeric overflow" {
  assert_true(
    @moonot.coupling_transport_cost([[1.0]], [[1.0], [2.0]])
    is Err(@moonot.CouplingError::CostRowCountMismatch),
  )
  assert_true(
    @moonot.coupling_transport_cost([[1.0, 2.0]], [[1.0]])
    is Err(@moonot.CouplingError::CostColumnCountMismatch(0)),
  )
  assert_true(
    @moonot.coupling_transport_cost([[1.0e308]], [[1.0e308]])
    is Err(@moonot.CouplingError::NumericalOverflow(0, 0)),
  )
  assert_true(
    @moonot.coupling_marginals([[1.0e308, 1.0e308]])
    is Err(@moonot.CouplingError::NumericalOverflow(0, 1)),
  )
}

///|
fn coupling_approximately_equal(
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
test "coupling utilities inspect a rectangular balanced solver plan" {
  let source = [0.4, 0.6]
  let target = [0.2, 0.3, 0.5]
  let cost = [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
  match
    @moonot.balanced_sinkhorn(source, target, cost, {
      epsilon: 0.4,
      tolerance: 1.0e-12,
      max_iterations: 1000,
    }) {
    Err(_) => assert_true(false)
    Ok(solution) => {
      match @moonot.coupling_marginals(solution.coupling) {
        Err(_) => assert_true(false)
        Ok(marginals) => {
          assert_true(
            coupling_approximately_equal(
              marginals.source[0],
              source[0],
              1.0e-10,
            ),
          )
          assert_true(
            coupling_approximately_equal(
              marginals.source[1],
              source[1],
              1.0e-10,
            ),
          )
          assert_true(
            coupling_approximately_equal(
              marginals.target[0],
              target[0],
              1.0e-10,
            ),
          )
          assert_true(
            coupling_approximately_equal(
              marginals.target[1],
              target[1],
              1.0e-10,
            ),
          )
          assert_true(
            coupling_approximately_equal(
              marginals.target[2],
              target[2],
              1.0e-10,
            ),
          )
        }
      }
      match @moonot.coupling_transport_cost(solution.coupling, cost) {
        Err(_) => assert_true(false)
        Ok(value) => {
          let expected = solution.coupling[0][0] * cost[0][0] +
            solution.coupling[0][1] * cost[0][1] +
            solution.coupling[0][2] * cost[0][2] +
            solution.coupling[1][0] * cost[1][0] +
            solution.coupling[1][1] * cost[1][1] +
            solution.coupling[1][2] * cost[1][2]
          assert_true(coupling_approximately_equal(value, expected, 1.0e-12))
        }
      }
    }
  }
}

///|
test "coupling utilities inspect an unbalanced solver plan" {
  let source = [0.6, 0.4]
  let target = [0.2, 0.3, 0.2]
  let cost = [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
  match
    @moonot.unbalanced_sinkhorn(source, target, cost, {
      epsilon: 0.5,
      rho_source: 1.0,
      rho_target: 2.0,
      cost_weight: 1.0,
      tolerance: 1.0e-12,
      max_iterations: 1000,
    }) {
    Err(_) => assert_true(false)
    Ok(solution) => {
      match @moonot.coupling_marginals(solution.coupling) {
        Err(_) => assert_true(false)
        Ok(marginals) => {
          let mut source_residual = 0.0
          let mut target_residual = 0.0
          for i in 0..<source.length() {
            let difference = if marginals.source[i] > source[i] {
              marginals.source[i] - source[i]
            } else {
              source[i] - marginals.source[i]
            }
            if difference > source_residual {
              source_residual = difference
            }
          }
          for j in 0..<target.length() {
            let difference = if marginals.target[j] > target[j] {
              marginals.target[j] - target[j]
            } else {
              target[j] - marginals.target[j]
            }
            if difference > target_residual {
              target_residual = difference
            }
          }
          assert_true(
            coupling_approximately_equal(
              source_residual,
              solution.source_marginal_residual,
              1.0e-12,
            ),
          )
          assert_true(
            coupling_approximately_equal(
              target_residual,
              solution.target_marginal_residual,
              1.0e-12,
            ),
          )
        }
      }
      match @moonot.coupling_transport_cost(solution.coupling, cost) {
        Err(_) => assert_true(false)
        Ok(value) =>
          assert_true(
            coupling_approximately_equal(
              value,
              solution.transport_cost,
              1.0e-12,
            ),
          )
      }
    }
  }
}

///|
test "coupling utilities report column and accumulated cost overflow" {
  assert_true(
    @moonot.coupling_marginals([[1.0e308], [1.0e308]])
    is Err(@moonot.CouplingError::NumericalOverflow(1, 0)),
  )
  assert_true(
    @moonot.coupling_transport_cost([[1.0, 1.0]], [[1.0e308, 1.0e308]])
    is Err(@moonot.CouplingError::NumericalOverflow(0, 1)),
  )
}
