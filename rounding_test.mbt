///|
fn rounding_ok(
  outcome : Result[
    @moonot.CouplingRoundingResult,
    @moonot.CouplingRoundingError,
  ],
) -> @moonot.CouplingRoundingResult {
  match outcome {
    Ok(result) => result
    Err(_) => abort("unexpected rounding error")
  }
}

///|
fn rounding_assert_close(
  actual : Double,
  expected : Double,
  tolerance : Double,
) -> Unit raise {
  let difference = if actual > expected {
    actual - expected
  } else {
    expected - actual
  }
  assert_true(difference <= tolerance)
}

///|
fn rounding_assert_feasible(
  result : @moonot.CouplingRoundingResult,
  source : Array[Double],
  target : Array[Double],
  tolerance : Double,
) -> Unit raise {
  assert_eq(result.coupling.length(), source.length())
  for row in result.coupling {
    assert_eq(row.length(), target.length())
    for value in row {
      assert_true(!value.is_nan() && !value.is_inf() && value >= 0.0)
    }
  }
  let marginals = match @moonot.coupling_marginals(result.coupling) {
    Ok(value) => value
    Err(_) => abort("invalid returned coupling")
  }
  let mut measured = 0.0
  for i in 0..<source.length() {
    let difference = if marginals.source[i] > source[i] {
      marginals.source[i] - source[i]
    } else {
      source[i] - marginals.source[i]
    }
    if difference > measured {
      measured = difference
    }
  }
  for j in 0..<target.length() {
    let difference = if marginals.target[j] > target[j] {
      marginals.target[j] - target[j]
    } else {
      target[j] - marginals.target[j]
    }
    if difference > measured {
      measured = difference
    }
  }
  assert_eq(result.marginal_residual, measured)
  assert_true(measured <= tolerance)
}

///|
test "rounding zero plan uses rank one deficits for rectangular marginals" {
  let source = [0.25, 0.75]
  let target = [0.125, 0.375, 0.5]
  let result = rounding_ok(
    @moonot.round_balanced_coupling(
      [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      source,
      target,
      1.0e-15,
    ),
  )
  assert_eq(result.coupling, [
    [0.03125, 0.09375, 0.125],
    [0.09375, 0.28125, 0.375],
  ])
  rounding_assert_feasible(result, source, target, 1.0e-15)
}

///|
test "rounding overfull and underfull plans retain clipped mass" {
  let source = [0.5, 0.5]
  let over = rounding_ok(
    @moonot.round_balanced_coupling(
      [[2.0, 0.0], [0.0, 2.0]],
      source,
      source,
      1.0e-15,
    ),
  )
  assert_eq(over.coupling, [[0.5, 0.0], [0.0, 0.5]])
  rounding_assert_feasible(over, source, source, 1.0e-15)
  let under = rounding_ok(
    @moonot.round_balanced_coupling(
      [[0.1, 0.0], [0.0, 0.1]],
      source,
      source,
      1.0e-15,
    ),
  )
  for i in 0..<2 {
    for j in 0..<2 {
      rounding_assert_close(
        under.coupling[i][j],
        if i == j {
          0.3
        } else {
          0.2
        },
        1.0e-15,
      )
    }
  }
  rounding_assert_feasible(under, source, source, 1.0e-15)
}

///|
test "rounding row clipping precedes column clipping and deficit repair" {
  let source = [0.6, 0.4]
  let target = [0.3, 0.7]
  let result = rounding_ok(
    @moonot.round_balanced_coupling(
      [[0.6, 0.4], [0.2, 0.8]],
      source,
      target,
      1.0e-15,
    ),
  )
  // Row clipping gives [[.36, .24], [.08, .32]], then column 0 clips by .3/.44.
  rounding_assert_close(result.coupling[0][0], 27.0 / 110.0, 1.0e-15)
  rounding_assert_close(result.coupling[0][1], 39.0 / 110.0, 1.0e-15)
  rounding_assert_close(result.coupling[1][0], 3.0 / 55.0, 1.0e-15)
  rounding_assert_close(result.coupling[1][1], 19.0 / 55.0, 1.0e-15)
  rounding_assert_feasible(result, source, target, 1.0e-15)
}

///|
test "rounding rectangular deterministic random plans have actual feasible residuals" {
  let mut state = 17
  for trial in 0..<60 {
    let rows = trial % 4 + 1
    let columns = trial % 5 + 1
    let source = Array::make(rows, 0.0)
    let target = Array::make(columns, 0.0)
    let mut source_total = 0.0
    let mut target_total = 0.0
    for i in 0..<rows {
      state = (state * 37 + 11) % 101
      source[i] = (state % 7 + 1).to_double()
      source_total = source_total + source[i]
    }
    for j in 0..<columns {
      state = (state * 37 + 11) % 101
      target[j] = (state % 7 + 1).to_double()
      target_total = target_total + target[j]
    }
    for i in 0..<rows {
      source[i] = source[i] / source_total
    }
    for j in 0..<columns {
      target[j] = target[j] / target_total
    }
    let plan : Array[Array[Double]] = []
    for _ in 0..<rows {
      let row = Array::make(columns, 0.0)
      for j in 0..<columns {
        state = (state * 37 + 11) % 101
        row[j] = state.to_double() / 25.0
      }
      plan.push(row)
    }
    let result = rounding_ok(
      @moonot.round_balanced_coupling(plan, source, target, 1.0e-12),
    )
    rounding_assert_feasible(result, source, target, 1.0e-12)
    let second = rounding_ok(
      @moonot.round_balanced_coupling(result.coupling, source, target, 1.0e-12),
    )
    rounding_assert_feasible(second, source, target, 1.0e-12)
    for i in 0..<rows {
      for j in 0..<columns {
        rounding_assert_close(
          second.coupling[i][j],
          result.coupling[i][j],
          1.0e-12,
        )
      }
    }
  }
}

///|
test "rounding exactly feasible plans are idempotent and returned rows are independent" {
  let plan = [[0.25, 0.25], [0.125, 0.375]]
  let result = rounding_ok(
    @moonot.round_balanced_coupling(plan, [0.5, 0.5], [0.375, 0.625], 1.0e-15),
  )
  assert_eq(result.coupling, plan)
  assert_eq(result.marginal_residual, 0.0)
  result.coupling[0][0] = 99.0
  assert_eq(plan, [[0.25, 0.25], [0.125, 0.375]])
  assert_eq(result.coupling[1][0], 0.125)
}

///|
test "rounding zero supports yield exact zero rows and columns" {
  let source = [0.0, 0.25, 0.75]
  let target = [0.4, 0.0, 0.6]
  let plan = [[7.0, 9.0, 3.0], [0.3, 0.5, 0.2], [0.0, 0.8, 0.0]]
  let result = rounding_ok(
    @moonot.round_balanced_coupling(plan, source, target, 1.0e-15),
  )
  for index in 0..<3 {
    assert_eq(result.coupling[0][index], 0.0)
    assert_eq(result.coupling[index][1], 0.0)
  }
  rounding_assert_feasible(result, source, target, 1.0e-15)
  assert_eq(plan, [[7.0, 9.0, 3.0], [0.3, 0.5, 0.2], [0.0, 0.8, 0.0]])
  assert_eq(source, [0.0, 0.25, 0.75])
  assert_eq(target, [0.4, 0.0, 0.6])
}

///|
test "rounding all zero totals returns a fresh zero plan even from enormous mass" {
  let plan = [[1.0e308, 1.0e308], [1.0e308, 1.0e308]]
  let result = rounding_ok(
    @moonot.round_balanced_coupling(plan, [0.0, 0.0], [0.0, 0.0], 1.0e-300),
  )
  assert_eq(result.coupling, [[0.0, 0.0], [0.0, 0.0]])
  assert_eq(result.marginal_residual, 0.0)
  assert_eq(plan, [[1.0e308, 1.0e308], [1.0e308, 1.0e308]])
  result.coupling[0][0] = 1.0
  assert_eq(result.coupling[1][0], 0.0)
}

///|
test "rounding scaled clipping handles overflowing row sums and tiny clip factors" {
  let large = rounding_ok(
    @moonot.round_balanced_coupling(
      [[1.0e308, 1.0e308]],
      [1.0],
      [0.5, 0.5],
      1.0e-15,
    ),
  )
  assert_eq(large.coupling, [[0.5, 0.5]])
  rounding_assert_feasible(large, [1.0], [0.5, 0.5], 1.0e-15)
  let tiny = rounding_ok(
    @moonot.round_balanced_coupling(
      [[1.0e308]],
      [1.0e-300],
      [1.0e-300],
      1.0e-310,
    ),
  )
  assert_eq(tiny.coupling, [[1.0e-300]])
  rounding_assert_feasible(tiny, [1.0e-300], [1.0e-300], 1.0e-310)
}

///|
test "rounding rank one repair avoids tiny products and tiny normalized smaller deficits" {
  let tiny = rounding_ok(
    @moonot.round_balanced_coupling([[0.0]], [1.0e-300], [1.0e-300], 1.0e-310),
  )
  assert_eq(tiny.coupling, [[1.0e-300]])
  let mixed = rounding_ok(
    @moonot.round_balanced_coupling(
      [[0.0, 0.0], [0.0, 0.0]],
      [1.0e-300, 1.0e300],
      [1.0e300, 1.0e-300],
      1.0e-310,
    ),
  )
  assert_eq(mixed.coupling, [[1.0e-300, 0.0], [1.0e300, 1.0e-300]])
  rounding_assert_feasible(
    mixed,
    [1.0e-300, 1.0e300],
    [1.0e300, 1.0e-300],
    1.0e-310,
  )
}

///|
test "rounding rank one repair avoids overflowing products at large finite totals" {
  let source = [4.0e307, 4.0e307]
  let result = rounding_ok(
    @moonot.round_balanced_coupling(
      [[0.0, 0.0], [0.0, 0.0]],
      source,
      source,
      1.0e293,
    ),
  )
  assert_eq(result.coupling, [[2.0e307, 2.0e307], [2.0e307, 2.0e307]])
  rounding_assert_feasible(result, source, source, 1.0e293)
}

///|
test "rounding accepts near balanced totals only if actual residual meets tolerance" {
  let source = [1.0]
  let target = [1.0 + 5.0e-13]
  let result = rounding_ok(
    @moonot.round_balanced_coupling([[0.0]], source, target, 1.0e-12),
  )
  assert_eq(result.coupling, [[1.0]])
  assert_eq(result.marginal_residual, target[0] - source[0])
  assert_true(result.marginal_residual > 1.0e-14)
  rounding_assert_feasible(result, source, target, 1.0e-12)
  match @moonot.round_balanced_coupling([[0.0]], source, target, 1.0e-14) {
    Err(@moonot.CouplingRoundingError::FeasibilityFailure(residual)) =>
      assert_eq(residual, result.marginal_residual)
    _ => assert_true(false)
  }
  let boundary = rounding_ok(
    @moonot.round_balanced_coupling(
      [[0.0]],
      source,
      target,
      result.marginal_residual,
    ),
  )
  assert_eq(boundary.marginal_residual, result.marginal_residual)
  assert_eq(source, [1.0])
  assert_eq(target, [1.0 + 5.0e-13])
}

///|
test "rounding rejects unequal totals without absolute scale floor or normalization" {
  for
    (source, target) in [
      (1.0, 2.0),
      (1.0, 1.0 + 2.0e-12),
      (1.0e-320, 2.0e-320),
      (0.0, 1.0e-320),
    ] {
    assert_true(
      @moonot.round_balanced_coupling([[0.0]], [source], [target], 1.0)
      is Err(@moonot.CouplingRoundingError::UnequalTotalMass),
    )
  }
}

///|
test "rounding validates empty rectangular plans and marginal dimensions" {
  assert_true(
    @moonot.round_balanced_coupling([], [], [], 1.0e-12)
    is Err(
      @moonot.CouplingRoundingError::InvalidCoupling(
        @moonot.CouplingError::EmptyCoupling
      )
    ),
  )
  assert_true(
    @moonot.round_balanced_coupling([[]], [0.0], [], 1.0e-12)
    is Err(
      @moonot.CouplingRoundingError::InvalidCoupling(
        @moonot.CouplingError::EmptyCouplingRow(0)
      )
    ),
  )
  assert_true(
    @moonot.round_balanced_coupling(
      [[0.0], [0.0, 0.0]],
      [0.0, 0.0],
      [0.0],
      1.0e-12,
    )
    is Err(
      @moonot.CouplingRoundingError::InvalidCoupling(
        @moonot.CouplingError::RaggedCouplingRow(1, 1, 2)
      )
    ),
  )
  assert_true(
    @moonot.round_balanced_coupling([[0.0]], [], [0.0], 1.0e-12)
    is Err(@moonot.CouplingRoundingError::SourceCountMismatch),
  )
  assert_true(
    @moonot.round_balanced_coupling([[0.0]], [0.0], [], 1.0e-12)
    is Err(@moonot.CouplingRoundingError::TargetCountMismatch),
  )
}

///|
test "rounding rejects negative and nonfinite inputs including zero total plans" {
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  assert_true(
    @moonot.round_balanced_coupling([[-1.0]], [0.0], [0.0], 1.0e-12)
    is Err(
      @moonot.CouplingRoundingError::InvalidCoupling(
        @moonot.CouplingError::NegativeCouplingEntry(0, 0)
      )
    ),
  )
  assert_true(
    @moonot.round_balanced_coupling([[0.0]], [-1.0], [0.0], 1.0e-12)
    is Err(@moonot.CouplingRoundingError::NegativeSourceMass(0)),
  )
  assert_true(
    @moonot.round_balanced_coupling([[0.0]], [0.0], [-1.0], 1.0e-12)
    is Err(@moonot.CouplingRoundingError::NegativeTargetMass(0)),
  )
  for value in [infinity, -infinity, nan] {
    assert_true(
      @moonot.round_balanced_coupling([[value]], [0.0], [0.0], 1.0e-12)
      is Err(
        @moonot.CouplingRoundingError::InvalidCoupling(
          @moonot.CouplingError::NonFiniteCouplingEntry(0, 0)
        )
      ),
    )
    assert_true(
      @moonot.round_balanced_coupling([[0.0]], [value], [0.0], 1.0e-12)
      is Err(@moonot.CouplingRoundingError::NonFiniteSourceMass(0)),
    )
    assert_true(
      @moonot.round_balanced_coupling([[0.0]], [0.0], [value], 1.0e-12)
      is Err(@moonot.CouplingRoundingError::NonFiniteTargetMass(0)),
    )
  }
}

///|
test "rounding rejects nonpositive nonfinite tolerance and overflowing marginal totals" {
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  for tolerance in [0.0, -1.0, infinity, -infinity, nan] {
    assert_true(
      @moonot.round_balanced_coupling([[0.0]], [0.0], [0.0], tolerance)
      is Err(@moonot.CouplingRoundingError::InvalidTolerance),
    )
  }
  assert_true(
    @moonot.round_balanced_coupling(
      [[0.0], [0.0]],
      [1.0e308, 1.0e308],
      [1.0e308],
      1.0e-12,
    )
    is Err(@moonot.CouplingRoundingError::NonFiniteSourceTotal),
  )
  assert_true(
    @moonot.round_balanced_coupling(
      [[0.0, 0.0]],
      [1.0e308],
      [1.0e308, 1.0e308],
      1.0e-12,
    )
    is Err(@moonot.CouplingRoundingError::NonFiniteTargetTotal),
  )
}

///|
test "rounding clipping preserves tiny entries when its factor is representable" {
  let source = [5.0e299]
  let target = [5.0e-301, 5.0e299]
  let result = rounding_ok(
    @moonot.round_balanced_coupling(
      [[1.0e-300, 1.0e300]],
      source,
      target,
      1.0e-310,
    ),
  )
  assert_eq(result.coupling, [[5.0e-301, 5.0e299]])
  rounding_assert_feasible(result, source, target, 1.0e-310)
}

///|
test "rounding one sided deficits do not divide by zero or hide infeasibility" {
  let slightly_larger = 1.0 + 5.0e-13
  for (source, target) in [(1.0, slightly_larger), (slightly_larger, 1.0)] {
    let result = rounding_ok(
      @moonot.round_balanced_coupling([[1.0]], [source], [target], 1.0e-12),
    )
    assert_eq(result.coupling, [[1.0]])
    assert_eq(result.marginal_residual, slightly_larger - 1.0)
    assert_true(
      @moonot.round_balanced_coupling([[1.0]], [source], [target], 1.0e-14)
      is Err(@moonot.CouplingRoundingError::FeasibilityFailure(_)),
    )
  }
}
