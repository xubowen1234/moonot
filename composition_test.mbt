///|
fn composition_value(
  result : Result[Array[Array[Double]], @moonot.CouplingCompositionError],
) -> Array[Array[Double]] raise {
  match result {
    Ok(value) => value
    Err(_) => {
      assert_true(false)
      []
    }
  }
}

///|
fn composition_assert_close(
  actual : Double,
  expected : Double,
  tolerance : Double,
) -> Unit raise {
  let difference = if actual >= expected {
    actual - expected
  } else {
    expected - actual
  }
  assert_true(difference <= tolerance)
}

///|
test "composition analytic rectangle and both exact marginals" {
  let left = [[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]]
  let right = [[1.0, 1.0, 2.0, 0.0], [0.0, 1.0, 0.0, 1.0], [2.0, 0.0, 1.0, 1.0]]
  let composed = composition_value(
    @moonot.coupling_compose(left, right, 1.0e-12),
  )
  assert_eq(composed, [[0.25, 1.25, 0.5, 1.0], [2.75, 0.75, 2.5, 1.0]])
  match @moonot.coupling_marginals(composed) {
    Err(_) => assert_true(false)
    Ok(marginals) => {
      assert_eq(marginals.source, [3.0, 7.0])
      assert_eq(marginals.target, [3.0, 2.0, 3.0, 2.0])
    }
  }
  // Success neither rewrites the right bridge distribution nor normalizes mass.
  assert_eq(left, [[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
  assert_eq(right, [
    [1.0, 1.0, 2.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [2.0, 0.0, 1.0, 1.0],
  ])
}

///|
test "composition three stage associativity on exact nonuniform bridges" {
  let a = [[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]]
  let b = [[1.0, 1.0, 2.0, 0.0], [0.0, 1.0, 0.0, 1.0], [2.0, 0.0, 1.0, 1.0]]
  let c = [[1.5, 1.5], [2.0, 0.0], [0.75, 2.25], [0.0, 2.0]]
  let ab = composition_value(@moonot.coupling_compose(a, b, 1.0e-12))
  let bc = composition_value(@moonot.coupling_compose(b, c, 1.0e-12))
  let ab_c = composition_value(@moonot.coupling_compose(ab, c, 1.0e-12))
  let a_bc = composition_value(@moonot.coupling_compose(a, bc, 1.0e-12))
  assert_eq(ab_c, a_bc)
  assert_eq(ab_c, [[1.5, 1.5], [2.75, 4.25]])
}

///|
test "composition shared zero bridges and zero plans" {
  let left = [[1.0, 0.0, 2.0], [0.0, -0.0, 2.0]]
  let right = [[1.0, 0.0], [0.0, -0.0], [2.0, 2.0]]
  assert_eq(composition_value(@moonot.coupling_compose(left, right, 1.0e-12)), [
    [2.0, 1.0],
    [1.0, 1.0],
  ])
  assert_eq(
    composition_value(
      @moonot.coupling_compose(
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        1.0e-12,
      ),
    ),
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  )
}

///|
test "composition rejects undefined conditional even inside tolerance" {
  assert_true(
    @moonot.coupling_compose([[1.0, 0.25]], [[1.0], [0.0]], 1.0)
    is Err(@moonot.CouplingCompositionError::UndefinedBridgeConditional(1)),
  )
  assert_true(
    @moonot.coupling_compose([[5.0e-324]], [[0.0]], 1.0)
    is Err(@moonot.CouplingCompositionError::UndefinedBridgeConditional(0)),
  )
}

///|
test "composition near match preserves left mass and attains target error bound" {
  // Per-bridge errors are +0.25, -0.25; both are accepted at the boundary.
  let left = [[1.25, 0.75]]
  let right = [[1.0, 0.0], [0.0, 1.0]]
  let composed = composition_value(@moonot.coupling_compose(left, right, 0.25))
  assert_eq(composed, [[1.25, 0.75]])
  match @moonot.coupling_marginals(composed) {
    Err(_) => assert_true(false)
    Ok(marginals) => {
      assert_eq(marginals.source, [2.0])
      assert_eq(marginals.target, [1.25, 0.75])
      let l1 = marginals.target[0] - 1.0 + (1.0 - marginals.target[1])
      assert_eq(l1, 0.5) // sum of absolute bridge discrepancies
    }
  }
  assert_eq(left, [[1.25, 0.75]])
  assert_eq(right, [[1.0, 0.0], [0.0, 1.0]])
  assert_true(
    @moonot.coupling_compose(left, right, 0.249)
    is Err(@moonot.CouplingCompositionError::BridgeMassMismatch(0, 1.25, 1.0)),
  )
}

///|
test "composition target coordinate bound and unequal total masses" {
  let composed = composition_value(
    @moonot.coupling_compose([[1.25, 1.25]], [[1.0], [1.0]], 0.25),
  )
  assert_eq(composed, [[2.5]])
  // Target coordinate error is 0.5 = bridge_count * tolerance, not tolerance.
  // Left total 2.5 is preserved; right total 2 is deliberately not imposed.
  assert_eq(composed[0][0] - 2.0, 0.5)
}

///|
test "composition permits zero left bridge with small positive right mass" {
  let left = [[0.0, 1.0]]
  let right = [[0.25, 0.25], [1.0, 0.0]]
  assert_eq(composition_value(@moonot.coupling_compose(left, right, 0.5)), [
    [1.0, 0.0],
  ])
  assert_eq(left, [[0.0, 1.0]])
  assert_eq(right, [[0.25, 0.25], [1.0, 0.0]])
}

///|
test "composition computes ratios before overflowing or underflowing products" {
  assert_eq(
    composition_value(
      @moonot.coupling_compose([[1.0e308]], [[5.0e307, 5.0e307]], 1.0e-12),
    ),
    [[5.0e307, 5.0e307]],
  )
  let tiny = composition_value(
    @moonot.coupling_compose([[1.0e-300]], [[4.0e-301, 6.0e-301]], 1.0e-310),
  )
  composition_assert_close(tiny[0][0], 4.0e-301, 1.0e-315)
  composition_assert_close(tiny[0][1], 6.0e-301, 1.0e-315)
  // A very permissive finite absolute tolerance is valid. No left/right
  // bridge ratio is formed; only the right conditional is divided.
  assert_eq(
    composition_value(
      @moonot.coupling_compose([[1.0e308]], [[5.0e-324]], 1.0e308),
    ),
    [[1.0e308]],
  )
}

///|
test "composition preserves subnormal mass with a deterministic conditional" {
  assert_eq(
    composition_value(
      @moonot.coupling_compose([[5.0e-324]], [[5.0e-324, 0.0]], 5.0e-324),
    ),
    [[5.0e-324, 0.0]],
  )
}

///|
test "composition validates finite strictly positive absolute tolerance" {
  let infinity = @math.exp(1000.0)
  for
    tolerance in [
      0.0,
      -0.0,
      -1.0,
      -5.0e-324,
      infinity,
      -infinity,
      infinity - infinity,
    ] {
    assert_true(
      @moonot.coupling_compose([[1.0]], [[1.0]], tolerance)
      is Err(@moonot.CouplingCompositionError::InvalidTolerance),
    )
  }
  assert_eq(
    composition_value(@moonot.coupling_compose([[1.0]], [[1.0]], 5.0e-324)),
    [[1.0]],
  )
}

///|
test "composition rejects incompatible bridge dimension and mass" {
  assert_true(
    @moonot.coupling_compose([[1.0, 2.0]], [[1.0, 2.0]], 1.0)
    is Err(@moonot.CouplingCompositionError::BridgeDimensionMismatch(2, 1)),
  )
  assert_true(
    @moonot.coupling_compose([[1.0]], [[1.0], [2.0]], 1.0)
    is Err(@moonot.CouplingCompositionError::BridgeDimensionMismatch(1, 2)),
  )
  assert_true(
    @moonot.coupling_compose([[1.0, 2.0]], [[1.0], [3.0]], 0.5)
    is Err(@moonot.CouplingCompositionError::BridgeMassMismatch(1, 2.0, 3.0)),
  )
}

///|
test "composition rejects empty and ragged plans on either side" {
  assert_true(
    @moonot.coupling_compose([], [[1.0]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidLeft(
        @moonot.CouplingError::EmptyCoupling
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[]], [[1.0]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidLeft(
        @moonot.CouplingError::EmptyCouplingRow(0)
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[1.0], []], [[1.0]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidLeft(
        @moonot.CouplingError::RaggedCouplingRow(1, 1, 0)
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[1.0]], [], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidRight(
        @moonot.CouplingError::EmptyCoupling
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[1.0]], [[]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidRight(
        @moonot.CouplingError::EmptyCouplingRow(0)
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[1.0, 1.0]], [[1.0], [1.0, 2.0]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidRight(
        @moonot.CouplingError::RaggedCouplingRow(1, 1, 2)
      )
    ),
  )
}

///|
test "composition rejects all negative entries including subnormals" {
  for negative in [-1.0, -1.0e-300, -5.0e-324] {
    assert_true(
      @moonot.coupling_compose([[negative]], [[1.0]], 1.0)
      is Err(
        @moonot.CouplingCompositionError::InvalidLeft(
          @moonot.CouplingError::NegativeCouplingEntry(0, 0)
        )
      ),
    )
    assert_true(
      @moonot.coupling_compose([[1.0]], [[negative]], 1.0)
      is Err(
        @moonot.CouplingCompositionError::InvalidRight(
          @moonot.CouplingError::NegativeCouplingEntry(0, 0)
        )
      ),
    )
  }
}

///|
test "composition rejects nonfinite entries on both sides" {
  let infinity = @math.exp(1000.0)
  for invalid in [infinity, -infinity, infinity - infinity] {
    assert_true(
      @moonot.coupling_compose([[0.0, invalid]], [[0.0], [1.0]], 1.0)
      is Err(
        @moonot.CouplingCompositionError::InvalidLeft(
          @moonot.CouplingError::NonFiniteCouplingEntry(0, 1)
        )
      ),
    )
    assert_true(
      @moonot.coupling_compose([[0.0, 1.0]], [[0.0], [invalid]], 1.0)
      is Err(
        @moonot.CouplingCompositionError::InvalidRight(
          @moonot.CouplingError::NonFiniteCouplingEntry(1, 0)
        )
      ),
    )
  }
}

///|
test "composition rejects overflowing input row and column marginals" {
  assert_true(
    @moonot.coupling_compose([[1.0e308, 1.0e308]], [[1.0e308], [0.0]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidLeft(
        @moonot.CouplingError::NumericalOverflow(0, 1)
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[1.0e308], [1.0e308]], [[1.0e308]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidLeft(
        @moonot.CouplingError::NumericalOverflow(1, 0)
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[1.0e308]], [[1.0e308, 1.0e308]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidRight(
        @moonot.CouplingError::NumericalOverflow(0, 1)
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose([[1.0e308, 1.0e308]], [[1.0e308], [1.0e308]], 1.0)
    is Err(
      @moonot.CouplingCompositionError::InvalidLeft(
        @moonot.CouplingError::NumericalOverflow(0, 1)
      )
    ),
  )
  assert_true(
    @moonot.coupling_compose(
      [[1.0e308, 0.0], [0.0, 1.0e308]],
      [[1.0e308], [1.0e308]],
      1.0,
    )
    is Err(
      @moonot.CouplingCompositionError::InvalidRight(
        @moonot.CouplingError::NumericalOverflow(1, 0)
      )
    ),
  )
}

///|
test "composition checks output marginals even when every entry is finite" {
  let left = [[1.0e308, 0.0], [0.0, 1.0e308]]
  let right = [[8.0e307], [8.0e307]]
  assert_true(
    @moonot.coupling_compose(left, right, 3.0e307)
    is Err(
      @moonot.CouplingCompositionError::InvalidOutput(
        @moonot.CouplingError::NumericalOverflow(1, 0)
      )
    ),
  )
  assert_eq(left, [[1.0e308, 0.0], [0.0, 1.0e308]])
  assert_eq(right, [[8.0e307], [8.0e307]])
}

///|
test "composition inputs unchanged on validation support and mismatch errors" {
  let left = [[1.0, 0.5]]
  let right = [[1.0], [0.0]]
  assert_true(
    @moonot.coupling_compose(left, right, 1.0)
    is Err(@moonot.CouplingCompositionError::UndefinedBridgeConditional(1)),
  )
  assert_eq(left, [[1.0, 0.5]])
  assert_eq(right, [[1.0], [0.0]])
  let mismatch = [[1.0], [1.0]]
  assert_true(@moonot.coupling_compose(left, mismatch, 0.1) is Err(_))
  assert_eq(left, [[1.0, 0.5]])
  assert_eq(mismatch, [[1.0], [1.0]])
  let invalid = [[-5.0e-324]]
  assert_true(@moonot.coupling_compose(invalid, [[1.0]], 1.0) is Err(_))
  assert_eq(invalid, [[-5.0e-324]])
}

///|
test "composition output rows and inputs do not alias" {
  let left = [[1.0], [2.0]]
  let right = [[1.0, 2.0]]
  let composed = composition_value(
    @moonot.coupling_compose(left, right, 1.0e-12),
  )
  let other_row = composed[1].copy()
  composed[0][0] = 99.0
  assert_eq(composed[1], other_row)
  assert_eq(left, [[1.0], [2.0]])
  assert_eq(right, [[1.0, 2.0]])
}
