///|
fn information_assert_close(
  result : Result[Double, @moonot.CouplingInformationError],
  expected : Double,
  tolerance : Double,
) -> Unit raise {
  match result {
    Err(_) => assert_true(false)
    Ok(actual) => {
      let difference = if actual >= expected {
        actual - expected
      } else {
        expected - actual
      }
      assert_true(difference <= tolerance)
    }
  }
}

///|
test "information entropy analytic unnormalized rectangular plan" {
  // Four units of mass; this is not Shannon entropy or a probability API.
  information_assert_close(
    @moonot.coupling_entropy([[0.0, 1.0], [2.0, 1.0]]),
    2.0 * @math.ln(2.0) - 4.0,
    1.0e-15,
  )
  information_assert_close(
    @moonot.coupling_entropy([[0.5]]),
    0.5 * @math.ln(0.5) - 0.5,
    0.0,
  )
}

///|
test "information zeros and signed zero obey support conventions" {
  information_assert_close(@moonot.coupling_entropy([[0.0, -0.0]]), 0.0, 0.0)
  information_assert_close(
    @moonot.coupling_kl([[0.0, -0.0]], [[0.0, 0.0]]),
    0.0,
    0.0,
  )
  information_assert_close(
    @moonot.coupling_kl([[0.0, 0.0]], [[2.0, 3.0]]),
    5.0,
    0.0,
  )
  assert_true(
    @moonot.coupling_kl([[0.0, 1.0]], [[2.0, 0.0]])
    is Err(@moonot.CouplingInformationError::ZeroReferenceSupport(0, 1)),
  )
  assert_true(
    @moonot.coupling_kl([[5.0e-324]], [[-0.0]])
    is Err(@moonot.CouplingInformationError::ZeroReferenceSupport(0, 0)),
  )
}

///|
test "information KL analytic unequal mass and asymmetry" {
  information_assert_close(
    @moonot.coupling_kl([[2.0, 0.0], [1.0, 3.0]], [[1.0, 4.0], [1.0, 3.0]]),
    2.0 * @math.ln(2.0) + 3.0,
    1.0e-15,
  )
  information_assert_close(
    @moonot.coupling_kl([[1.0]], [[2.0]]),
    1.0 - @math.ln(2.0),
    1.0e-15,
  )
}

///|
test "information identical plans give exact zero without total mass overflow" {
  let p = [[0.0, 5.0e-324, 1.0e-300], [1.0, 1.7e308, 1.7e308]]
  information_assert_close(@moonot.coupling_kl(p, p), 0.0, 0.0)
}

///|
test "information KL retains near equal divergence in both directions" {
  let p = 1.0000000000000002
  let q = 1.0
  let difference = p - q
  let leading = 0.5 * difference * difference
  information_assert_close(
    @moonot.coupling_kl([[p]], [[q]]),
    leading,
    leading * 1.0e-14,
  )
  information_assert_close(
    @moonot.coupling_kl([[q]], [[p]]),
    leading,
    leading * 1.0e-14,
  )
  // These huge adjacent-scale masses can have identical rounded logarithms.
  let huge_p = 1.000000000000001e300
  let huge_q = 1.0e300
  let huge_difference = huge_p - huge_q
  let huge_leading = huge_difference * (huge_difference / huge_q * 0.5)
  information_assert_close(
    @moonot.coupling_kl([[huge_p]], [[huge_q]]),
    huge_leading,
    huge_leading * 1.0e-12,
  )
  information_assert_close(
    @moonot.coupling_kl([[huge_q]], [[huge_p]]),
    huge_leading,
    huge_leading * 1.0e-12,
  )
}

///|
test "information KL series endpoints and branch boundaries" {
  information_assert_close(
    @moonot.coupling_kl([[1.5]], [[1.0]]),
    0.10819766216224657297,
    2.0e-16,
  )
  information_assert_close(
    @moonot.coupling_kl([[0.5]], [[1.0]]),
    0.15342640972002734529,
    2.0e-16,
  )
  information_assert_close(
    @moonot.coupling_kl([[1.5000000000000002]], [[1.0]]),
    0.10819766216224666300,
    3.0e-16,
  )
  information_assert_close(
    @moonot.coupling_kl([[0.49999999999999994]], [[1.0]]),
    0.15342640972002738377,
    3.0e-16,
  )
}

///|
test "information tiny and huge masses do not require a finite ratio" {
  // p/q overflows in the first direction and underflows in the second.
  information_assert_close(
    @moonot.coupling_kl([[1.0e300]], [[1.0e-300]]),
    1.38055105579642741041e303,
    3.0e289,
  )
  information_assert_close(
    @moonot.coupling_kl([[1.0e-300]], [[1.0e300]]),
    1.0e300,
    0.0,
  )
  information_assert_close(
    @moonot.coupling_entropy([[1.0e-300]]),
    -6.91775527898213705205e-298,
    2.0e-311,
  )
  information_assert_close(
    @moonot.coupling_entropy([[5.0e-324]]),
    -3.681e-321,
    5.0e-324,
  )
}

///|
test "information KL avoids spurious intermediate overflow" {
  // p*log(p/q) overflows, but p*(log(p)-log(q)-1)+q is finite.
  information_assert_close(
    @moonot.coupling_kl([[1.0e308]], [[1.0e307]]),
    1.40258509299404568402e308,
    2.0e295,
  )
  // A near-equal tiny pair has a positive subnormal divergence.
  let p = 1.000000000000001e-280
  let q = 1.0e-280
  let difference = p - q
  let leading = difference * (difference / q * 0.5)
  assert_true(leading > 0.0)
  information_assert_close(@moonot.coupling_kl([[p]], [[q]]), leading, 5.0e-324)
}

///|
test "information diagnostics reject empty and ragged plans" {
  assert_true(
    @moonot.coupling_entropy([])
    is Err(
      @moonot.CouplingInformationError::InvalidCoupling(
        @moonot.CouplingError::EmptyCoupling
      )
    ),
  )
  assert_true(
    @moonot.coupling_entropy([[]])
    is Err(
      @moonot.CouplingInformationError::InvalidCoupling(
        @moonot.CouplingError::EmptyCouplingRow(0)
      )
    ),
  )
  assert_true(
    @moonot.coupling_entropy([[1.0], []])
    is Err(
      @moonot.CouplingInformationError::InvalidCoupling(
        @moonot.CouplingError::RaggedCouplingRow(1, 1, 0)
      )
    ),
  )
  assert_true(
    @moonot.coupling_kl([[1.0], [1.0, 2.0]], [[1.0]])
    is Err(
      @moonot.CouplingInformationError::InvalidCoupling(
        @moonot.CouplingError::RaggedCouplingRow(1, 1, 2)
      )
    ),
  )
  assert_true(
    @moonot.coupling_kl([[1.0]], [])
    is Err(
      @moonot.CouplingInformationError::InvalidReference(
        @moonot.CouplingError::EmptyCoupling
      )
    ),
  )
  assert_true(
    @moonot.coupling_kl([[1.0]], [[]])
    is Err(
      @moonot.CouplingInformationError::InvalidReference(
        @moonot.CouplingError::EmptyCouplingRow(0)
      )
    ),
  )
  assert_true(
    @moonot.coupling_kl([[1.0]], [[1.0], [1.0, 2.0]])
    is Err(
      @moonot.CouplingInformationError::InvalidReference(
        @moonot.CouplingError::RaggedCouplingRow(1, 1, 2)
      )
    ),
  )
}

///|
test "information KL requires identical row and column counts" {
  assert_true(
    @moonot.coupling_kl([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]])
    is Err(@moonot.CouplingInformationError::RowCountMismatch(1, 2)),
  )
  assert_true(
    @moonot.coupling_kl([[1.0, 2.0]], [[1.0]])
    is Err(@moonot.CouplingInformationError::ColumnCountMismatch(2, 1)),
  )
}

///|
test "information rejects every finite negative including subnormal mass" {
  for negative in [-1.0, -1.0e-300, -5.0e-324] {
    assert_true(
      @moonot.coupling_entropy([[0.0, negative]])
      is Err(
        @moonot.CouplingInformationError::InvalidCoupling(
          @moonot.CouplingError::NegativeCouplingEntry(0, 1)
        )
      ),
    )
    assert_true(
      @moonot.coupling_kl([[negative]], [[1.0]])
      is Err(
        @moonot.CouplingInformationError::InvalidCoupling(
          @moonot.CouplingError::NegativeCouplingEntry(0, 0)
        )
      ),
    )
    assert_true(
      @moonot.coupling_kl([[1.0]], [[negative]])
      is Err(
        @moonot.CouplingInformationError::InvalidReference(
          @moonot.CouplingError::NegativeCouplingEntry(0, 0)
        )
      ),
    )
  }
}

///|
test "information rejects NaN and both infinities in either plan" {
  let infinity = @math.exp(1000.0)
  for invalid in [infinity, -infinity, infinity - infinity] {
    assert_true(
      @moonot.coupling_entropy([[invalid]])
      is Err(
        @moonot.CouplingInformationError::InvalidCoupling(
          @moonot.CouplingError::NonFiniteCouplingEntry(0, 0)
        )
      ),
    )
    assert_true(
      @moonot.coupling_kl([[invalid]], [[1.0]])
      is Err(
        @moonot.CouplingInformationError::InvalidCoupling(
          @moonot.CouplingError::NonFiniteCouplingEntry(0, 0)
        )
      ),
    )
    assert_true(
      @moonot.coupling_kl([[1.0]], [[invalid]])
      is Err(
        @moonot.CouplingInformationError::InvalidReference(
          @moonot.CouplingError::NonFiniteCouplingEntry(0, 0)
        )
      ),
    )
  }
  // Validation of all entries precedes support/numerical evaluation.
  assert_true(
    @moonot.coupling_kl([[1.0, -1.0]], [[0.0, 1.0]])
    is Err(
      @moonot.CouplingInformationError::InvalidCoupling(
        @moonot.CouplingError::NegativeCouplingEntry(0, 1)
      )
    ),
  )
}

///|
test "information reports typed term and accumulated overflow" {
  assert_true(
    @moonot.coupling_entropy([[1.0e308]])
    is Err(@moonot.CouplingInformationError::NumericalOverflow(0, 0)),
  )
  assert_true(
    @moonot.coupling_entropy([[1.0e305, 1.0e305, 1.0e305]])
    is Err(@moonot.CouplingInformationError::NumericalOverflow(0, 2)),
  )
  assert_true(
    @moonot.coupling_kl([[1.0e308]], [[1.0e-308]])
    is Err(@moonot.CouplingInformationError::NumericalOverflow(0, 0)),
  )
  assert_true(
    @moonot.coupling_kl([[0.0], [0.0]], [[1.0e308], [1.0e308]])
    is Err(@moonot.CouplingInformationError::NumericalOverflow(1, 0)),
  )
}

///|
test "information inputs remain unchanged on success and errors" {
  let p = [[0.0, 1.0], [2.0, 3.0]]
  let q = [[4.0, 1.0], [1.0, 3.0]]
  assert_true(@moonot.coupling_entropy(p) is Ok(_))
  assert_true(@moonot.coupling_kl(p, q) is Ok(_))
  assert_eq(p, [[0.0, 1.0], [2.0, 3.0]])
  assert_eq(q, [[4.0, 1.0], [1.0, 3.0]])
  let zero = [[0.0, 0.0], [0.0, 0.0]]
  assert_true(
    @moonot.coupling_kl(p, zero)
    is Err(@moonot.CouplingInformationError::ZeroReferenceSupport(0, 1)),
  )
  assert_eq(p, [[0.0, 1.0], [2.0, 3.0]])
  assert_eq(zero, [[0.0, 0.0], [0.0, 0.0]])
  let invalid = [[-5.0e-324]]
  assert_true(@moonot.coupling_entropy(invalid) is Err(_))
  assert_eq(invalid, [[-5.0e-324]])
  let huge = [[1.0e308]]
  assert_true(@moonot.coupling_entropy(huge) is Err(_))
  assert_eq(huge, [[1.0e308]])
}

///|
test "information KL nonnegativity over small and large scales" {
  for scale in [1.0e-300, 1.0, 1.0e300] {
    for
      ratio in [
        0.01, 0.49, 0.5, 0.75, 0.99999999, 1.0, 1.00000001, 1.25, 1.5, 1.51, 100.0,
      ] {
      match @moonot.coupling_kl([[scale * ratio]], [[scale]]) {
        Err(_) => assert_true(false)
        Ok(value) => assert_true(value >= 0.0)
      }
    }
  }
}

///|
test "information KL compensates accumulation of small positive terms" {
  information_assert_close(
    @moonot.coupling_kl([[0.0, 0.0, 0.0]], [[1.0e16, 1.0, 1.0]]),
    1.0000000000000002e16,
    0.0,
  )
  // A positive mathematical divergence below Double resolution may round
  // to zero, but must never become negative.
  information_assert_close(
    @moonot.coupling_kl([[1.0000000000000002e-300]], [[1.0e-300]]),
    0.0,
    0.0,
  )
}

///|
test "information KL handles subnormal series threshold without a rounded half q" {
  information_assert_close(
    @moonot.coupling_kl([[2.5e-323]], [[1.5e-323]]),
    5.0e-324,
    0.0,
  )
  information_assert_close(
    @moonot.coupling_kl([[1.5e-323]], [[2.5e-323]]),
    0.0,
    0.0,
  )
}
