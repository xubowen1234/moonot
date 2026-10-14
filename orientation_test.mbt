///|
test "coupling transpose swaps marginals and round trips rectangular plans" {
  let p = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
  match @moonot.coupling_transpose(p) {
    Err(_) => assert_true(false)
    Ok(t) => {
      assert_eq(t, [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
      match (@moonot.coupling_marginals(p), @moonot.coupling_marginals(t)) {
        (Ok(a), Ok(b)) => {
          assert_eq(a.source, b.target)
          assert_eq(a.target, b.source)
        }
        _ => assert_true(false)
      }
      match @moonot.coupling_transpose(t) {
        Ok(r) => assert_eq(r, p)
        Err(_) => assert_true(false)
      }
      t[0][0] = 9.0
      assert_eq(p[0][0], 0.0)
    }
  }
}

///|
test "coupling permutation follows output-to-input convention without aliasing" {
  let p = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
  let a = [1, 0]
  let b = [2, 0, 1]
  match @moonot.coupling_permute(p, a, b) {
    Err(_) => assert_true(false)
    Ok(r) => {
      assert_eq(r, [[5.0, 3.0, 4.0], [2.0, 0.0, 1.0]])
      r[0][0] = 99.0
      assert_eq(p[1][2], 5.0)
    }
  }
  assert_eq(a, [1, 0])
  assert_eq(b, [2, 0, 1])
}

///|
test "coupling orientation rejects invalid plans before permutations" {
  assert_true(
    @moonot.coupling_transpose([]) is Err(@moonot.CouplingError::EmptyCoupling),
  )
  assert_true(
    @moonot.coupling_transpose([[0.0], []])
    is Err(@moonot.CouplingError::RaggedCouplingRow(1, 1, 0)),
  )
  assert_true(
    @moonot.coupling_permute([[-1.0]], [], [])
    is Err(
      @moonot.CouplingOrientationError::InvalidCoupling(
        @moonot.CouplingError::NegativeCouplingEntry(0, 0)
      )
    ),
  )
  let inf = @math.exp(1000.0)
  assert_true(
    @moonot.coupling_transpose([[inf]])
    is Err(@moonot.CouplingError::NonFiniteCouplingEntry(0, 0)),
  )
}

///|
test "coupling permutation demands complete bijections" {
  let p = [[1.0, 2.0], [3.0, 4.0]]
  assert_true(
    @moonot.coupling_permute(p, [0], [0, 1])
    is Err(@moonot.CouplingOrientationError::SourcePermutationLength),
  )
  assert_true(
    @moonot.coupling_permute(p, [0, 1], [0])
    is Err(@moonot.CouplingOrientationError::TargetPermutationLength),
  )
  assert_true(
    @moonot.coupling_permute(p, [-1, 0], [0, 1])
    is Err(@moonot.CouplingOrientationError::SourceIndexOutOfRange(0)),
  )
  assert_true(
    @moonot.coupling_permute(p, [0, 1], [2, 0])
    is Err(@moonot.CouplingOrientationError::TargetIndexOutOfRange(0)),
  )
  assert_true(
    @moonot.coupling_permute(p, [0, 0], [0, 1])
    is Err(@moonot.CouplingOrientationError::DuplicateSourceIndex(1)),
  )
  assert_true(
    @moonot.coupling_permute(p, [0, 1], [1, 1])
    is Err(@moonot.CouplingOrientationError::DuplicateTargetIndex(1)),
  )
}

///|
test "coupling orientation preserves every finite entry including extreme scales" {
  let p = [[1.0e308, 1.0e-320]]
  match @moonot.coupling_permute(p, [0], [1, 0]) {
    Ok(r) => assert_eq(r, [[1.0e-320, 1.0e308]])
    Err(_) => assert_true(false)
  }
  match @moonot.coupling_transpose(p) {
    Ok(r) => assert_eq(r, [[1.0e308], [1.0e-320]])
    Err(_) => assert_true(false)
  }
}
