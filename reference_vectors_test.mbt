// Independent analytic reference vectors. No POT source or upstream fixtures copied.
// Derivations and optional executed POT validation: docs/reference-validation.md.

///|
fn reference_close(actual : Double, expected : Double) -> Unit raise {
  let error = if actual > expected {
    actual - expected
  } else {
    expected - actual
  }
  let scale = if expected > 0.0 { expected } else { -expected }
  assert_true(error <= 2.0e-10 * (1.0 + scale))
}

///|
fn reference_kl(x : Double, y : Double) -> Double {
  if x == 0.0 {
    y
  } else {
    x * @math.ln(x / y) - x + y
  }
}

///|
fn reference_balanced(
  a : Array[Double],
  b : Array[Double],
  c : Array[Array[Double]],
  epsilon : Double,
  expected : Array[Array[Double]],
) -> Unit raise {
  let r = match
    @moonot.balanced_sinkhorn(a, b, c, {
      epsilon,
      tolerance: 1.0e-13,
      max_iterations: 100000,
    }) {
    Ok(r) => r
    Err(_) => {
      assert_true(false)
      return
    }
  }
  assert_true(r.termination is @moonot.SinkhornTermination::Converged)
  assert_true(r.marginal_residual <= 1.0e-13)
  let mut transport = 0.0
  let mut entropy = 0.0
  for i in 0..<a.length() {
    for j in 0..<b.length() {
      let p = expected[i][j]
      reference_close(r.coupling[i][j], p)
      if p == 0.0 {
        assert_eq(r.coupling[i][j], 0.0)
      }
      transport = transport + p * c[i][j]
      if p > 0.0 {
        entropy = entropy + epsilon * p * (@math.ln(p) - 1.0)
      }
    }
  }
  reference_close(r.transport_cost, transport)
  reference_close(r.entropy_regularizer, entropy)
  reference_close(r.regularized_objective, transport + entropy)
}

///|
fn reference_unbalanced(
  a : Array[Double],
  b : Array[Double],
  c : Array[Array[Double]],
  epsilon : Double,
  rs : Double,
  rt : Double,
  weight : Double,
  expected : Array[Array[Double]],
) -> Unit raise {
  let r = match
    @moonot.unbalanced_sinkhorn(a, b, c, {
      epsilon,
      rho_source: rs,
      rho_target: rt,
      cost_weight: weight,
      tolerance: 1.0e-13,
      max_iterations: 100000,
    }) {
    Ok(r) => r
    Err(_) => {
      assert_true(false)
      return
    }
  }
  assert_true(r.termination is @moonot.UnbalancedSinkhornTermination::Converged)
  assert_true(r.scaling_residual <= 1.0e-13)
  let rows = Array::make(a.length(), 0.0)
  let cols = Array::make(b.length(), 0.0)
  let mut transport = 0.0
  let mut coupling_kl = 0.0
  let mut source_kl = 0.0
  let mut target_kl = 0.0
  for i in 0..<a.length() {
    for j in 0..<b.length() {
      let p = expected[i][j]
      reference_close(r.coupling[i][j], p)
      if p == 0.0 {
        assert_eq(r.coupling[i][j], 0.0)
      }
      rows[i] = rows[i] + p
      cols[j] = cols[j] + p
      transport = transport + weight * c[i][j] * p
      coupling_kl = coupling_kl + reference_kl(p, a[i] * b[j])
    }
    source_kl = source_kl + reference_kl(rows[i], a[i])
  }
  for j in 0..<b.length() {
    target_kl = target_kl + reference_kl(cols[j], b[j])
  }
  reference_close(r.transport_cost, transport)
  reference_close(r.reference_kl, coupling_kl)
  reference_close(r.source_marginal_kl, source_kl)
  reference_close(r.target_marginal_kl, target_kl)
  reference_close(
    r.objective,
    transport + epsilon * coupling_kl + rs * source_kl + rt * target_kl,
  )
}

///|
test "reference balanced analytic scalar includes negative mass entropy term" {
  reference_balanced([2.0], [2.0], [[0.4]], 0.3, [[2.0]])
}

///|
test "reference balanced analytic one active row rectangular zero supports" {
  reference_balanced(
    [0.0, 2.0, 0.0],
    [0.5, 0.0, 1.25, 0.25],
    [[0.2, 0.5, 0.8, 0.9], [1.0, 0.3, 0.4, 0.7], [0.1, 0.0, 0.3, 0.2]],
    0.4,
    [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 1.25, 0.25], [0.0, 0.0, 0.0, 0.0]],
  )
}

///|
test "reference balanced analytic zero cost outer product divided by total" {
  reference_balanced(
    [0.5, 1.5],
    [0.4, 0.6, 1.0],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    0.7,
    [[0.1, 0.15, 0.25], [0.3, 0.45, 0.75]],
  )
}

///|
test "reference balanced analytic symmetric Gibbs ratio" {
  let d = 0.5 / (1.0 + @math.exp(-1.0))
  let o = 0.5 - d
  reference_balanced([0.5, 0.5], [0.5, 0.5], [[0.0, 1.0], [1.0, 0.0]], 1.0, [
    [d, o],
    [o, d],
  ])
}

///|
test "reference balanced analytic all zero rectangle" {
  reference_balanced(
    [0.0, 0.0],
    [0.0, 0.0, 0.0],
    [[0.2, 0.8, 0.0], [1.0, 0.4, 0.2]],
    0.7,
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  )
}

///|
test "reference unbalanced analytic scalar asymmetric and zero penalties" {
  let eps = 2.0
  let a = 2.0
  let b = 3.0
  for penalties in [(1.0, 3.0), (0.0, 0.0), (0.0, 4.0), (5.0, 0.0)] {
    let (rs, rt) = penalties
    let p = @math.exp(
      ((eps + rs) * @math.ln(a) + (eps + rt) * @math.ln(b) - 2.0) /
      (eps + rs + rt),
    )
    reference_unbalanced([a], [b], [[4.0]], eps, rs, rt, 0.5, [[p]])
  }
}

///|
test "reference unbalanced analytic no marginal penalties is weighted Gibbs" {
  let a = [0.0, 0.2, 1.3]
  let b = [0.8, 0.4]
  let c = [[0.2, 0.7], [0.0, 1.0], [0.6, 0.1]]
  let expected : Array[Array[Double]] = []
  for i in 0..<a.length() {
    let row : Array[Double] = []
    for j in 0..<b.length() {
      row.push(a[i] * b[j] * @math.exp(-1.2 * c[i][j] / 0.6))
    }
    expected.push(row)
  }
  reference_unbalanced(a, b, c, 0.6, 0.0, 0.0, 1.2, expected)
}

///|
test "reference unbalanced analytic zero weighted cost rank one with penalties" {
  let a = [0.5, 1.5]
  let b = [0.2, 0.0, 2.0]
  let eps = 0.9
  let rs = 0.7
  let rt = 2.3
  let scale = @math.exp(
    -(rs * @math.ln(2.2) + rt * @math.ln(2.0)) / (eps + rs + rt),
  )
  let expected : Array[Array[Double]] = []
  for i in 0..<a.length() {
    let row : Array[Double] = []
    for j in 0..<b.length() {
      row.push(a[i] * b[j] * scale)
    }
    expected.push(row)
  }
  reference_unbalanced(
    a,
    b,
    [[0.3, 0.2, 0.7], [0.4, 0.1, 1.0]],
    eps,
    rs,
    rt,
    0.0,
    expected,
  )
}

///|
test "reference unbalanced analytic empty positive supports retain KL constants" {
  let c = [[0.2, 0.8, 0.3], [1.0, 0.4, 0.0]]
  let z = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  reference_unbalanced([0.0, 0.0], [1.0, 2.0, 0.0], c, 0.7, 2.0, 3.0, 1.0, z)
  reference_unbalanced([1.0, 2.0], [0.0, 0.0, 0.0], c, 0.7, 2.0, 3.0, 1.0, z)
  reference_unbalanced([0.0, 0.0], [0.0, 0.0, 0.0], c, 0.7, 0.0, 0.0, 1.0, z)
}
