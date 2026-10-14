// Deterministic transport metamorphic tests, independent of the POT oracle slice.
// Relations follow the documented objectives by substitution/stationarity.
// No snapshots, random seeds, Python dependencies, or external reference data.

///|
fn metamorphic_close(actual : Double, expected : Double) -> Unit raise {
  let difference = if actual > expected {
    actual - expected
  } else {
    expected - actual
  }
  let magnitude = if expected > 0.0 { expected } else { -expected }
  assert_true(difference <= 5.0e-10 * (1.0 + magnitude))
}

///|
fn metamorphic_source() -> Array[Double] {
  [0.0, 0.7, 1.3]
}

///|
fn metamorphic_target() -> Array[Double] {
  [0.4, 0.0, 0.9, 0.7]
}

///|
fn metamorphic_cost() -> Array[Array[Double]] {
  [[0.3, 0.1, 0.8, 0.4], [1.0, 0.2, 0.3, 0.7], [0.2, 0.9, 1.1, 0.1]]
}

///|
fn metamorphic_scale_vector(x : Array[Double], scale : Double) -> Array[Double] {
  x.map(value => scale * value)
}

///|
fn metamorphic_affine_cost(
  c : Array[Array[Double]],
  scale : Double,
  shift : Double,
) -> Array[Array[Double]] {
  c.map(row => row.map(value => scale * value + shift))
}

///|
fn metamorphic_sum(x : Array[Double]) -> Double {
  let mut total = 0.0
  for value in x {
    total = total + value
  }
  total
}

///|
fn metamorphic_plan_mass(p : Array[Array[Double]]) -> Double {
  let mut mass = 0.0
  for row in p {
    mass = mass + metamorphic_sum(row)
  }
  mass
}

///|
fn metamorphic_scaled_plan(
  actual : Array[Array[Double]],
  original : Array[Array[Double]],
  scale : Double,
) -> Unit raise {
  assert_eq(actual.length(), original.length())
  for i in 0..<original.length() {
    assert_eq(actual[i].length(), original[i].length())
    for j in 0..<original[i].length() {
      metamorphic_close(actual[i][j], scale * original[i][j])
      if original[i][j] == 0.0 {
        assert_eq(actual[i][j], 0.0)
      }
    }
  }
}

///|
fn metamorphic_balanced(
  a : Array[Double],
  b : Array[Double],
  c : Array[Array[Double]],
  eps : Double,
) -> @moonot.SinkhornResult raise {
  let r = match
    @moonot.balanced_sinkhorn(a, b, c, {
      epsilon: eps,
      tolerance: 1.0e-13,
      max_iterations: 100000,
    }) {
    Ok(r) => r
    Err(_) => abort("metamorphic balanced solver error")
  }
  assert_true(r.termination is @moonot.SinkhornTermination::Converged)
  assert_true(r.marginal_residual <= 1.0e-13)
  r
}

///|
fn metamorphic_unbalanced(
  a : Array[Double],
  b : Array[Double],
  c : Array[Array[Double]],
  eps : Double,
  rs : Double,
  rt : Double,
  weight : Double,
) -> @moonot.UnbalancedSinkhornResult raise {
  let r = match
    @moonot.unbalanced_sinkhorn(a, b, c, {
      epsilon: eps,
      rho_source: rs,
      rho_target: rt,
      cost_weight: weight,
      tolerance: 1.0e-13,
      max_iterations: 100000,
    }) {
    Ok(r) => r
    Err(_) => abort("metamorphic unbalanced solver error")
  }
  assert_true(r.termination is @moonot.UnbalancedSinkhornTermination::Converged)
  assert_true(r.scaling_residual <= 1.0e-13)
  r
}

///|
fn metamorphic_permute_vector(
  x : Array[Double],
  order : Array[Int],
) -> Array[Double] {
  order.map(index => x[index])
}

///|
fn metamorphic_permute_matrix(
  x : Array[Array[Double]],
  rows : Array[Int],
  cols : Array[Int],
) -> Array[Array[Double]] {
  rows.map(i => cols.map(j => x[i][j]))
}

///|
fn metamorphic_permuted_plan(
  actual : Array[Array[Double]],
  original : Array[Array[Double]],
  rows : Array[Int],
  cols : Array[Int],
) -> Unit raise {
  metamorphic_scaled_plan(
    actual,
    metamorphic_permute_matrix(original, rows, cols),
    1.0,
  )
}

///|
test "metamorphic balanced joint cost epsilon scaling preserves coupling" {
  let a = metamorphic_source()
  let b = metamorphic_target()
  let c = metamorphic_cost()
  let eps = 0.6
  let original = metamorphic_balanced(a, b, c, eps)
  for scale in [0.2, 1.7, 4.0] {
    let changed = metamorphic_balanced(
      a,
      b,
      metamorphic_affine_cost(c, scale, 0.0),
      scale * eps,
    )
    metamorphic_scaled_plan(changed.coupling, original.coupling, 1.0)
    metamorphic_close(changed.transport_cost, scale * original.transport_cost)
    metamorphic_close(
      changed.entropy_regularizer,
      scale * original.entropy_regularizer,
    )
    metamorphic_close(
      changed.regularized_objective,
      scale * original.regularized_objective,
    )
  }
}

///|
test "metamorphic balanced mass scaling includes entropy log mass correction" {
  let a = metamorphic_source()
  let b = metamorphic_target()
  let c = metamorphic_cost()
  let eps = 0.6
  let original = metamorphic_balanced(a, b, c, eps)
  let mass = metamorphic_sum(a)
  // H(sP) = s H(P) + s*mass*log(s); no implicit probability normalization.
  for scale in [0.25, 1.7, 3.0] {
    let changed = metamorphic_balanced(
      metamorphic_scale_vector(a, scale),
      metamorphic_scale_vector(b, scale),
      c,
      eps,
    )
    let correction = eps * scale * mass * @math.ln(scale)
    metamorphic_scaled_plan(changed.coupling, original.coupling, scale)
    metamorphic_close(changed.transport_cost, scale * original.transport_cost)
    metamorphic_close(
      changed.entropy_regularizer,
      scale * original.entropy_regularizer + correction,
    )
    metamorphic_close(
      changed.regularized_objective,
      scale * original.regularized_objective + correction,
    )
  }
}

///|
test "metamorphic balanced independent support permutations preserve objective" {
  let a = metamorphic_source()
  let b = metamorphic_target()
  let c = metamorphic_cost()
  let original = metamorphic_balanced(a, b, c, 0.6)
  for orders in [([2, 0, 1], [3, 1, 0, 2]), ([1, 2, 0], [2, 0, 3, 1])] {
    let (rows, cols) = orders
    let changed = metamorphic_balanced(
      metamorphic_permute_vector(a, rows),
      metamorphic_permute_vector(b, cols),
      metamorphic_permute_matrix(c, rows, cols),
      0.6,
    )
    metamorphic_permuted_plan(changed.coupling, original.coupling, rows, cols)
    metamorphic_close(changed.transport_cost, original.transport_cost)
    metamorphic_close(changed.entropy_regularizer, original.entropy_regularizer)
    metamorphic_close(
      changed.regularized_objective,
      original.regularized_objective,
    )
  }
}

///|
test "metamorphic balanced constant cost shift preserves plan adds shift times mass" {
  let a = metamorphic_source()
  let b = metamorphic_target()
  let c = metamorphic_cost()
  let original = metamorphic_balanced(a, b, c, 0.6)
  for shift in [0.1, 0.8, 2.0] {
    let changed = metamorphic_balanced(
      a,
      b,
      metamorphic_affine_cost(c, 1.0, shift),
      0.6,
    )
    metamorphic_scaled_plan(changed.coupling, original.coupling, 1.0)
    metamorphic_close(
      changed.transport_cost,
      original.transport_cost + shift * metamorphic_sum(a),
    )
    metamorphic_close(changed.entropy_regularizer, original.entropy_regularizer)
    metamorphic_close(
      changed.regularized_objective,
      original.regularized_objective + shift * metamorphic_sum(a),
    )
  }
}

///|
test "metamorphic KL unbalanced joint regularizer penalty cost scaling preserves plan" {
  let a = metamorphic_source()
  let b = [0.3, 0.0, 1.2, 0.9]
  let c = metamorphic_cost()
  for penalties in [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (0.7, 2.3)] {
    let (rs, rt) = penalties
    let original = metamorphic_unbalanced(a, b, c, 0.6, rs, rt, 0.8)
    for scale in [0.3, 2.5] {
      let changed = metamorphic_unbalanced(
        a,
        b,
        c,
        scale * 0.6,
        scale * rs,
        scale * rt,
        scale * 0.8,
      )
      metamorphic_scaled_plan(changed.coupling, original.coupling, 1.0)
      metamorphic_close(changed.transport_cost, scale * original.transport_cost)
      metamorphic_close(changed.reference_kl, original.reference_kl)
      metamorphic_close(changed.source_marginal_kl, original.source_marginal_kl)
      metamorphic_close(changed.target_marginal_kl, original.target_marginal_kl)
      metamorphic_close(changed.objective, scale * original.objective)
    }
  }
}

///|
test "metamorphic KL unbalanced rectangular support permutation including zero masses" {
  let a = metamorphic_source()
  let b = [0.3, 0.0, 1.2, 0.9]
  let c = metamorphic_cost()
  let rows = [2, 0, 1]
  let cols = [3, 1, 0, 2]
  for penalties in [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (0.7, 2.3)] {
    let (rs, rt) = penalties
    let original = metamorphic_unbalanced(a, b, c, 0.6, rs, rt, 0.8)
    let changed = metamorphic_unbalanced(
      metamorphic_permute_vector(a, rows),
      metamorphic_permute_vector(b, cols),
      metamorphic_permute_matrix(c, rows, cols),
      0.6,
      rs,
      rt,
      0.8,
    )
    metamorphic_permuted_plan(changed.coupling, original.coupling, rows, cols)
    metamorphic_close(changed.transport_cost, original.transport_cost)
    metamorphic_close(changed.reference_kl, original.reference_kl)
    metamorphic_close(changed.source_marginal_kl, original.source_marginal_kl)
    metamorphic_close(changed.target_marginal_kl, original.target_marginal_kl)
    metamorphic_close(changed.objective, original.objective)
    metamorphic_close(
      changed.source_marginal_residual,
      original.source_marginal_residual,
    )
    metamorphic_close(
      changed.target_marginal_residual,
      original.target_marginal_residual,
    )
  }
}

///|
// D(g*x || h*y) = g*D(x||y) + g*sum(x)*log(g/h) + (h-g)*sum(y).
fn metamorphic_rescaled_kl(
  divergence : Double,
  plan_mass : Double,
  reference_mass : Double,
  plan_scale : Double,
  reference_scale : Double,
) -> Double {
  plan_scale * divergence +
  plan_scale * plan_mass * @math.ln(plan_scale / reference_scale) +
  (reference_scale - plan_scale) * reference_mass
}

///|
test "metamorphic KL mass scaling uses reference product not balanced linear scaling" {
  let a = metamorphic_source()
  let b = [0.3, 0.0, 1.2, 0.9]
  let c = metamorphic_cost()
  let eps = 0.6
  for penalties in [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (0.7, 2.3)] {
    let (rs, rt) = penalties
    let original = metamorphic_unbalanced(a, b, c, eps, rs, rt, 0.8)
    let mass = metamorphic_plan_mass(original.coupling)
    for scales in [(1.7, 0.6), (0.25, 0.25), (2.0, 2.0)] {
      let (sa, sb) = scales
      // Stationarity: (eps+rs+rt)*log(g)=(eps+rs)*log(sa)+(eps+rt)*log(sb).
      let g = @math.exp(
        ((eps + rs) * @math.ln(sa) + (eps + rt) * @math.ln(sb)) /
        (eps + rs + rt),
      )
      let changed = metamorphic_unbalanced(
        metamorphic_scale_vector(a, sa),
        metamorphic_scale_vector(b, sb),
        c,
        eps,
        rs,
        rt,
        0.8,
      )
      let reference = metamorphic_rescaled_kl(
        original.reference_kl,
        mass,
        metamorphic_sum(a) * metamorphic_sum(b),
        g,
        sa * sb,
      )
      let source = metamorphic_rescaled_kl(
        original.source_marginal_kl,
        mass,
        metamorphic_sum(a),
        g,
        sa,
      )
      let target = metamorphic_rescaled_kl(
        original.target_marginal_kl,
        mass,
        metamorphic_sum(b),
        g,
        sb,
      )
      metamorphic_scaled_plan(changed.coupling, original.coupling, g)
      metamorphic_close(changed.transport_cost, g * original.transport_cost)
      metamorphic_close(changed.reference_kl, reference)
      metamorphic_close(changed.source_marginal_kl, source)
      metamorphic_close(changed.target_marginal_kl, target)
      metamorphic_close(
        changed.objective,
        g * original.transport_cost +
        eps * reference +
        rs * source +
        rt * target,
      )
    }
  }
}

///|
test "metamorphic KL constant cost shift changes mass unlike balanced transport" {
  let a = metamorphic_source()
  let b = [0.3, 0.0, 1.2, 0.9]
  let c = metamorphic_cost()
  let eps = 0.6
  for penalties in [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (0.7, 2.3)] {
    let (rs, rt) = penalties
    for weight in [0.0, 0.8, 1.5] {
      let original = metamorphic_unbalanced(a, b, c, eps, rs, rt, weight)
      let mass = metamorphic_plan_mass(original.coupling)
      for shift in [0.2, 0.9] {
        let g = @math.exp(-weight * shift / (eps + rs + rt))
        let changed = metamorphic_unbalanced(
          a,
          b,
          metamorphic_affine_cost(c, 1.0, shift),
          eps,
          rs,
          rt,
          weight,
        )
        let transport = g * (original.transport_cost + weight * shift * mass)
        let reference = metamorphic_rescaled_kl(
          original.reference_kl,
          mass,
          metamorphic_sum(a) * metamorphic_sum(b),
          g,
          1.0,
        )
        let source = metamorphic_rescaled_kl(
          original.source_marginal_kl,
          mass,
          metamorphic_sum(a),
          g,
          1.0,
        )
        let target = metamorphic_rescaled_kl(
          original.target_marginal_kl,
          mass,
          metamorphic_sum(b),
          g,
          1.0,
        )
        metamorphic_scaled_plan(changed.coupling, original.coupling, g)
        metamorphic_close(changed.transport_cost, transport)
        metamorphic_close(changed.reference_kl, reference)
        metamorphic_close(changed.source_marginal_kl, source)
        metamorphic_close(changed.target_marginal_kl, target)
        metamorphic_close(
          changed.objective,
          transport + eps * reference + rs * source + rt * target,
        )
        if weight > 0.0 {
          assert_true(metamorphic_plan_mass(changed.coupling) < mass)
        }
      }
    }
  }
}

///|
test "metamorphic balanced versus KL no penalty zero cost nonunit reference mass" {
  let a = [0.5, 1.5]
  let b = [0.2, 0.0, 0.6, 1.2]
  let c = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
  let eps = 0.8
  let balanced = metamorphic_balanced(a, b, c, eps)
  let kl = metamorphic_unbalanced(a, b, c, eps, 0.0, 0.0, 1.0)
  // Balanced entropy: a*b/A. KL without marginal penalties: a*b (not normalized).
  metamorphic_scaled_plan(kl.coupling, balanced.coupling, metamorphic_sum(a))
  metamorphic_close(metamorphic_plan_mass(balanced.coupling), 2.0)
  metamorphic_close(metamorphic_plan_mass(kl.coupling), 4.0)
  metamorphic_close(kl.reference_kl, 0.0)
  metamorphic_close(kl.objective, 0.0)
  // Ordinary entropy is not KL(P||a*b); its objective is nonzero at P=a*b.
  let mut entropy = 0.0
  for row in kl.coupling {
    for p in row {
      if p > 0.0 {
        entropy = entropy + eps * p * (@math.ln(p) - 1.0)
      }
    }
  }
  assert_true(entropy < -1.0)
  assert_true(balanced.regularized_objective < -1.0)
}
