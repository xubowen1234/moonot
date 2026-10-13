///|
test "generalized I-divergence honors both zero-reference conventions" {
  match unbalanced_kl_from_log_reference(0.0, @math.ln(3.0)) {
    Err(_) => assert_true(false)
    Ok(value) => assert_true(wb_approximately_equal(value, 3.0, 1.0e-14))
  }
  match unbalanced_kl_from_log_reference(2.0, @math.ln(2.0)) {
    Err(_) => assert_true(false)
    Ok(value) => assert_eq(value, 0.0)
  }
}

///|
test "generalized I-divergence near equality avoids cancellation" {
  match unbalanced_kl_from_log_reference(1.00000001, 0.0) {
    Err(_) => assert_true(false)
    Ok(value) => {
      assert_true(value >= 0.0)
      assert_true(wb_approximately_equal(value, 5.0e-17, 1.0e-24))
    }
  }
}

///|
test "epsilon ratio rescaling handles representable extremes and rejects overflow" {
  match unbalanced_rescale_seed_log(1.0e-308, 1.0e308, 1.0e-308) {
    Err(_) => assert_true(false)
    Ok(value) => {
      assert_true(is_finite(value))
      assert_true(wb_approximately_equal(value, 1.0e308, 1.0e293))
    }
  }
  assert_true(unbalanced_rescale_seed_log(1.0e100, 1.0e308, 1.0e-308) is Err(_))
  match unbalanced_rescale_seed_log(-2.5, 0.3, 0.3) {
    Err(_) => assert_true(false)
    Ok(value) => assert_eq(value, -2.5)
  }
}

///|
test "zero-support warm seed is neutral and does not alias the returned plan" {
  let options : UnbalancedSinkhornOptions = {
    epsilon: 0.7,
    rho_source: 2.0,
    rho_target: 0.5,
    cost_weight: 1.0,
    tolerance: 1.0e-10,
    max_iterations: 10,
  }
  match
    unbalanced_sinkhorn_with_warm_start(
      [0.0, 0.0],
      [2.0, 1.0],
      [[0.0, 1.0], [1.0, 0.0]],
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((result, seed)) => {
      assert_eq(seed.log_u, [0.0, 0.0])
      assert_eq(seed.log_v, [0.0, 0.0])
      result.coupling[0][0] = 9.0
      match
        unbalanced_sinkhorn_with_warm_start(
          [0.0, 0.0],
          [2.0, 1.0],
          [[0.0, 1.0], [1.0, 0.0]],
          options,
          Some(seed),
        ) {
        Err(_) => assert_true(false)
        Ok((reused, _)) => assert_eq(reused.coupling[0][0], 0.0)
      }
    }
  }
}

///|
test "malformed unbalanced seed data is rejected without mutating the supplied seed" {
  let malformed = UnbalancedSinkhornWarmStart::{
    log_u: [@math.exp(1000.0)],
    log_v: [0.0],
    epsilon: 1.0,
    rows: 1,
    columns: 1,
  }
  assert_true(
    unbalanced_prepare_warm_start(Some(malformed), [1.0], [1.0], {
      epsilon: 1.0,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: 1.0,
      tolerance: 1.0e-10,
      max_iterations: 10,
    })
    is Err(UnbalancedSinkhornWarmStartError::InvalidWarmStart),
  )
  assert_true(malformed.log_u[0].is_inf())
  assert_eq(malformed.log_v[0], 0.0)
}

///|
test "seed input arrays and returned coupling are isolated from internal state" {
  let options : UnbalancedSinkhornOptions = {
    epsilon: 0.8,
    rho_source: 1.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-12,
    max_iterations: 1000,
  }
  match
    unbalanced_sinkhorn_with_warm_start(
      [0.8, 1.1],
      [0.9, 0.6],
      [[0.0, 0.5], [0.3, 0.2]],
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((_, seed)) => {
      let saved_u = seed.log_u[0]
      let saved_v = seed.log_v[0]
      match
        unbalanced_sinkhorn_with_warm_start(
          [0.8, 1.1],
          [0.9, 0.6],
          [[0.0, 0.5], [0.3, 0.2]],
          options,
          Some(seed),
        ) {
        Err(_) => assert_true(false)
        Ok((continued, _)) => {
          assert_eq(seed.log_u[0], saved_u)
          assert_eq(seed.log_v[0], saved_v)
          continued.coupling[0][0] = 8.0
          assert_eq(seed.log_u[0], saved_u)
          assert_eq(seed.log_v[0], saved_v)
        }
      }
    }
  }
}

///|
test "dormant finite warm-start potentials skip epsilon migration" {
  let seed = UnbalancedSinkhornWarmStart::{
    log_u: [1.0e100],
    log_v: [0.0],
    epsilon: 1.0e308,
    rows: 1,
    columns: 1,
  }
  let options : UnbalancedSinkhornOptions = {
    epsilon: 1.0e-308,
    rho_source: 1.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-10,
    max_iterations: 10,
  }
  match
    unbalanced_sinkhorn_with_warm_start(
      [0.0],
      [1.0],
      [[0.0]],
      options,
      Some(seed),
    ) {
    Err(_) => assert_true(false)
    Ok((warm, returned_seed)) => {
      match unbalanced_sinkhorn([0.0], [1.0], [[0.0]], options) {
        Err(_) => assert_true(false)
        Ok(cold) => {
          assert_eq(warm.coupling, cold.coupling)
          assert_eq(warm.objective, cold.objective)
          assert_eq(warm.transport_cost, cold.transport_cost)
          assert_eq(warm.reference_kl, cold.reference_kl)
          assert_eq(warm.source_marginal_kl, cold.source_marginal_kl)
          assert_eq(warm.target_marginal_kl, cold.target_marginal_kl)
        }
      }
      assert_eq(returned_seed.log_u, [0.0])
      assert_eq(returned_seed.log_v, [0.0])
    }
  }

  let zero_source_penalty_options : UnbalancedSinkhornOptions = {
    epsilon: 1.0e-308,
    rho_source: 0.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-10,
    max_iterations: 10,
  }
  match
    unbalanced_sinkhorn_with_warm_start(
      [1.0],
      [1.0],
      [[0.0]],
      zero_source_penalty_options,
      Some(seed),
    ) {
    Err(_) => assert_true(false)
    Ok((warm, returned_seed)) => {
      match
        unbalanced_sinkhorn([1.0], [1.0], [[0.0]], zero_source_penalty_options) {
        Err(_) => assert_true(false)
        Ok(cold) => {
          assert_eq(warm.coupling, cold.coupling)
          assert_eq(warm.objective, cold.objective)
        }
      }
      assert_eq(returned_seed.log_u, [0.0])
    }
  }

  let nonfinite_dormant_seed = UnbalancedSinkhornWarmStart::{
    log_u: [@math.exp(1000.0)],
    log_v: [0.0],
    epsilon: 1.0e308,
    rows: 1,
    columns: 1,
  }
  assert_true(
    unbalanced_sinkhorn_with_warm_start(
      [0.0],
      [1.0],
      [[0.0]],
      options,
      Some(nonfinite_dormant_seed),
    )
    is Err(UnbalancedSinkhornWarmStartError::InvalidWarmStart),
  )
}

///|
test "unrepresentable public epsilon migration reports numerical failure" {
  let seed = UnbalancedSinkhornWarmStart::{
    log_u: [1.0e100],
    log_v: [0.0],
    epsilon: 1.0e308,
    rows: 1,
    columns: 1,
  }
  assert_true(
    unbalanced_sinkhorn_with_warm_start(
      [1.0],
      [1.0],
      [[0.0]],
      {
        epsilon: 1.0e-308,
        rho_source: 1.0,
        rho_target: 1.0,
        cost_weight: 1.0,
        tolerance: 1.0e-10,
        max_iterations: 10,
      },
      Some(seed),
    )
    is Err(
      UnbalancedSinkhornWarmStartError::SolverError(
        UnbalancedSinkhornError::NumericalFailure
      )
    ),
  )
  assert_eq(seed.log_u[0], 1.0e100)
}

///|
test "seed dimensions and backing arrays must both match the current supports" {
  let seed = UnbalancedSinkhornWarmStart::{
    log_u: [0.0],
    log_v: [0.0, 0.0],
    epsilon: 1.0,
    rows: 1,
    columns: 1,
  }
  assert_true(
    unbalanced_prepare_warm_start(Some(seed), [1.0], [1.0], {
      epsilon: 1.0,
      rho_source: 1.0,
      rho_target: 1.0,
      cost_weight: 1.0,
      tolerance: 1.0e-10,
      max_iterations: 10,
    })
    is Err(UnbalancedSinkhornWarmStartError::InvalidWarmStart),
  )
}
