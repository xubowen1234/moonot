///|
fn unbalanced_warm_results_close(
  left : @moonot.UnbalancedSinkhornResult,
  right : @moonot.UnbalancedSinkhornResult,
  tolerance : Double,
) -> Bool {
  if !unbalanced_approximately_equal(left.objective, right.objective, tolerance) ||
    !unbalanced_approximately_equal(
      left.transport_cost,
      right.transport_cost,
      tolerance,
    ) ||
    !unbalanced_approximately_equal(
      left.reference_kl,
      right.reference_kl,
      tolerance,
    ) ||
    !unbalanced_approximately_equal(
      left.source_marginal_kl,
      right.source_marginal_kl,
      tolerance,
    ) ||
    !unbalanced_approximately_equal(
      left.target_marginal_kl,
      right.target_marginal_kl,
      tolerance,
    ) {
    return false
  }
  for i in 0..<left.coupling.length() {
    for j in 0..<left.coupling[i].length() {
      if !unbalanced_approximately_equal(
          left.coupling[i][j],
          right.coupling[i][j],
          tolerance,
        ) {
        return false
      }
    }
  }
  true
}

///|
fn unbalanced_termination_equal(
  left : @moonot.UnbalancedSinkhornTermination,
  right : @moonot.UnbalancedSinkhornTermination,
) -> Bool {
  match (left, right) {
    (Converged, Converged) => true
    (IterationLimit, IterationLimit) => true
    _ => false
  }
}

///|
test "unbalanced warm None exactly preserves the cold solver path" {
  let source = [0.7, 1.3]
  let target = [0.4, 0.9, 0.6]
  let cost = [[0.1, 0.4, 0.7], [0.3, 0.2, 0.8]]
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.65,
    rho_source: 1.2,
    rho_target: 0.8,
    cost_weight: 1.4,
    tolerance: 1.0e-12,
    max_iterations: 2000,
  }
  match
    (
      @moonot.unbalanced_sinkhorn(source, target, cost, options),
      @moonot.unbalanced_sinkhorn_with_warm_start(
        source,
        target,
        cost,
        options,
        None,
      ),
    ) {
    (Ok(cold), Ok((warm, _))) => {
      assert_eq(cold.iterations, warm.iterations)
      assert_eq(cold.scaling_residual, warm.scaling_residual)
      assert_true(
        unbalanced_termination_equal(cold.termination, warm.termination),
      )
      assert_eq(cold.objective, warm.objective)
      assert_eq(cold.transport_cost, warm.transport_cost)
      assert_eq(cold.reference_kl, warm.reference_kl)
      assert_eq(cold.source_marginal_kl, warm.source_marginal_kl)
      assert_eq(cold.target_marginal_kl, warm.target_marginal_kl)
      assert_eq(cold.coupling, warm.coupling)
    }
    _ => assert_true(false)
  }
}

///|
test "iteration-limited unbalanced seed resumes the exact subsequent sweeps" {
  let source = [1.0, 1.0]
  let target = [1.0, 1.0]
  let cost = [[0.0, 10.0], [10.0, 0.0]]
  let one_sweep : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.0,
    rho_source: 1.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-15,
    max_iterations: 1,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      one_sweep,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((first, first_seed)) => {
      assert_true(
        first.termination
        is @moonot.UnbalancedSinkhornTermination::IterationLimit,
      )
      let second = @moonot.unbalanced_sinkhorn_with_warm_start(
        source,
        target,
        cost,
        one_sweep,
        Some(first_seed),
      )
      let two_sweeps : @moonot.UnbalancedSinkhornOptions = {
        epsilon: 1.0,
        rho_source: 1.0,
        rho_target: 1.0,
        cost_weight: 1.0,
        tolerance: 1.0e-15,
        max_iterations: 2,
      }
      match
        (second, @moonot.unbalanced_sinkhorn(source, target, cost, two_sweeps)) {
        (Ok((continued, _)), Ok(uninterrupted)) => {
          assert_eq(continued.iterations, 1)
          assert_eq(continued.coupling, uninterrupted.coupling)
          assert_eq(continued.objective, uninterrupted.objective)
          assert_eq(continued.scaling_residual, uninterrupted.scaling_residual)
          assert_true(
            unbalanced_termination_equal(
              continued.termination,
              uninterrupted.termination,
            ),
          )
        }
        _ => assert_true(false)
      }
    }
  }
}

///|
test "epsilon continuation on a rectangular changed problem preserves stationarity" {
  let old_source = [1.0, 0.5]
  let old_target = [0.8, 0.7, 0.6]
  let old_cost = [[0.3, 0.5, 0.8], [0.6, 0.2, 0.4]]
  let old_options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.7,
    rho_source: 0.4,
    rho_target: 1.5,
    cost_weight: 0.7,
    tolerance: 1.0e-12,
    max_iterations: 3000,
  }
  let source = [0.7, 1.1]
  let target = [0.3, 0.9, 0.5]
  let cost = [[0.2, 0.9, 0.4], [0.8, 0.3, 0.6]]
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.55,
    rho_source: 1.8,
    rho_target: 0.65,
    cost_weight: 1.3,
    tolerance: 1.0e-12,
    max_iterations: 4000,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      old_source,
      old_target,
      old_cost,
      old_options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((_, seed)) =>
      match
        (
          @moonot.unbalanced_sinkhorn(source, target, cost, options),
          @moonot.unbalanced_sinkhorn_with_warm_start(
            source,
            target,
            cost,
            options,
            Some(seed),
          ),
        ) {
        (Ok(cold), Ok((warm, _))) => {
          assert_true(unbalanced_warm_results_close(cold, warm, 2.0e-10))
          for i in 0..<source.length() {
            let mut row = 0.0
            for j in 0..<target.length() {
              row = row + warm.coupling[i][j]
            }
            for j in 0..<target.length() {
              let mut column = 0.0
              for k in 0..<source.length() {
                column = column + warm.coupling[k][j]
              }
              let stationarity = options.cost_weight * cost[i][j] +
                options.epsilon *
                @math.ln(warm.coupling[i][j] / (source[i] * target[j])) +
                options.rho_source * @math.ln(row / source[i]) +
                options.rho_target * @math.ln(column / target[j])
              assert_true(stationarity > -2.0e-9 && stationarity < 2.0e-9)
            }
          }
        }
        _ => assert_true(false)
      }
  }
}

///|
test "warm unbalanced one-cell result matches the analytic objective components" {
  let source = 2.0
  let target = 3.0
  let cost = 4.0
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 2.0,
    rho_source: 1.0,
    rho_target: 3.0,
    cost_weight: 0.5,
    tolerance: 1.0e-13,
    max_iterations: 1000,
  }
  let seed_options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.3,
    rho_source: 0.8,
    rho_target: 0.2,
    cost_weight: 2.0,
    tolerance: 1.0e-12,
    max_iterations: 1000,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      [0.5],
      [4.0],
      [[0.7]],
      seed_options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((_, seed)) =>
      match
        @moonot.unbalanced_sinkhorn_with_warm_start(
          [source],
          [target],
          [[cost]],
          options,
          Some(seed),
        ) {
        Err(_) => assert_true(false)
        Ok((solution, _)) => {
          let weighted_cost = options.cost_weight * cost
          let mass = @math.exp(
            (
              -weighted_cost +
              options.epsilon * (@math.ln(source) + @math.ln(target)) +
              options.rho_source * @math.ln(source) +
              options.rho_target * @math.ln(target)
            ) /
            (options.epsilon + options.rho_source + options.rho_target),
          )
          let reference_kl = mass * (@math.ln(mass / (source * target)) - 1.0) +
            source * target
          let source_kl = mass * (@math.ln(mass / source) - 1.0) + source
          let target_kl = mass * (@math.ln(mass / target) - 1.0) + target
          assert_true(
            unbalanced_approximately_equal(
              solution.coupling[0][0],
              mass,
              1.0e-12,
            ),
          )
          assert_true(
            unbalanced_approximately_equal(
              solution.transport_cost,
              weighted_cost * mass,
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
            unbalanced_approximately_equal(
              solution.objective,
              weighted_cost * mass +
              options.epsilon * reference_kl +
              options.rho_source * source_kl +
              options.rho_target * target_kl,
              1.0e-11,
            ),
          )
        }
      }
  }
}

///|
test "support changes and zero penalties reset the corresponding warm scaling" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.8,
    rho_source: 1.2,
    rho_target: 0.9,
    cost_weight: 1.1,
    tolerance: 1.0e-12,
    max_iterations: 2500,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      [0.4, 0.8, 0.6],
      [0.5, 0.7],
      [[0.1, 0.5], [0.4, 0.2], [0.8, 0.3]],
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((_, seed)) => {
      let next_options : @moonot.UnbalancedSinkhornOptions = {
        epsilon: 0.8,
        rho_source: 0.0,
        rho_target: 0.9,
        cost_weight: 1.1,
        tolerance: 1.0e-12,
        max_iterations: 2500,
      }
      let source = [0.4, 0.0, 0.6]
      let target = [0.5, 0.7]
      let cost = [[0.1, 0.5], [0.4, 0.2], [0.8, 0.3]]
      match
        (
          @moonot.unbalanced_sinkhorn(source, target, cost, next_options),
          @moonot.unbalanced_sinkhorn_with_warm_start(
            source,
            target,
            cost,
            next_options,
            Some(seed),
          ),
        ) {
        (Ok(cold), Ok((warm, _))) => {
          assert_true(unbalanced_warm_results_close(cold, warm, 1.0e-12))
          assert_eq(warm.coupling[1][0], 0.0)
          assert_eq(warm.coupling[1][1], 0.0)
        }
        _ => assert_true(false)
      }
    }
  }
}

///|
test "warm zero-support solves keep objective constants and return neutral reusable state" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.7,
    rho_source: 2.0,
    rho_target: 0.5,
    cost_weight: 1.0,
    tolerance: 1.0e-10,
    max_iterations: 10,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      [0.0, 0.0],
      [2.0, 1.0],
      [[0.0, 1.0], [1.0, 0.0]],
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((solution, seed)) => {
      assert_eq(solution.iterations, 0)
      assert_eq(solution.objective, 1.5)
      assert_true(
        solution.termination is @moonot.UnbalancedSinkhornTermination::Converged,
      )
      match
        @moonot.unbalanced_sinkhorn_with_warm_start(
          [0.0, 0.0],
          [2.0, 1.0],
          [[0.0, 1.0], [1.0, 0.0]],
          options,
          Some(seed),
        ) {
        Err(_) => assert_true(false)
        Ok((reused, _)) => assert_eq(reused.objective, solution.objective)
      }
    }
  }
}

///|
test "unbalanced warm-start validation follows input errors and checks dimensions" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 1.0,
    rho_source: 1.0,
    rho_target: 1.0,
    cost_weight: 1.0,
    tolerance: 1.0e-10,
    max_iterations: 100,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      [1.0],
      [1.0],
      [[0.0]],
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((_, seed)) => {
      assert_true(
        @moonot.unbalanced_sinkhorn_with_warm_start(
          [1.0, 2.0],
          [1.0],
          [[0.0], [0.0]],
          options,
          Some(seed),
        )
        is Err(@moonot.UnbalancedSinkhornWarmStartError::InvalidWarmStart),
      )
      assert_true(
        @moonot.unbalanced_sinkhorn_with_warm_start(
          [1.0, 2.0],
          [1.0],
          [[0.0], [0.0]],
          {
            epsilon: -1.0,
            rho_source: 1.0,
            rho_target: 1.0,
            cost_weight: 1.0,
            tolerance: 1.0e-10,
            max_iterations: 100,
          },
          Some(seed),
        )
        is Err(
          @moonot.UnbalancedSinkhornWarmStartError::SolverError(
            @moonot.UnbalancedSinkhornError::InvalidEpsilon
          )
        ),
      )
    }
  }
}

///|
test "unbalanced cost shifts change transported mass as required by the objective" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.9,
    rho_source: 0.7,
    rho_target: 1.3,
    cost_weight: 1.4,
    tolerance: 1.0e-13,
    max_iterations: 1000,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      [2.0],
      [3.0],
      [[0.0]],
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((base, seed)) =>
      match
        @moonot.unbalanced_sinkhorn_with_warm_start(
          [2.0],
          [3.0],
          [[1.0]],
          options,
          Some(seed),
        ) {
        Err(_) => assert_true(false)
        Ok((shifted, _)) => {
          let expected_ratio = @math.exp(
            -options.cost_weight /
            (options.epsilon + options.rho_source + options.rho_target),
          )
          assert_true(shifted.coupling[0][0] < base.coupling[0][0])
          assert_true(
            unbalanced_approximately_equal(
              shifted.coupling[0][0] / base.coupling[0][0],
              expected_ratio,
              1.0e-13,
            ),
          )
        }
      }
  }
}

///|
test "coordinates entering support after a neutral warm seed match cold initialization" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.6,
    rho_source: 1.1,
    rho_target: 0.8,
    cost_weight: 1.2,
    tolerance: 1.0e-12,
    max_iterations: 3000,
  }
  let cost = [[0.1, 0.4], [0.6, 0.2], [0.3, 0.7]]
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      [0.4, 0.0, 0.6],
      [0.5, 0.7],
      cost,
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((_, seed)) => {
      let source = [0.4, 0.5, 0.6]
      match
        (
          @moonot.unbalanced_sinkhorn(source, [0.5, 0.7], cost, options),
          @moonot.unbalanced_sinkhorn_with_warm_start(
            source,
            [0.5, 0.7],
            cost,
            options,
            Some(seed),
          ),
        ) {
        (Ok(cold), Ok((warm, _))) => {
          assert_true(unbalanced_warm_results_close(cold, warm, 2.0e-10))
          assert_true(warm.coupling[1][0] > 0.0)
          assert_true(warm.coupling[1][1] > 0.0)
        }
        _ => assert_true(false)
      }
    }
  }
}

///|
test "warm zero-target support preserves source-side KL constants" {
  let options : @moonot.UnbalancedSinkhornOptions = {
    epsilon: 0.9,
    rho_source: 0.25,
    rho_target: 1.5,
    cost_weight: 1.0,
    tolerance: 1.0e-10,
    max_iterations: 1000,
  }
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      [2.0, 1.0],
      [1.0, 2.0],
      [[0.0, 0.2], [0.4, 0.1]],
      options,
      None,
    ) {
    Err(_) => assert_true(false)
    Ok((_, seed)) =>
      match
        @moonot.unbalanced_sinkhorn_with_warm_start(
          [2.0, 1.0],
          [0.0, 0.0],
          [[0.0, 0.2], [0.4, 0.1]],
          options,
          Some(seed),
        ) {
        Err(_) => assert_true(false)
        Ok((solution, _)) => {
          assert_eq(solution.iterations, 0)
          assert_eq(solution.coupling[0][0], 0.0)
          assert_eq(solution.coupling[1][1], 0.0)
          assert_eq(solution.source_marginal_kl, 3.0)
          assert_eq(solution.objective, 0.75)
        }
      }
  }
}
