///|
fn continuation_ok(
  outcome : Result[
    @moonot.SinkhornContinuationResult,
    @moonot.SinkhornContinuationError,
  ],
) -> @moonot.SinkhornContinuationResult {
  match outcome {
    Ok(result) => result
    Err(_) => abort("unexpected continuation error")
  }
}

///|
fn continuation_warm_ok(
  outcome : Result[
    (@moonot.SinkhornResult, @moonot.SinkhornWarmStart),
    @moonot.SinkhornError,
  ],
) -> (@moonot.SinkhornResult, @moonot.SinkhornWarmStart) {
  match outcome {
    Ok(result) => result
    Err(_) => abort("unexpected warm solver error")
  }
}

///|
fn continuation_cold_ok(
  outcome : Result[@moonot.SinkhornResult, @moonot.SinkhornError],
) -> @moonot.SinkhornResult {
  match outcome {
    Ok(result) => result
    Err(_) => abort("unexpected cold solver error")
  }
}

///|
fn continuation_assert_close(
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
test "continuation analytic symmetric plan agrees with cold final solver" {
  let schedule : Array[@moonot.SinkhornOptions] = [
    { epsilon: 2.0, tolerance: 1.0e-6, max_iterations: 3, },
    { epsilon: 1.0, tolerance: 1.0e-9, max_iterations: 5, },
    { epsilon: 0.25, tolerance: 1.0e-12, max_iterations: 100, },
  ]
  let source = [0.5, 0.5]
  let cost = [[0.0, 1.0], [1.0, 0.0]]
  let result = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(source, source, cost, schedule, None),
  )
  let cold = continuation_cold_ok(
    @moonot.balanced_sinkhorn(source, source, cost, schedule[2]),
  )
  assert_eq(result.stages.length(), 3)
  assert_true(result.termination is @moonot.SinkhornTermination::Converged)
  let mut count = 0L
  for index in 0..<3 {
    let stage = result.stages[index]
    assert_eq(stage.options.epsilon, schedule[index].epsilon)
    assert_eq(stage.options.tolerance, schedule[index].tolerance)
    assert_eq(stage.options.max_iterations, schedule[index].max_iterations)
    assert_true(stage.result.iterations <= schedule[index].max_iterations)
    assert_true(stage.result.marginal_residual <= schedule[index].tolerance)
    let diagonal = 0.5 / (1.0 + @math.exp(-1.0 / schedule[index].epsilon))
    continuation_assert_close(stage.result.coupling[0][0], diagonal, 1.0e-12)
    continuation_assert_close(
      stage.result.coupling[0][1],
      0.5 - diagonal,
      1.0e-12,
    )
    continuation_assert_close(
      stage.result.coupling[1][0],
      0.5 - diagonal,
      1.0e-12,
    )
    continuation_assert_close(stage.result.coupling[1][1], diagonal, 1.0e-12)
    count = count + stage.result.iterations.to_int64()
  }
  assert_eq(result.total_iterations, count)
  let final_solution = result.stages[2].result
  for i in 0..<2 {
    for j in 0..<2 {
      continuation_assert_close(
        final_solution.coupling[i][j],
        cold.coupling[i][j],
        1.0e-12,
      )
    }
  }
  continuation_assert_close(
    final_solution.transport_cost,
    cold.transport_cost,
    1.0e-12,
  )
  continuation_assert_close(
    final_solution.entropy_regularizer,
    cold.entropy_regularizer,
    1.0e-12,
  )
  continuation_assert_close(
    final_solution.regularized_objective,
    cold.regularized_objective,
    1.0e-12,
  )
}

///|
test "continuation single epsilon exactly matches warm solver with or without seed" {
  let source = [0.31, 0.28, 0.41]
  let target = [0.11, 0.53, 0.36]
  let cost = [[5.13, 0.15, 4.65], [2.24, 3.79, 0.89], [6.44, 4.28, 2.27]]
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.5,
    tolerance: 1.0e-10,
    max_iterations: 1000,
  }
  let (_, initial_seed) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      { epsilon: 2.0, tolerance: 1.0e-6, max_iterations: 10, },
      None,
    ),
  )
  for seed in [None, Some(initial_seed)] {
    let result = continuation_ok(
      @moonot.balanced_sinkhorn_continuation(
        source,
        target,
        cost,
        [options],
        seed,
      ),
    )
    let (direct, _) = continuation_warm_ok(
      @moonot.balanced_sinkhorn_with_warm_start(
        source, target, cost, options, seed,
      ),
    )
    assert_eq(result.stages.length(), 1)
    assert_eq(result.stages[0].result.coupling, direct.coupling)
    assert_eq(
      result.stages[0].result.regularized_objective,
      direct.regularized_objective,
    )
    assert_eq(result.total_iterations, direct.iterations.to_int64())
    assert_true(result.termination is @moonot.SinkhornTermination::Converged)
  }
}

///|
test "continuation intermediate limit does not determine final convergence" {
  let source = [0.31, 0.28, 0.41]
  let target = [0.11, 0.53, 0.36]
  let cost = [[5.13, 0.15, 4.65], [2.24, 3.79, 0.89], [6.44, 4.28, 2.27]]
  let schedule : Array[@moonot.SinkhornOptions] = [
    { epsilon: 2.0, tolerance: 1.0e-14, max_iterations: 1, },
    { epsilon: 0.5, tolerance: 1.0e-10, max_iterations: 2000, },
  ]
  let result = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(source, target, cost, schedule, None),
  )
  assert_true(
    result.stages[0].result.termination
    is @moonot.SinkhornTermination::IterationLimit,
  )
  assert_eq(result.stages[0].result.iterations, 1)
  assert_true(result.stages[0].result.marginal_residual > schedule[0].tolerance)
  assert_true(result.termination is @moonot.SinkhornTermination::Converged)
  assert_true(
    result.stages[1].result.marginal_residual <= schedule[1].tolerance,
  )
  let (first, seed) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      schedule[0],
      None,
    ),
  )
  let (last, _) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      schedule[1],
      Some(seed),
    ),
  )
  assert_eq(result.stages[0].result.coupling, first.coupling)
  assert_eq(result.stages[1].result.coupling, last.coupling)
  assert_eq(
    result.total_iterations,
    first.iterations.to_int64() + last.iterations.to_int64(),
  )
  let cold = continuation_cold_ok(
    @moonot.balanced_sinkhorn(source, target, cost, schedule[1]),
  )
  for i in 0..<3 {
    for j in 0..<3 {
      continuation_assert_close(
        last.coupling[i][j],
        cold.coupling[i][j],
        1.0e-8,
      )
    }
  }
}

///|
test "continuation final limit is successful but unconverged and seed is reusable" {
  let source = [0.5, 0.5]
  let cost = [[0.0, 0.0], [0.0, 10.0]]
  let schedule : Array[@moonot.SinkhornOptions] = [
    { epsilon: 2.0, tolerance: 1.0, max_iterations: 1, },
    { epsilon: 1.0, tolerance: 1.0e-15, max_iterations: 1, },
  ]
  let result = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(source, source, cost, schedule, None),
  )
  assert_true(
    result.stages[0].result.termination
    is @moonot.SinkhornTermination::Converged,
  )
  assert_true(result.termination is @moonot.SinkhornTermination::IterationLimit)
  assert_true(
    result.stages[1].result.termination
    is @moonot.SinkhornTermination::IterationLimit,
  )
  assert_eq(result.total_iterations, 2L)
  assert_eq(result.stages[1].result.iterations, 1)
  assert_true(result.stages[1].result.marginal_residual > 1.0e-15)
  let resumed = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(
      source,
      source,
      cost,
      [schedule[1]],
      Some(result.last_seed),
    ),
  )
  let (manual, _) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      source,
      cost,
      schedule[1],
      Some(result.last_seed),
    ),
  )
  assert_eq(resumed.stages[0].result.coupling, manual.coupling)
  assert_eq(resumed.total_iterations, 1L)
}

///|
test "continuation retains exact zero support rows and columns" {
  let source = [0.0, 0.25, 0.75]
  let target = [0.25, 0.0, 0.75]
  let cost = [[0.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 2.0, 0.0]]
  let schedule : Array[@moonot.SinkhornOptions] = [
    { epsilon: 1.0, tolerance: 1.0e-10, max_iterations: 1000, },
    { epsilon: 0.5, tolerance: 1.0e-12, max_iterations: 2000, },
  ]
  let result = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(source, target, cost, schedule, None),
  )
  assert_true(result.termination is @moonot.SinkhornTermination::Converged)
  for stage in result.stages {
    for index in 0..<3 {
      assert_eq(stage.result.coupling[0][index], 0.0)
      assert_eq(stage.result.coupling[index][1], 0.0)
    }
    assert_true(stage.result.marginal_residual <= stage.options.tolerance)
  }
}

///|
test "continuation all zero marginals complete every stage with no iterations" {
  let schedule : Array[@moonot.SinkhornOptions] = [
    { epsilon: 1.0, tolerance: 1.0e-9, max_iterations: 1, },
    { epsilon: 0.1, tolerance: 1.0e-12, max_iterations: 1, },
  ]
  let result = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(
      [0.0, 0.0],
      [0.0],
      [[2.0], [3.0]],
      schedule,
      None,
    ),
  )
  assert_eq(result.stages.length(), 2)
  assert_eq(result.total_iterations, 0L)
  assert_true(result.termination is @moonot.SinkhornTermination::Converged)
  for stage in result.stages {
    assert_eq(stage.result.coupling, [[0.0], [0.0]])
    assert_eq(stage.result.iterations, 0)
    assert_eq(stage.result.marginal_residual, 0.0)
    assert_eq(stage.result.regularized_objective, 0.0)
    assert_true(
      stage.result.termination is @moonot.SinkhornTermination::Converged,
    )
  }
  let resumed = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(
      [0.0, 0.0],
      [0.0],
      [[2.0], [3.0]],
      [schedule[1]],
      Some(result.last_seed),
    ),
  )
  assert_eq(resumed.total_iterations, 0L)
}

///|
test "continuation validates empty equal increasing and nonfinite epsilons" {
  let valid : @moonot.SinkhornOptions = {
    epsilon: 2.0,
    tolerance: 1.0e-9,
    max_iterations: 10,
  }
  assert_true(
    @moonot.balanced_sinkhorn_continuation([1.0], [1.0], [[0.0]], [], None)
    is Err(@moonot.SinkhornContinuationError::EmptySchedule),
  )
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  for epsilon in [0.0, -0.0, -1.0, infinity, -infinity, nan] {
    let invalid : @moonot.SinkhornOptions = {
      epsilon,
      tolerance: 1.0e-9,
      max_iterations: 10,
    }
    assert_true(
      @moonot.balanced_sinkhorn_continuation(
        [1.0],
        [1.0],
        [[0.0]],
        [invalid],
        None,
      )
      is Err(@moonot.SinkhornContinuationError::InvalidEpsilon(0)),
    )
    assert_true(
      @moonot.balanced_sinkhorn_continuation(
        [1.0],
        [1.0],
        [[0.0]],
        [valid, invalid],
        None,
      )
      is Err(@moonot.SinkhornContinuationError::InvalidEpsilon(1)),
    )
  }
  for epsilon in [2.0, 3.0] {
    let invalid : @moonot.SinkhornOptions = {
      epsilon,
      tolerance: 1.0e-9,
      max_iterations: 10,
    }
    assert_true(
      @moonot.balanced_sinkhorn_continuation(
        [1.0],
        [1.0],
        [[0.0]],
        [valid, invalid],
        None,
      )
      is Err(@moonot.SinkhornContinuationError::NonDecreasingEpsilon(1)),
    )
  }
}

///|
test "continuation validates every tolerance and budget before solving" {
  let valid : @moonot.SinkhornOptions = {
    epsilon: 2.0,
    tolerance: 1.0e-9,
    max_iterations: 10,
  }
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  for tolerance in [0.0, -1.0, infinity, -infinity, nan] {
    let invalid : @moonot.SinkhornOptions = {
      epsilon: 1.0,
      tolerance,
      max_iterations: 10,
    }
    assert_true(
      @moonot.balanced_sinkhorn_continuation(
        [1.0],
        [1.0],
        [[0.0]],
        [invalid],
        None,
      )
      is Err(@moonot.SinkhornContinuationError::InvalidTolerance(0)),
    )
    // Deliberately invalid input: a late schedule error must still take precedence.
    assert_true(
      @moonot.balanced_sinkhorn_continuation([], [], [], [valid, invalid], None)
      is Err(@moonot.SinkhornContinuationError::InvalidTolerance(1)),
    )
  }
  for max_iterations in [0, -1] {
    let invalid : @moonot.SinkhornOptions = {
      epsilon: 1.0,
      tolerance: 1.0e-9,
      max_iterations,
    }
    assert_true(
      @moonot.balanced_sinkhorn_continuation(
        [1.0],
        [1.0],
        [[0.0]],
        [invalid],
        None,
      )
      is Err(@moonot.SinkhornContinuationError::InvalidIterationLimit(0)),
    )
    assert_true(
      @moonot.balanced_sinkhorn_continuation([], [], [], [valid, invalid], None)
      is Err(@moonot.SinkhornContinuationError::InvalidIterationLimit(1)),
    )
  }
}

///|
test "continuation mismatched totals and shape errors preserve their cause" {
  let schedule : Array[@moonot.SinkhornOptions] = [
    { epsilon: 1.0, tolerance: 1.0e-9, max_iterations: 10, },
  ]
  match
    @moonot.balanced_sinkhorn_continuation(
      [1.0],
      [0.5],
      [[0.0]],
      schedule,
      None,
    ) {
    Err(
      @moonot.SinkhornContinuationError::StageFailed(
        0,
        @moonot.SinkhornError::UnequalTotalMass,
        progress
      )
    ) => {
      assert_eq(progress.stages.length(), 0)
      assert_eq(progress.total_iterations, 0L)
      assert_true(progress.last_seed is None)
    }
    _ => assert_true(false)
  }
  assert_true(
    @moonot.balanced_sinkhorn_continuation([1.0], [1.0], [], schedule, None)
    is Err(
      @moonot.SinkhornContinuationError::StageFailed(
        0,
        @moonot.SinkhornError::CostRowCountMismatch,
        _
      )
    ),
  )
  assert_true(
    @moonot.balanced_sinkhorn_continuation(
      [1.0],
      [1.0],
      [[-1.0]],
      schedule,
      None,
    )
    is Err(
      @moonot.SinkhornContinuationError::StageFailed(
        0,
        @moonot.SinkhornError::NegativeCost(0, 0),
        _
      )
    ),
  )
}

///|
test "continuation invalid initial seed is reported without discarding it" {
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-9,
    max_iterations: 10,
  }
  let (_, seed) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      [0.5, 0.5],
      [0.5, 0.5],
      [[0.0, 1.0], [1.0, 0.0]],
      options,
      None,
    ),
  )
  match
    @moonot.balanced_sinkhorn_continuation(
      [1.0],
      [1.0],
      [[0.0]],
      [options],
      Some(seed),
    ) {
    Err(
      @moonot.SinkhornContinuationError::StageFailed(
        0,
        @moonot.SinkhornError::InvalidWarmStart,
        progress
      )
    ) => {
      assert_eq(progress.stages.length(), 0)
      assert_eq(progress.total_iterations, 0L)
      let (_, _) = continuation_warm_ok(
        @moonot.balanced_sinkhorn_with_warm_start(
          [0.5, 0.5],
          [0.5, 0.5],
          [[0.0, 1.0], [1.0, 0.0]],
          options,
          progress.last_seed,
        ),
      )
      assert_true(progress.last_seed is Some(_))
    }
    _ => assert_true(false)
  }
}

///|
test "continuation later numerical failure returns completed progress and last seed" {
  let source = [0.5, 0.5]
  let cost = [[0.0, 1.0], [1.0, 0.0]]
  let options : @moonot.SinkhornOptions = {
    epsilon: 1.0,
    tolerance: 1.0e-12,
    max_iterations: 10,
  }
  let schedule : Array[@moonot.SinkhornOptions] = [
    options,
    { epsilon: 1.0e-320, tolerance: 1.0e-12, max_iterations: 10, },
    { epsilon: 1.0e-321, tolerance: 1.0e-12, max_iterations: 10, },
  ]
  let (first, seed) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      source,
      cost,
      options,
      None,
    ),
  )
  match
    @moonot.balanced_sinkhorn_continuation(source, source, cost, schedule, None) {
    Err(
      @moonot.SinkhornContinuationError::StageFailed(
        1,
        @moonot.SinkhornError::NumericalFailure,
        progress
      )
    ) => {
      assert_eq(progress.stages.length(), 1)
      assert_eq(progress.stages[0].result.coupling, first.coupling)
      assert_eq(progress.total_iterations, first.iterations.to_int64())
      assert_true(progress.last_seed is Some(_))
      let (replayed, _) = continuation_warm_ok(
        @moonot.balanced_sinkhorn_with_warm_start(
          source,
          source,
          cost,
          options,
          progress.last_seed,
        ),
      )
      let (expected, _) = continuation_warm_ok(
        @moonot.balanced_sinkhorn_with_warm_start(
          source,
          source,
          cost,
          options,
          Some(seed),
        ),
      )
      assert_eq(replayed.coupling, expected.coupling)
      assert_eq(replayed.iterations, expected.iterations)
    }
    _ => assert_true(false)
  }
}

///|
test "continuation does not mutate caller inputs or initial and final opaque seeds" {
  let source = [0.31, 0.28, 0.41]
  let target = [0.11, 0.53, 0.36]
  let cost = [[5.13, 0.15, 4.65], [2.24, 3.79, 0.89], [6.44, 4.28, 2.27]]
  let options : @moonot.SinkhornOptions = {
    epsilon: 2.0,
    tolerance: 1.0e-12,
    max_iterations: 1,
  }
  let (_, seed) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      options,
      None,
    ),
  )
  let (before, _) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      options,
      Some(seed),
    ),
  )
  let schedule : Array[@moonot.SinkhornOptions] = [
    options,
    { epsilon: 0.5, tolerance: 1.0e-10, max_iterations: 1000, },
  ]
  let result = continuation_ok(
    @moonot.balanced_sinkhorn_continuation(
      source,
      target,
      cost,
      schedule,
      Some(seed),
    ),
  )
  assert_eq(source, [0.31, 0.28, 0.41])
  assert_eq(target, [0.11, 0.53, 0.36])
  assert_eq(cost, [[5.13, 0.15, 4.65], [2.24, 3.79, 0.89], [6.44, 4.28, 2.27]])
  assert_eq(schedule.length(), 2)
  assert_eq(schedule[0].epsilon, 2.0)
  assert_eq(schedule[0].tolerance, 1.0e-12)
  assert_eq(schedule[0].max_iterations, 1)
  assert_eq(schedule[1].epsilon, 0.5)
  assert_eq(schedule[1].tolerance, 1.0e-10)
  assert_eq(schedule[1].max_iterations, 1000)
  let (after, _) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      options,
      Some(seed),
    ),
  )
  assert_eq(after.coupling, before.coupling)
  assert_eq(after.iterations, before.iterations)
  let (final_before, _) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      options,
      Some(result.last_seed),
    ),
  )
  let last_value = result.stages[1].result.coupling[0][0]
  result.stages[0].result.coupling[0][0] = 123.0
  assert_eq(result.stages[1].result.coupling[0][0], last_value)
  result.stages[1].result.coupling[0][0] = 456.0
  let (final_after, _) = continuation_warm_ok(
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      options,
      Some(result.last_seed),
    ),
  )
  assert_eq(final_after.coupling, final_before.coupling)
  assert_eq(final_after.iterations, final_before.iterations)
}
