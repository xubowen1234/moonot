///|
/// One completed continuation stage, in caller-supplied schedule order.
/// `result.iterations` counts full row/column updates at this epsilon only.
pub(all) struct SinkhornContinuationStage {
  options : SinkhornOptions
  result : SinkhornResult
}

///|
/// Work completed before a solver error. The failing stage is not included.
/// `last_seed` is the last completed stage's seed, or the caller's initial seed
/// when no stage completed. Iterations performed inside a failing solve are not
/// available from the underlying solver and are not counted.
pub(all) struct SinkhornContinuationProgress {
  stages : Array[SinkhornContinuationStage]
  total_iterations : Int64
  last_seed : SinkhornWarmStart?
}

///|
/// Completed schedule, including stages that reached their iteration limits.
/// `termination` is exclusively the final stage's termination, not an aggregate.
/// The final transport plan is `stages[stages.length() - 1].result`.
pub(all) struct SinkhornContinuationResult {
  stages : Array[SinkhornContinuationStage]
  total_iterations : Int64
  last_seed : SinkhornWarmStart
  termination : SinkhornTermination
}

///|
/// Schedule validation errors and lossless, stage-indexed solver errors.
/// All indices are zero-based. Validation completes before any solver runs.
/// `NonDecreasingEpsilon(i)` compares stage i to stage i - 1.
pub(all) enum SinkhornContinuationError {
  EmptySchedule
  InvalidEpsilon(Int)
  NonDecreasingEpsilon(Int)
  InvalidTolerance(Int)
  InvalidIterationLimit(Int)
  StageFailed(Int, SinkhornError, SinkhornContinuationProgress)
}

///|
/// Solve balanced transport along a strictly decreasing epsilon schedule.
///
/// Each schedule entry supplies a finite positive epsilon and absolute marginal
/// tolerance, plus a positive iteration limit for that stage (not a shared
/// budget). A single entry is valid. Entire-schedule validation precedes input
/// validation by the solver. At each index, epsilon, decreasing order, tolerance,
/// then iteration limit are checked, in that order.
///
/// Every stage runs `balanced_sinkhorn_with_warm_start`; the returned opaque
/// cost-unit seed is passed unchanged to the next stage. An intermediate
/// IterationLimit is recorded and does not stop continuation. Only the final
/// stage determines convergence. Solver errors stop immediately and preserve
/// both their original cause and the completed progress. No input or seed is
/// mutated, and no schedule is generated, sorted, or normalized.
pub fn balanced_sinkhorn_continuation(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  schedule : Array[SinkhornOptions],
  warm_start : SinkhornWarmStart?,
) -> Result[SinkhornContinuationResult, SinkhornContinuationError] {
  if schedule.is_empty() {
    return Err(SinkhornContinuationError::EmptySchedule)
  }
  for index in 0..<schedule.length() {
    let options = schedule[index]
    if !is_finite(options.epsilon) || options.epsilon <= 0.0 {
      return Err(SinkhornContinuationError::InvalidEpsilon(index))
    }
    if index > 0 && options.epsilon >= schedule[index - 1].epsilon {
      return Err(SinkhornContinuationError::NonDecreasingEpsilon(index))
    }
    if !is_finite(options.tolerance) || options.tolerance <= 0.0 {
      return Err(SinkhornContinuationError::InvalidTolerance(index))
    }
    if options.max_iterations <= 0 {
      return Err(SinkhornContinuationError::InvalidIterationLimit(index))
    }
  }
  let stages : Array[SinkhornContinuationStage] = []
  let (first_result, first_seed) = match
    balanced_sinkhorn_with_warm_start(
      source,
      target,
      cost,
      schedule[0],
      warm_start,
    ) {
    Err(error) =>
      return Err(
        SinkhornContinuationError::StageFailed(0, error, {
          stages,
          total_iterations: 0L,
          last_seed: warm_start,
        }),
      )
    Ok(value) => value
  }
  stages.push({ options: schedule[0], result: first_result, })
  let mut total_iterations = first_result.iterations.to_int64()
  let mut last_seed = first_seed
  let mut termination = first_result.termination
  for index in 1..<schedule.length() {
    let (result, seed) = match
      balanced_sinkhorn_with_warm_start(
        source,
        target,
        cost,
        schedule[index],
        Some(last_seed),
      ) {
      Err(error) =>
        return Err(
          SinkhornContinuationError::StageFailed(index, error, {
            stages,
            total_iterations,
            last_seed: Some(last_seed),
          }),
        )
      Ok(value) => value
    }
    stages.push({ options: schedule[index], result, })
    total_iterations = total_iterations + result.iterations.to_int64()
    last_seed = seed
    termination = result.termination
  }
  Ok({ stages, total_iterations, last_seed, termination, })
}
