///|
/// A regularized OT solve contributing to a balanced Sinkhorn divergence.
///
/// The result retains each underlying solve, including its coupling and
/// convergence diagnostics. `value` uses the three
/// `regularized_objective` values exactly as returned by `balanced_sinkhorn`.
pub(all) struct SinkhornDivergenceResult {
  value : Double
  cross : SinkhornResult
  source_self : SinkhornResult
  target_self : SinkhornResult
}

///|
/// Shape, symmetry, wrapped-solver, and arithmetic errors for
/// `balanced_sinkhorn_divergence`.
pub(all) enum SinkhornDivergenceError {
  SourceSelfCostRowCountMismatch
  SourceSelfCostColumnCountMismatch(Int)
  TargetSelfCostRowCountMismatch
  TargetSelfCostColumnCountMismatch(Int)
  AsymmetricSourceSelfCost(Int, Int)
  AsymmetricTargetSelfCost(Int, Int)
  CrossTransport(SinkhornError)
  SourceSelfTransport(SinkhornError)
  TargetSelfTransport(SinkhornError)
  NumericalFailure
}

///|
/// Compute the balanced entropy-regularized Sinkhorn divergence
///
/// `OT_epsilon(source, target; cross_cost)`
/// `- 0.5 * OT_epsilon(source, source; source_self_cost)`
/// `- 0.5 * OT_epsilon(target, target; target_self_cost)`.
///
/// Each `OT_epsilon` value is exactly the `regularized_objective` from
/// `balanced_sinkhorn`, namely `<P, C> + epsilon * sum(P * (log(P) - 1))`.
/// The source and target self-cost matrices must be square for their
/// respective marginals and symmetric up to relative/absolute tolerance
/// `1.0e-12 * max(1, |x|, |y|)`. The cross-cost matrix is validated by the
/// underlying solver. Solver errors are wrapped with the solve that produced
/// them, and non-finite final arithmetic returns `NumericalFailure`.
///
/// This API does not promise a nonnegative result for arbitrary cost matrices.
/// Nonnegativity requires suitable mutually consistent ground-cost assumptions
/// (for example, costs induced by an appropriate metric); symmetry alone is not
/// sufficient. Each solve otherwise follows `balanced_sinkhorn` input rules,
/// including finite, nonnegative cost entries and balanced finite marginals.
pub fn balanced_sinkhorn_divergence(
  source : Array[Double],
  target : Array[Double],
  cross_cost : Array[Array[Double]],
  source_self_cost : Array[Array[Double]],
  target_self_cost : Array[Array[Double]],
  options : SinkhornOptions,
) -> Result[SinkhornDivergenceResult, SinkhornDivergenceError] {
  match validate_symmetric_square_cost(source_self_cost, source.length()) {
    Err(SelfCostShapeError::RowCountMismatch) =>
      return Err(SinkhornDivergenceError::SourceSelfCostRowCountMismatch)
    Err(SelfCostShapeError::ColumnCountMismatch(row)) =>
      return Err(
        SinkhornDivergenceError::SourceSelfCostColumnCountMismatch(row),
      )
    Err(SelfCostShapeError::Asymmetric(row, column)) =>
      return Err(SinkhornDivergenceError::AsymmetricSourceSelfCost(row, column))
    Ok(_) => ()
  }
  match validate_symmetric_square_cost(target_self_cost, target.length()) {
    Err(SelfCostShapeError::RowCountMismatch) =>
      return Err(SinkhornDivergenceError::TargetSelfCostRowCountMismatch)
    Err(SelfCostShapeError::ColumnCountMismatch(row)) =>
      return Err(
        SinkhornDivergenceError::TargetSelfCostColumnCountMismatch(row),
      )
    Err(SelfCostShapeError::Asymmetric(row, column)) =>
      return Err(SinkhornDivergenceError::AsymmetricTargetSelfCost(row, column))
    Ok(_) => ()
  }

  let cross = match balanced_sinkhorn(source, target, cross_cost, options) {
    Err(error) => return Err(SinkhornDivergenceError::CrossTransport(error))
    Ok(result) => result
  }
  let source_self = match
    balanced_sinkhorn(source, source, source_self_cost, options) {
    Err(error) =>
      return Err(SinkhornDivergenceError::SourceSelfTransport(error))
    Ok(result) => result
  }
  let target_self = match
    balanced_sinkhorn(target, target, target_self_cost, options) {
    Err(error) =>
      return Err(SinkhornDivergenceError::TargetSelfTransport(error))
    Ok(result) => result
  }
  let value = match
    sinkhorn_divergence_combine(
      cross.regularized_objective,
      source_self.regularized_objective,
      target_self.regularized_objective,
    ) {
    Err(error) => return Err(error)
    Ok(value) => value
  }
  Ok({ value, cross, source_self, target_self, })
}

///|
priv enum SelfCostShapeError {
  RowCountMismatch
  ColumnCountMismatch(Int)
  Asymmetric(Int, Int)
}

///|
fn validate_symmetric_square_cost(
  cost : Array[Array[Double]],
  size : Int,
) -> Result[Bool, SelfCostShapeError] {
  if cost.length() != size {
    return Err(SelfCostShapeError::RowCountMismatch)
  }
  for i in 0..<size {
    if cost[i].length() != size {
      return Err(SelfCostShapeError::ColumnCountMismatch(i))
    }
  }
  for i in 0..<size {
    for j in (i + 1)..<size {
      let left = cost[i][j]
      let right = cost[j][i]
      if is_finite(left) && is_finite(right) {
        let left_magnitude = abs(left)
        let right_magnitude = abs(right)
        let magnitude = if left_magnitude > right_magnitude {
          left_magnitude
        } else {
          right_magnitude
        }
        let scale = if magnitude > 1.0 { magnitude } else { 1.0 }
        if abs(left - right) > 1.0e-12 * scale {
          return Err(SelfCostShapeError::Asymmetric(i, j))
        }
      }
    }
  }
  Ok(true)
}

///|
fn sinkhorn_divergence_combine(
  cross_objective : Double,
  source_self_objective : Double,
  target_self_objective : Double,
) -> Result[Double, SinkhornDivergenceError] {
  if !is_finite(cross_objective) ||
    !is_finite(source_self_objective) ||
    !is_finite(target_self_objective) {
    return Err(SinkhornDivergenceError::NumericalFailure)
  }
  let cross_magnitude = abs(cross_objective)
  let source_magnitude = abs(source_self_objective)
  let target_magnitude = abs(target_self_objective)
  let first_max = if cross_magnitude > source_magnitude {
    cross_magnitude
  } else {
    source_magnitude
  }
  let scale = if first_max > target_magnitude {
    first_max
  } else {
    target_magnitude
  }
  if scale == 0.0 {
    return Ok(0.0)
  }
  let normalized = cross_objective / scale -
    0.5 * (source_self_objective / scale) -
    0.5 * (target_self_objective / scale)
  let value = normalized * scale
  if !is_finite(value) {
    return Err(SinkhornDivergenceError::NumericalFailure)
  }
  Ok(value)
}
