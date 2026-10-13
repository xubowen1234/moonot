///|
/// Termination state of a regularized discrete Wasserstein barycenter solve.
pub(all) enum BarycenterTermination {
  Converged
  IterationLimit
}

///|
/// The support weights, objective estimate, and convergence diagnostics of a
/// regularized barycenter.
pub(all) struct BarycenterResult {
  barycenter : Array[Double]
  weighted_objective : Double
  maximum_transport_residual : Double
  log_barycenter_residual : Double
  iterations : Int
  termination : BarycenterTermination
}

///|
/// Validation and numerical errors reported by `regularized_barycenter`.
pub(all) enum BarycenterError {
  EmptyMeasures
  WeightCountMismatch
  EmptySupport
  EmptyMeasure(Int)
  CostRowCountMismatch
  CostColumnCountMismatch(Int)
  NonFiniteWeight(Int)
  NonPositiveWeight(Int)
  NonFiniteWeightTotal
  NonFiniteMass(Int, Int)
  NegativeMass(Int, Int)
  NonFiniteMeasureTotal(Int)
  UnequalMeasureTotal(Int)
  NonFiniteCost(Int, Int)
  NegativeCost(Int, Int)
  InvalidEpsilon
  InvalidTolerance
  InvalidIterationLimit
  SolverError(SinkhornError)
  NumericalFailure
}

///|
/// Compute the fixed-support entropy-regularized Wasserstein barycenter.
///
/// Every input measure and the barycenter share one support with equal total
/// mass. Weights must be finite and strictly positive; they are normalized
/// internally. The common square cost matrix must be finite and nonnegative.
/// The iteration uses log-domain generalized Sinkhorn updates. Its tolerance
/// bounds the maximum change in normalized log barycenter between iterations.
/// The objective is the weighted sum of balanced Sinkhorn objectives, computed
/// after the barycenter iteration. The reported transport residual is the
/// largest marginal residual among those objective-evaluation solves.
pub fn regularized_barycenter(
  measures : Array[Array[Double]],
  weights : Array[Double],
  cost : Array[Array[Double]],
  options : SinkhornOptions,
) -> Result[BarycenterResult, BarycenterError] {
  let measure_count = measures.length()
  if measure_count == 0 {
    return Err(BarycenterError::EmptyMeasures)
  }
  if weights.length() != measure_count {
    return Err(BarycenterError::WeightCountMismatch)
  }
  if !is_finite(options.epsilon) || options.epsilon <= 0.0 {
    return Err(BarycenterError::InvalidEpsilon)
  }
  if !is_finite(options.tolerance) || options.tolerance <= 0.0 {
    return Err(BarycenterError::InvalidTolerance)
  }
  if options.max_iterations <= 0 {
    return Err(BarycenterError::InvalidIterationLimit)
  }

  let support_size = measures[0].length()
  if support_size == 0 {
    return Err(BarycenterError::EmptySupport)
  }
  if cost.length() != support_size {
    return Err(BarycenterError::CostRowCountMismatch)
  }
  for i in 0..<support_size {
    if cost[i].length() != support_size {
      return Err(BarycenterError::CostColumnCountMismatch(i))
    }
    for j in 0..<support_size {
      if !is_finite(cost[i][j]) {
        return Err(BarycenterError::NonFiniteCost(i, j))
      }
      if cost[i][j] < 0.0 {
        return Err(BarycenterError::NegativeCost(i, j))
      }
    }
  }

  let mut maximum_weight = 0.0
  for k in 0..<measure_count {
    if !is_finite(weights[k]) {
      return Err(BarycenterError::NonFiniteWeight(k))
    }
    if weights[k] <= 0.0 {
      return Err(BarycenterError::NonPositiveWeight(k))
    }
    if weights[k] > maximum_weight {
      maximum_weight = weights[k]
    }
  }
  let scaled_weights = Array::make(measure_count, 0.0)
  let mut scaled_weight_total = 0.0
  for k in 0..<measure_count {
    scaled_weights[k] = weights[k] / maximum_weight
    scaled_weight_total = scaled_weight_total + scaled_weights[k]
  }
  if !is_finite(scaled_weight_total) || scaled_weight_total <= 0.0 {
    return Err(BarycenterError::NonFiniteWeightTotal)
  }
  for k in 0..<measure_count {
    scaled_weights[k] = scaled_weights[k] / scaled_weight_total
  }

  let mut common_total = 0.0
  let log_measures : Array[Array[Double]] = []
  for k in 0..<measure_count {
    if measures[k].length() != support_size {
      return Err(BarycenterError::EmptyMeasure(k))
    }
    let log_measure = Array::make(support_size, 0.0)
    let mut measure_total = 0.0
    for i in 0..<support_size {
      let mass = measures[k][i]
      if !is_finite(mass) {
        return Err(BarycenterError::NonFiniteMass(k, i))
      }
      if mass < 0.0 {
        return Err(BarycenterError::NegativeMass(k, i))
      }
      measure_total = measure_total + mass
      if !is_finite(measure_total) {
        return Err(BarycenterError::NonFiniteMeasureTotal(k))
      }
      if mass > 0.0 {
        log_measure[i] = @math.ln(mass)
      }
    }
    if k == 0 {
      common_total = measure_total
    } else if measure_total > 0.0 || common_total > 0.0 {
      let scale = if measure_total > common_total {
        measure_total
      } else {
        common_total
      }
      if abs(measure_total - common_total) > 1.0e-12 * scale {
        return Err(BarycenterError::UnequalMeasureTotal(k))
      }
    }
    log_measures.push(log_measure)
  }

  let zero_barycenter = Array::make(support_size, 0.0)
  if common_total == 0.0 {
    return Ok({
      barycenter: zero_barycenter,
      weighted_objective: 0.0,
      maximum_transport_residual: 0.0,
      log_barycenter_residual: 0.0,
      iterations: 0,
      termination: BarycenterTermination::Converged,
    })
  }

  let mut minimum_cost = cost[0][0]
  for i in 0..<support_size {
    for j in 0..<support_size {
      if cost[i][j] < minimum_cost {
        minimum_cost = cost[i][j]
      }
    }
  }
  let log_kernel = Array::make(support_size, Array::make(support_size, 0.0))
  for i in 0..<support_size {
    let row = Array::make(support_size, 0.0)
    for j in 0..<support_size {
      let scaled_cost = (cost[i][j] - minimum_cost) / options.epsilon
      if !is_finite(scaled_cost) {
        return Err(BarycenterError::NumericalFailure)
      }
      row[j] = -scaled_cost
    }
    log_kernel[i] = row
  }

  let log_scalings : Array[Array[Double]] = []
  for _ in 0..<measure_count {
    log_scalings.push(Array::make(support_size, 0.0))
  }
  let mut log_barycenter = Array::make(support_size, 0.0)
  let mut log_barycenter_residual = 0.0
  let mut iterations = 0
  let mut termination = BarycenterTermination::IterationLimit
  for iteration in 1..<=options.max_iterations {
    let current = barycenter_scaling_state(
      measures, scaled_weights, log_kernel, log_scalings, common_total,
    )
    let previous = current.log_barycenter

    for k in 0..<measure_count {
      let log_row_scale = Array::make(support_size, 0.0)
      for i in 0..<support_size {
        log_row_scale[i] = previous[i] - current.log_kernel_products[k][i]
        if !is_finite(log_row_scale[i]) {
          return Err(BarycenterError::NumericalFailure)
        }
      }
      for j in 0..<support_size {
        if measures[k][j] > 0.0 {
          let terms : Array[Double] = []
          for i in 0..<support_size {
            terms.push(log_kernel[i][j] + log_row_scale[i])
          }
          let normalizer = logsumexp(terms)
          log_scalings[k][j] = log_measures[k][j] - normalizer
          if !is_finite(log_scalings[k][j]) {
            return Err(BarycenterError::NumericalFailure)
          }
        }
      }
    }

    let next = barycenter_scaling_state(
      measures, scaled_weights, log_kernel, log_scalings, common_total,
    )
    log_barycenter_residual = 0.0
    for i in 0..<support_size {
      let change = abs(next.log_barycenter[i] - previous[i])
      if !is_finite(change) {
        return Err(BarycenterError::NumericalFailure)
      }
      if change > log_barycenter_residual {
        log_barycenter_residual = change
      }
    }
    log_barycenter = next.log_barycenter
    iterations = iteration
    if log_barycenter_residual <= options.tolerance {
      termination = BarycenterTermination::Converged
      break
    }
  }

  let barycenter = barycenter_materialize(log_barycenter, common_total)
  match barycenter {
    Err(_) => Err(BarycenterError::NumericalFailure)
    Ok(mass) =>
      barycenter_evaluate_objective(
        mass, measures, scaled_weights, cost, options, log_barycenter_residual, iterations,
        termination,
      )
  }
}

///|
priv struct BarycenterScalingState {
  log_barycenter : Array[Double]
  log_kernel_products : Array[Array[Double]]
}

///|
fn barycenter_scaling_state(
  measures : Array[Array[Double]],
  weights : Array[Double],
  log_kernel : Array[Array[Double]],
  log_scalings : Array[Array[Double]],
  total_mass : Double,
) -> BarycenterScalingState {
  let support_size = log_kernel.length()
  let kernel_products : Array[Array[Double]] = []
  for _ in 0..<measures.length() {
    kernel_products.push(Array::make(support_size, 0.0))
  }
  for k in 0..<measures.length() {
    for i in 0..<support_size {
      let terms : Array[Double] = []
      for j in 0..<support_size {
        if measures[k][j] > 0.0 {
          terms.push(log_kernel[i][j] + log_scalings[k][j])
        }
      }
      kernel_products[k][i] = logsumexp(terms)
    }
  }

  let weighted_logs = Array::make(support_size, 0.0)
  for i in 0..<support_size {
    for k in 0..<measures.length() {
      weighted_logs[i] = weighted_logs[i] + weights[k] * kernel_products[k][i]
    }
  }
  let log_total = logsumexp(weighted_logs)
  let normalized_log_barycenter = Array::make(support_size, 0.0)
  let total_offset = @math.ln(total_mass) - log_total
  for i in 0..<support_size {
    normalized_log_barycenter[i] = weighted_logs[i] + total_offset
  }
  {
    log_barycenter: normalized_log_barycenter,
    log_kernel_products: kernel_products,
  }
}

///|
fn barycenter_materialize(
  log_barycenter : Array[Double],
  total_mass : Double,
) -> Result[Array[Double], BarycenterError] {
  let barycenter = Array::make(log_barycenter.length(), 0.0)
  let mut actual_total = 0.0
  for i in 0..<log_barycenter.length() {
    barycenter[i] = @math.exp(log_barycenter[i])
    if !is_finite(barycenter[i]) {
      return Err(BarycenterError::NumericalFailure)
    }
    actual_total = actual_total + barycenter[i]
    if !is_finite(actual_total) {
      return Err(BarycenterError::NumericalFailure)
    }
  }
  if actual_total <= 0.0 {
    return Err(BarycenterError::NumericalFailure)
  }
  let correction = total_mass / actual_total
  if !is_finite(correction) || correction <= 0.0 {
    return Err(BarycenterError::NumericalFailure)
  }
  for i in 0..<barycenter.length() {
    barycenter[i] = barycenter[i] * correction
  }
  Ok(barycenter)
}

///|
fn barycenter_evaluate_objective(
  barycenter : Array[Double],
  measures : Array[Array[Double]],
  weights : Array[Double],
  cost : Array[Array[Double]],
  options : SinkhornOptions,
  log_residual : Double,
  iterations : Int,
  termination : BarycenterTermination,
) -> Result[BarycenterResult, BarycenterError] {
  let mut final_termination = termination
  let mut objective = 0.0
  let mut maximum_residual = 0.0
  for k in 0..<measures.length() {
    match balanced_sinkhorn(barycenter, measures[k], cost, options) {
      Err(error) => return Err(BarycenterError::SolverError(error))
      Ok(solution) => {
        objective = objective + weights[k] * solution.regularized_objective
        if !is_finite(objective) {
          return Err(BarycenterError::NumericalFailure)
        }
        if solution.marginal_residual > maximum_residual {
          maximum_residual = solution.marginal_residual
        }
        if solution.termination is SinkhornTermination::IterationLimit {
          final_termination = BarycenterTermination::IterationLimit
        }
      }
    }
  }
  Ok({
    barycenter,
    weighted_objective: objective,
    maximum_transport_residual: maximum_residual,
    log_barycenter_residual: log_residual,
    iterations,
    termination: final_termination,
  })
}
