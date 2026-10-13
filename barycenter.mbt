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
/// Validation and numerical errors from `regularized_barycenter_with_costs`.
pub(all) enum BarycenterCostsError {
  InputError(BarycenterError)
  CostCountMismatch
  CostRowCountMismatch(Int)
  CostColumnCountMismatch(Int, Int)
  NonFiniteCost(Int, Int, Int)
  NegativeCost(Int, Int, Int)
  SolverError(Int, SinkhornError)
  NumericalFailure
}

///|
priv struct BarycenterValidatedInput {
  measures : Array[Array[Double]]
  scaled_weights : Array[Double]
  log_measures : Array[Array[Double]]
  support_size : Int
  total_mass : Double
}

///|
/// A shared matrix stays a single allocation for the legacy entry point;
/// heterogeneous matrices retain one matrix per input measure.
priv enum BarycenterCostMatrices {
  Shared(Array[Array[Double]])
  PerMeasure(Array[Array[Array[Double]]])
}

///|
priv enum BarycenterComputationError {
  SolverError(Int, SinkhornError)
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
  let support_size = match
    barycenter_validate_header(measures, weights, options) {
    Err(error) => return Err(error)
    Ok(size) => size
  }
  match barycenter_validate_shared_cost(cost, support_size) {
    Err(error) => return Err(error)
    Ok(_) => ()
  }
  let input = match
    barycenter_validate_measure_data(measures, weights, support_size) {
    Err(error) => return Err(error)
    Ok(validated) => validated
  }
  match
    barycenter_compute(input, BarycenterCostMatrices::Shared(cost), options) {
    Err(BarycenterComputationError::SolverError(_, error)) =>
      Err(BarycenterError::SolverError(error))
    Err(BarycenterComputationError::NumericalFailure) =>
      Err(BarycenterError::NumericalFailure)
    Ok(result) => Ok(result)
  }
}

///|
/// Compute a fixed-support regularized barycenter with one ground-cost matrix
/// per input measure.
///
/// `costs[k][i][j]` is the cost from common barycenter-support position `i` to
/// position `j` of `measures[k]`. Every matrix is square and has the same row
/// ordering, while its columns align with its paired measure. Costs need not be
/// symmetric or metric. The same epsilon and normalized positive weights apply
/// to all subproblems; this function does not learn or rescale costs.
pub fn regularized_barycenter_with_costs(
  measures : Array[Array[Double]],
  weights : Array[Double],
  costs : Array[Array[Array[Double]]],
  options : SinkhornOptions,
) -> Result[BarycenterResult, BarycenterCostsError] {
  let support_size = match
    barycenter_validate_header(measures, weights, options) {
    Err(error) => return Err(BarycenterCostsError::InputError(error))
    Ok(size) => size
  }
  match barycenter_validate_costs(costs, measures.length(), support_size) {
    Err(error) => return Err(error)
    Ok(_) => ()
  }
  let input = match
    barycenter_validate_measure_data(measures, weights, support_size) {
    Err(error) => return Err(BarycenterCostsError::InputError(error))
    Ok(validated) => validated
  }
  match
    barycenter_compute(
      input,
      BarycenterCostMatrices::PerMeasure(costs),
      options,
    ) {
    Err(BarycenterComputationError::SolverError(index, error)) =>
      Err(BarycenterCostsError::SolverError(index, error))
    Err(BarycenterComputationError::NumericalFailure) =>
      Err(BarycenterCostsError::NumericalFailure)
    Ok(result) => Ok(result)
  }
}

///|
fn barycenter_validate_header(
  measures : Array[Array[Double]],
  weights : Array[Double],
  options : SinkhornOptions,
) -> Result[Int, BarycenterError] {
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
  Ok(support_size)
}

///|
fn barycenter_validate_shared_cost(
  cost : Array[Array[Double]],
  support_size : Int,
) -> Result[Unit, BarycenterError] {
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
  Ok(())
}

///|
fn barycenter_validate_costs(
  costs : Array[Array[Array[Double]]],
  measure_count : Int,
  support_size : Int,
) -> Result[Unit, BarycenterCostsError] {
  if costs.length() != measure_count {
    return Err(BarycenterCostsError::CostCountMismatch)
  }
  for k in 0..<costs.length() {
    if costs[k].length() != support_size {
      return Err(BarycenterCostsError::CostRowCountMismatch(k))
    }
    for i in 0..<support_size {
      if costs[k][i].length() != support_size {
        return Err(BarycenterCostsError::CostColumnCountMismatch(k, i))
      }
      for j in 0..<support_size {
        if !is_finite(costs[k][i][j]) {
          return Err(BarycenterCostsError::NonFiniteCost(k, i, j))
        }
        if costs[k][i][j] < 0.0 {
          return Err(BarycenterCostsError::NegativeCost(k, i, j))
        }
      }
    }
  }
  Ok(())
}

///|
fn barycenter_validate_measure_data(
  measures : Array[Array[Double]],
  weights : Array[Double],
  support_size : Int,
) -> Result[BarycenterValidatedInput, BarycenterError] {
  let measure_count = measures.length()
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
  Ok({
    measures,
    scaled_weights,
    log_measures,
    support_size,
    total_mass: common_total,
  })
}

///|
fn barycenter_zero_result(support_size : Int) -> BarycenterResult {
  {
    barycenter: Array::make(support_size, 0.0),
    weighted_objective: 0.0,
    maximum_transport_residual: 0.0,
    log_barycenter_residual: 0.0,
    iterations: 0,
    termination: BarycenterTermination::Converged,
  }
}

///|
fn barycenter_compute(
  input : BarycenterValidatedInput,
  costs : BarycenterCostMatrices,
  options : SinkhornOptions,
) -> Result[BarycenterResult, BarycenterComputationError] {
  if input.total_mass == 0.0 {
    return Ok(barycenter_zero_result(input.support_size))
  }
  let log_kernels = match barycenter_build_log_kernels(costs, options.epsilon) {
    Err(error) => return Err(error)
    Ok(kernels) => kernels
  }
  let log_scalings : Array[Array[Double]] = []
  for _ in 0..<input.measures.length() {
    log_scalings.push(Array::make(input.support_size, 0.0))
  }
  let mut log_barycenter = Array::make(input.support_size, 0.0)
  let mut log_barycenter_residual = 0.0
  let mut iterations = 0
  let mut termination = BarycenterTermination::IterationLimit
  for iteration in 1..<=options.max_iterations {
    let current = barycenter_scaling_state(
      input.measures,
      input.scaled_weights,
      log_kernels,
      log_scalings,
      input.total_mass,
    )
    let previous = current.log_barycenter

    for k in 0..<input.measures.length() {
      let log_row_scale = Array::make(input.support_size, 0.0)
      for i in 0..<input.support_size {
        log_row_scale[i] = previous[i] - current.log_kernel_products[k][i]
        if !is_finite(log_row_scale[i]) {
          return Err(BarycenterComputationError::NumericalFailure)
        }
      }
      for j in 0..<input.support_size {
        if input.measures[k][j] > 0.0 {
          let terms : Array[Double] = []
          let log_kernel = barycenter_cost_matrix(log_kernels, k)
          for i in 0..<input.support_size {
            terms.push(log_kernel[i][j] + log_row_scale[i])
          }
          let normalizer = logsumexp(terms)
          log_scalings[k][j] = input.log_measures[k][j] - normalizer
          if !is_finite(log_scalings[k][j]) {
            return Err(BarycenterComputationError::NumericalFailure)
          }
        }
      }
    }

    let next = barycenter_scaling_state(
      input.measures,
      input.scaled_weights,
      log_kernels,
      log_scalings,
      input.total_mass,
    )
    log_barycenter_residual = 0.0
    for i in 0..<input.support_size {
      let change = abs(next.log_barycenter[i] - previous[i])
      if !is_finite(change) {
        return Err(BarycenterComputationError::NumericalFailure)
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

  match barycenter_materialize(log_barycenter, input.total_mass) {
    Err(_) => Err(BarycenterComputationError::NumericalFailure)
    Ok(barycenter) =>
      barycenter_evaluate_objective(
        barycenter,
        input.measures,
        input.scaled_weights,
        costs,
        options,
        log_barycenter_residual,
        iterations,
        termination,
      )
  }
}

///|
fn barycenter_cost_matrix(
  matrices : BarycenterCostMatrices,
  measure_index : Int,
) -> Array[Array[Double]] {
  match matrices {
    BarycenterCostMatrices::Shared(matrix) => matrix
    BarycenterCostMatrices::PerMeasure(per_measure) =>
      per_measure[measure_index]
  }
}

///|
fn barycenter_build_log_kernels(
  costs : BarycenterCostMatrices,
  epsilon : Double,
) -> Result[BarycenterCostMatrices, BarycenterComputationError] {
  match costs {
    BarycenterCostMatrices::Shared(cost) =>
      match barycenter_build_log_kernel(cost, epsilon) {
        Err(error) => Err(error)
        Ok(kernel) => Ok(BarycenterCostMatrices::Shared(kernel))
      }
    BarycenterCostMatrices::PerMeasure(per_measure) => {
      let kernels : Array[Array[Array[Double]]] = []
      for measure_index in 0..<per_measure.length() {
        match barycenter_build_log_kernel(per_measure[measure_index], epsilon) {
          Err(error) => return Err(error)
          Ok(kernel) => kernels.push(kernel)
        }
      }
      Ok(BarycenterCostMatrices::PerMeasure(kernels))
    }
  }
}

///|
fn barycenter_build_log_kernel(
  cost : Array[Array[Double]],
  epsilon : Double,
) -> Result[Array[Array[Double]], BarycenterComputationError] {
  let support_size = cost.length()
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
      let scaled_cost = (cost[i][j] - minimum_cost) / epsilon
      if !is_finite(scaled_cost) {
        return Err(BarycenterComputationError::NumericalFailure)
      }
      row[j] = -scaled_cost
    }
    log_kernel[i] = row
  }
  Ok(log_kernel)
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
  log_kernels : BarycenterCostMatrices,
  log_scalings : Array[Array[Double]],
  total_mass : Double,
) -> BarycenterScalingState {
  let support_size = match log_kernels {
    BarycenterCostMatrices::Shared(matrix) => matrix.length()
    BarycenterCostMatrices::PerMeasure(matrices) => matrices[0].length()
  }
  let kernel_products : Array[Array[Double]] = []
  for _ in 0..<measures.length() {
    kernel_products.push(Array::make(support_size, 0.0))
  }
  for k in 0..<measures.length() {
    let log_kernel = barycenter_cost_matrix(log_kernels, k)
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
  costs : BarycenterCostMatrices,
  options : SinkhornOptions,
  log_residual : Double,
  iterations : Int,
  termination : BarycenterTermination,
) -> Result[BarycenterResult, BarycenterComputationError] {
  let mut final_termination = termination
  let mut objective = 0.0
  let mut maximum_residual = 0.0
  for k in 0..<measures.length() {
    let cost = barycenter_cost_matrix(costs, k)
    match balanced_sinkhorn(barycenter, measures[k], cost, options) {
      Err(error) =>
        return Err(BarycenterComputationError::SolverError(k, error))
      Ok(solution) => {
        objective = objective + weights[k] * solution.regularized_objective
        if !is_finite(objective) {
          return Err(BarycenterComputationError::NumericalFailure)
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
