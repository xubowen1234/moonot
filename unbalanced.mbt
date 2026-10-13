///|
/// End state of a KL-unbalanced Sinkhorn iteration.
pub(all) enum UnbalancedSinkhornTermination {
  Converged
  IterationLimit
}

///|
/// Parameters for `unbalanced_sinkhorn`.
///
/// `cost_weight` multiplies every entry of the cost matrix. `tolerance` bounds
/// the maximum change in the log scaling factors between complete iterations.
pub(all) struct UnbalancedSinkhornOptions {
  epsilon : Double
  rho_source : Double
  rho_target : Double
  cost_weight : Double
  tolerance : Double
  max_iterations : Int
}

///|
/// Opaque log-scaling state from a KL-unbalanced Sinkhorn solve.
///
/// A seed is only an initialization for a solve with the same source and
/// target dimensions. Its arrays and source epsilon are private and copied on
/// reuse; coordinate meaning and ordering remain the caller's responsibility.
pub struct UnbalancedSinkhornWarmStart {
  log_u : Array[Double]
  log_v : Array[Double]
  epsilon : Double
  rows : Int
  columns : Int
}

///|
/// Plan, objective components, and diagnostics for a KL-unbalanced solve.
///
/// The objective is
/// `<P, cost_weight * C> + epsilon * KL(P || a tensor b)`
/// `+ rho_source * KL(P 1 || a) + rho_target * KL(P^T 1 || b)`.
/// Each KL is the generalized I-divergence `x log(x/y) - x + y`.
pub(all) struct UnbalancedSinkhornResult {
  coupling : Array[Array[Double]]
  objective : Double
  transport_cost : Double
  reference_kl : Double
  source_marginal_kl : Double
  target_marginal_kl : Double
  source_marginal_residual : Double
  target_marginal_residual : Double
  scaling_residual : Double
  iterations : Int
  termination : UnbalancedSinkhornTermination
}

///|
/// Validation and numerical errors from `unbalanced_sinkhorn`.
pub(all) enum UnbalancedSinkhornError {
  EmptySourceMarginal
  EmptyTargetMarginal
  CostRowCountMismatch
  CostColumnCountMismatch(Int)
  NegativeSourceMass(Int)
  NegativeTargetMass(Int)
  NonFiniteSourceMass(Int)
  NonFiniteTargetMass(Int)
  NegativeCost(Int, Int)
  NonFiniteCost(Int, Int)
  InvalidEpsilon
  InvalidSourcePenalty
  InvalidTargetPenalty
  InvalidCostWeight
  InvalidTolerance
  InvalidIterationLimit
  NumericalFailure
}

///|
/// Compatibility and solver errors from the unbalanced warm-start API.
pub(all) enum UnbalancedSinkhornWarmStartError {
  SolverError(UnbalancedSinkhornError)
  InvalidWarmStart
}

///|
/// Solve entropy-regularized unbalanced optimal transport in the log domain.
///
/// It minimizes
/// `sum(P[i,j] * cost_weight * C[i,j])`
/// `+ epsilon * KL(P || a tensor b)`
/// `+ rho_source * KL(P 1 || a) + rho_target * KL(P^T 1 || b)`,
/// where every KL is the generalized I-divergence
/// `sum(x * log(x / y) - x + y)`. The scalar `cost_weight` is finite and
/// nonnegative. Costs, marginals, and penalties must be finite and
/// nonnegative; epsilon and tolerance must be strictly positive. Marginals
/// need not have equal totals.
///
/// If a reference mass `y` is zero, finite generalized KL requires `x == 0`;
/// `KL(0 || 0)` is zero and `KL(0 || y)` for positive `y` is `y`. Since
/// `a tensor b` is the reference for the coupling KL, zero source or target
/// entries force the corresponding rows or columns of `P` to zero. A fully
/// zero source or target therefore has the unique zero coupling. The returned
/// scaling residual is the maximum absolute log-scaling update; marginal
/// residuals are maximum absolute differences from the supplied marginals and
/// are diagnostics, not convergence criteria.
pub fn unbalanced_sinkhorn(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : UnbalancedSinkhornOptions,
) -> Result[UnbalancedSinkhornResult, UnbalancedSinkhornError] {
  match unbalanced_validate_inputs(source, target, cost, options) {
    Err(error) => Err(error)
    Ok(_) =>
      match
        unbalanced_sinkhorn_internal(source, target, cost, options, None, false) {
        Err(error) => Err(error)
        Ok((result, _)) => Ok(result)
      }
  }
}

///|
/// Solve KL-unbalanced transport from an optional prior solve's scaling state.
///
/// The returned seed is reusable even when this call reaches its iteration
/// limit. A seed only initializes the next call: the new inputs, objective,
/// tolerance, and iteration budget are evaluated normally. No speedup is
/// guaranteed. Coordinates retain the caller-defined ordering.
pub fn unbalanced_sinkhorn_with_warm_start(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : UnbalancedSinkhornOptions,
  warm_start : UnbalancedSinkhornWarmStart?,
) -> Result[
  (UnbalancedSinkhornResult, UnbalancedSinkhornWarmStart),
  UnbalancedSinkhornWarmStartError,
] {
  match unbalanced_validate_inputs(source, target, cost, options) {
    Err(error) => Err(UnbalancedSinkhornWarmStartError::SolverError(error))
    Ok(_) =>
      match unbalanced_prepare_warm_start(warm_start, source, target, options) {
        Err(error) => Err(error)
        Ok(initial) =>
          match
            unbalanced_sinkhorn_internal(
              source, target, cost, options, initial, true,
            ) {
            Err(error) =>
              Err(UnbalancedSinkhornWarmStartError::SolverError(error))
            Ok((result, Some(seed))) => Ok((result, seed))
            Ok(_) =>
              Err(
                UnbalancedSinkhornWarmStartError::SolverError(
                  UnbalancedSinkhornError::NumericalFailure,
                ),
              )
          }
      }
  }
}

///|
fn unbalanced_validate_inputs(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : UnbalancedSinkhornOptions,
) -> Result[Unit, UnbalancedSinkhornError] {
  let rows = source.length()
  let columns = target.length()
  if rows == 0 {
    return Err(UnbalancedSinkhornError::EmptySourceMarginal)
  }
  if columns == 0 {
    return Err(UnbalancedSinkhornError::EmptyTargetMarginal)
  }
  if cost.length() != rows {
    return Err(UnbalancedSinkhornError::CostRowCountMismatch)
  }
  if !is_finite(options.epsilon) || options.epsilon <= 0.0 {
    return Err(UnbalancedSinkhornError::InvalidEpsilon)
  }
  if !is_finite(options.rho_source) || options.rho_source < 0.0 {
    return Err(UnbalancedSinkhornError::InvalidSourcePenalty)
  }
  if !is_finite(options.rho_target) || options.rho_target < 0.0 {
    return Err(UnbalancedSinkhornError::InvalidTargetPenalty)
  }
  if !is_finite(options.cost_weight) || options.cost_weight < 0.0 {
    return Err(UnbalancedSinkhornError::InvalidCostWeight)
  }
  if !is_finite(options.tolerance) || options.tolerance <= 0.0 {
    return Err(UnbalancedSinkhornError::InvalidTolerance)
  }
  if options.max_iterations <= 0 {
    return Err(UnbalancedSinkhornError::InvalidIterationLimit)
  }

  for i in 0..<rows {
    if !is_finite(source[i]) {
      return Err(UnbalancedSinkhornError::NonFiniteSourceMass(i))
    }
    if source[i] < 0.0 {
      return Err(UnbalancedSinkhornError::NegativeSourceMass(i))
    }
  }
  for j in 0..<columns {
    if !is_finite(target[j]) {
      return Err(UnbalancedSinkhornError::NonFiniteTargetMass(j))
    }
    if target[j] < 0.0 {
      return Err(UnbalancedSinkhornError::NegativeTargetMass(j))
    }
  }
  for i in 0..<rows {
    if cost[i].length() != columns {
      return Err(UnbalancedSinkhornError::CostColumnCountMismatch(i))
    }
    for j in 0..<columns {
      if !is_finite(cost[i][j]) {
        return Err(UnbalancedSinkhornError::NonFiniteCost(i, j))
      }
      if cost[i][j] < 0.0 {
        return Err(UnbalancedSinkhornError::NegativeCost(i, j))
      }
    }
  }

  Ok(())
}

///|
fn unbalanced_sinkhorn_internal(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : UnbalancedSinkhornOptions,
  initial_scalings : (Array[Double], Array[Double])?,
  capture_warm_start : Bool,
) -> Result[
  (UnbalancedSinkhornResult, UnbalancedSinkhornWarmStart?),
  UnbalancedSinkhornError,
] {
  let rows = source.length()
  let columns = target.length()

  let log_source = Array::make(rows, 0.0)
  let log_target = Array::make(columns, 0.0)
  let mut has_source_support = false
  let mut has_target_support = false
  for i in 0..<rows {
    if source[i] > 0.0 {
      log_source[i] = @math.ln(source[i])
      has_source_support = true
    }
  }
  for j in 0..<columns {
    if target[j] > 0.0 {
      log_target[j] = @math.ln(target[j])
      has_target_support = true
    }
  }

  let log_kernel = Array::make(rows, Array::make(columns, 0.0))
  for i in 0..<rows {
    let kernel_row = Array::make(columns, 0.0)
    if source[i] > 0.0 && has_target_support {
      for j in 0..<columns {
        if target[j] > 0.0 {
          match
            unbalanced_scaled_cost(
              cost[i][j],
              options.cost_weight,
              options.epsilon,
            ) {
            Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
            Ok(scaled_cost) => kernel_row[j] = -scaled_cost
          }
        }
      }
    }
    log_kernel[i] = kernel_row
  }

  let coupling = unbalanced_zero_plan(rows, columns)
  if !has_source_support || !has_target_support {
    return match
      unbalanced_make_result(
        coupling,
        source,
        target,
        cost,
        options,
        0.0,
        0,
        UnbalancedSinkhornTermination::Converged,
      ) {
      Err(error) => Err(error)
      Ok(result) => {
        let seed : UnbalancedSinkhornWarmStart? = if capture_warm_start {
          Some(
            unbalanced_make_seed(
              Array::make(rows, 0.0),
              Array::make(columns, 0.0),
              options.epsilon,
            ),
          )
        } else {
          None
        }
        Ok((result, seed))
      }
    }
  }

  let source_relaxation = unbalanced_relaxation(
    options.rho_source,
    options.epsilon,
  )
  let target_relaxation = unbalanced_relaxation(
    options.rho_target,
    options.epsilon,
  )
  let log_u = Array::make(rows, 0.0)
  let log_v = Array::make(columns, 0.0)
  match initial_scalings {
    None => ()
    Some((initial_u, initial_v)) => {
      for i in 0..<rows {
        if source[i] > 0.0 && options.rho_source > 0.0 {
          log_u[i] = initial_u[i]
        }
      }
      for j in 0..<columns {
        if target[j] > 0.0 && options.rho_target > 0.0 {
          log_v[j] = initial_v[j]
        }
      }
    }
  }
  let mut plan = coupling
  let mut scaling_residual = 0.0
  let mut iterations = 0
  let mut termination = UnbalancedSinkhornTermination::IterationLimit

  for iteration in 1..<=options.max_iterations {
    let old_u = Array::make(rows, 0.0)
    let old_v = Array::make(columns, 0.0)
    for i in 0..<rows {
      old_u[i] = log_u[i]
    }
    for j in 0..<columns {
      old_v[j] = log_v[j]
    }

    for i in 0..<rows {
      if source[i] > 0.0 {
        let terms : Array[Double] = []
        for j in 0..<columns {
          if target[j] > 0.0 {
            let value = log_target[j] + log_v[j] + log_kernel[i][j]
            if !is_finite(value) {
              return Err(UnbalancedSinkhornError::NumericalFailure)
            }
            terms.push(value)
          }
        }
        let normalizer = logsumexp(terms)
        log_u[i] = -source_relaxation * normalizer
        if !is_finite(log_u[i]) {
          return Err(UnbalancedSinkhornError::NumericalFailure)
        }
      }
    }

    for j in 0..<columns {
      if target[j] > 0.0 {
        let terms : Array[Double] = []
        for i in 0..<rows {
          if source[i] > 0.0 {
            let value = log_source[i] + log_u[i] + log_kernel[i][j]
            if !is_finite(value) {
              return Err(UnbalancedSinkhornError::NumericalFailure)
            }
            terms.push(value)
          }
        }
        let normalizer = logsumexp(terms)
        log_v[j] = -target_relaxation * normalizer
        if !is_finite(log_v[j]) {
          return Err(UnbalancedSinkhornError::NumericalFailure)
        }
      }
    }

    scaling_residual = 0.0
    for i in 0..<rows {
      let change = abs(log_u[i] - old_u[i])
      if !is_finite(change) {
        return Err(UnbalancedSinkhornError::NumericalFailure)
      }
      if change > scaling_residual {
        scaling_residual = change
      }
    }
    for j in 0..<columns {
      let change = abs(log_v[j] - old_v[j])
      if !is_finite(change) {
        return Err(UnbalancedSinkhornError::NumericalFailure)
      }
      if change > scaling_residual {
        scaling_residual = change
      }
    }

    let next_plan : Array[Array[Double]] = []
    for i in 0..<rows {
      let plan_row = Array::make(columns, 0.0)
      if source[i] > 0.0 {
        for j in 0..<columns {
          if target[j] > 0.0 {
            let log_mass = log_source[i] +
              log_target[j] +
              log_kernel[i][j] +
              log_u[i] +
              log_v[j]
            if !is_finite(log_mass) {
              return Err(UnbalancedSinkhornError::NumericalFailure)
            }
            plan_row[j] = @math.exp(log_mass)
            if !is_finite(plan_row[j]) {
              return Err(UnbalancedSinkhornError::NumericalFailure)
            }
          }
        }
      }
      next_plan.push(plan_row)
    }
    plan = next_plan
    iterations = iteration
    if scaling_residual <= options.tolerance {
      termination = UnbalancedSinkhornTermination::Converged
      break
    }
  }

  match
    unbalanced_make_result(
      plan, source, target, cost, options, scaling_residual, iterations, termination,
    ) {
    Err(error) => Err(error)
    Ok(result) => {
      let seed : UnbalancedSinkhornWarmStart? = if capture_warm_start {
        Some(unbalanced_make_seed(log_u, log_v, options.epsilon))
      } else {
        None
      }
      Ok((result, seed))
    }
  }
}

///|
fn unbalanced_prepare_warm_start(
  warm_start : UnbalancedSinkhornWarmStart?,
  source : Array[Double],
  target : Array[Double],
  options : UnbalancedSinkhornOptions,
) -> Result[(Array[Double], Array[Double])?, UnbalancedSinkhornWarmStartError] {
  match warm_start {
    None => Ok(None)
    Some(seed) => {
      let rows = source.length()
      let columns = target.length()
      if seed.rows != rows ||
        seed.columns != columns ||
        seed.log_u.length() != rows ||
        seed.log_v.length() != columns ||
        !is_finite(seed.epsilon) ||
        seed.epsilon <= 0.0 {
        return Err(UnbalancedSinkhornWarmStartError::InvalidWarmStart)
      }
      let log_u = Array::make(rows, 0.0)
      let log_v = Array::make(columns, 0.0)
      for i in 0..<rows {
        if !is_finite(seed.log_u[i]) {
          return Err(UnbalancedSinkhornWarmStartError::InvalidWarmStart)
        }
        match
          unbalanced_rescale_seed_log(
            seed.log_u[i],
            seed.epsilon,
            options.epsilon,
          ) {
          Err(_) =>
            return Err(
              UnbalancedSinkhornWarmStartError::SolverError(
                UnbalancedSinkhornError::NumericalFailure,
              ),
            )
          Ok(value) =>
            if source[i] > 0.0 && options.rho_source > 0.0 {
              log_u[i] = value
            }
        }
      }
      for j in 0..<columns {
        if !is_finite(seed.log_v[j]) {
          return Err(UnbalancedSinkhornWarmStartError::InvalidWarmStart)
        }
        match
          unbalanced_rescale_seed_log(
            seed.log_v[j],
            seed.epsilon,
            options.epsilon,
          ) {
          Err(_) =>
            return Err(
              UnbalancedSinkhornWarmStartError::SolverError(
                UnbalancedSinkhornError::NumericalFailure,
              ),
            )
          Ok(value) =>
            if target[j] > 0.0 && options.rho_target > 0.0 {
              log_v[j] = value
            }
        }
      }
      Ok(Some((log_u, log_v)))
    }
  }
}

///|
fn unbalanced_rescale_seed_log(
  value : Double,
  old_epsilon : Double,
  new_epsilon : Double,
) -> Result[Double, Unit] {
  if value == 0.0 || old_epsilon == new_epsilon {
    return Ok(value)
  }
  let ratio = old_epsilon / new_epsilon
  if is_finite(ratio) && ratio > 0.0 {
    let direct = value * ratio
    if is_finite(direct) {
      return Ok(direct)
    }
  }
  let staged = value / new_epsilon * old_epsilon
  if is_finite(staged) {
    return Ok(staged)
  }
  let log_magnitude = @math.ln(abs(value)) +
    @math.ln(old_epsilon) -
    @math.ln(new_epsilon)
  if !is_finite(log_magnitude) || log_magnitude > 709.782712893384 {
    return Err(())
  }
  if log_magnitude < -744.4400719213812 {
    return Ok(0.0)
  }
  let magnitude = @math.exp(log_magnitude)
  if !is_finite(magnitude) {
    return Err(())
  }
  Ok(if value < 0.0 { -magnitude } else { magnitude })
}

///|
fn unbalanced_make_seed(
  log_u : Array[Double],
  log_v : Array[Double],
  epsilon : Double,
) -> UnbalancedSinkhornWarmStart {
  let copied_u = Array::make(log_u.length(), 0.0)
  let copied_v = Array::make(log_v.length(), 0.0)
  for i in 0..<log_u.length() {
    copied_u[i] = log_u[i]
  }
  for j in 0..<log_v.length() {
    copied_v[j] = log_v[j]
  }
  {
    log_u: copied_u,
    log_v: copied_v,
    epsilon,
    rows: log_u.length(),
    columns: log_v.length(),
  }
}

///|
priv struct UnbalancedObjectiveParts {
  objective : Double
  transport_cost : Double
  reference_kl : Double
  source_marginal_kl : Double
  target_marginal_kl : Double
  source_marginal_residual : Double
  target_marginal_residual : Double
}

///|
fn unbalanced_make_result(
  coupling : Array[Array[Double]],
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : UnbalancedSinkhornOptions,
  scaling_residual : Double,
  iterations : Int,
  termination : UnbalancedSinkhornTermination,
) -> Result[UnbalancedSinkhornResult, UnbalancedSinkhornError] {
  match unbalanced_objective_parts(coupling, source, target, cost, options) {
    Err(_) => Err(UnbalancedSinkhornError::NumericalFailure)
    Ok(parts) =>
      Ok({
        coupling,
        objective: parts.objective,
        transport_cost: parts.transport_cost,
        reference_kl: parts.reference_kl,
        source_marginal_kl: parts.source_marginal_kl,
        target_marginal_kl: parts.target_marginal_kl,
        source_marginal_residual: parts.source_marginal_residual,
        target_marginal_residual: parts.target_marginal_residual,
        scaling_residual,
        iterations,
        termination,
      })
  }
}

///|
fn unbalanced_objective_parts(
  coupling : Array[Array[Double]],
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : UnbalancedSinkhornOptions,
) -> Result[UnbalancedObjectiveParts, UnbalancedSinkhornError] {
  let rows = source.length()
  let columns = target.length()
  let row_sums = Array::make(rows, 0.0)
  let column_sums = Array::make(columns, 0.0)
  let mut transport_cost = 0.0
  let mut reference_kl = 0.0

  for i in 0..<rows {
    for j in 0..<columns {
      let mass = coupling[i][j]
      row_sums[i] = row_sums[i] + mass
      column_sums[j] = column_sums[j] + mass
      if !is_finite(row_sums[i]) || !is_finite(column_sums[j]) {
        return Err(UnbalancedSinkhornError::NumericalFailure)
      }
      match unbalanced_product3(options.cost_weight, cost[i][j], mass) {
        Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
        Ok(value) => transport_cost = transport_cost + value
      }
      if !is_finite(transport_cost) {
        return Err(UnbalancedSinkhornError::NumericalFailure)
      }
      if source[i] > 0.0 && target[j] > 0.0 {
        let log_reference = @math.ln(source[i]) + @math.ln(target[j])
        match unbalanced_kl_from_log_reference(mass, log_reference) {
          Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
          Ok(value) => reference_kl = reference_kl + value
        }
        if !is_finite(reference_kl) {
          return Err(UnbalancedSinkhornError::NumericalFailure)
        }
      }
    }
  }

  let mut source_marginal_kl = 0.0
  let mut target_marginal_kl = 0.0
  let mut source_marginal_residual = 0.0
  let mut target_marginal_residual = 0.0
  for i in 0..<rows {
    let difference = abs(row_sums[i] - source[i])
    if !is_finite(difference) {
      return Err(UnbalancedSinkhornError::NumericalFailure)
    }
    if difference > source_marginal_residual {
      source_marginal_residual = difference
    }
    if source[i] > 0.0 {
      match unbalanced_kl_from_log_reference(row_sums[i], @math.ln(source[i])) {
        Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
        Ok(value) => source_marginal_kl = source_marginal_kl + value
      }
      if !is_finite(source_marginal_kl) {
        return Err(UnbalancedSinkhornError::NumericalFailure)
      }
    }
  }
  for j in 0..<columns {
    let difference = abs(column_sums[j] - target[j])
    if !is_finite(difference) {
      return Err(UnbalancedSinkhornError::NumericalFailure)
    }
    if difference > target_marginal_residual {
      target_marginal_residual = difference
    }
    if target[j] > 0.0 {
      match
        unbalanced_kl_from_log_reference(column_sums[j], @math.ln(target[j])) {
        Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
        Ok(value) => target_marginal_kl = target_marginal_kl + value
      }
      if !is_finite(target_marginal_kl) {
        return Err(UnbalancedSinkhornError::NumericalFailure)
      }
    }
  }

  let source_penalty = match
    unbalanced_product2(options.rho_source, source_marginal_kl) {
    Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
    Ok(value) => value
  }
  let target_penalty = match
    unbalanced_product2(options.rho_target, target_marginal_kl) {
    Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
    Ok(value) => value
  }
  let entropy_penalty = match
    unbalanced_product2(options.epsilon, reference_kl) {
    Err(_) => return Err(UnbalancedSinkhornError::NumericalFailure)
    Ok(value) => value
  }
  let objective = transport_cost +
    entropy_penalty +
    source_penalty +
    target_penalty
  if !is_finite(objective) {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  Ok({
    objective,
    transport_cost,
    reference_kl,
    source_marginal_kl,
    target_marginal_kl,
    source_marginal_residual,
    target_marginal_residual,
  })
}

///|
fn unbalanced_relaxation(rho : Double, epsilon : Double) -> Double {
  if rho == 0.0 {
    0.0
  } else if rho >= epsilon {
    1.0 / (1.0 + epsilon / rho)
  } else {
    let ratio = rho / epsilon
    ratio / (1.0 + ratio)
  }
}

///|
fn unbalanced_scaled_cost(
  cost : Double,
  weight : Double,
  epsilon : Double,
) -> Result[Double, UnbalancedSinkhornError] {
  if cost == 0.0 || weight == 0.0 {
    return Ok(0.0)
  }
  let quotient = cost / epsilon
  if is_finite(quotient) && quotient > 0.0 {
    let direct = quotient * weight
    if is_finite(direct) && direct > 0.0 {
      return Ok(direct)
    }
  }
  let log_value = @math.ln(cost) + @math.ln(weight) - @math.ln(epsilon)
  if !is_finite(log_value) {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  if log_value > 709.782712893384 {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  if log_value < -744.4400719213812 {
    return Ok(0.0)
  }
  let value = @math.exp(log_value)
  if !is_finite(value) {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  Ok(value)
}

///|
fn unbalanced_kl_from_log_reference(
  mass : Double,
  log_reference : Double,
) -> Result[Double, UnbalancedSinkhornError] {
  if mass == 0.0 {
    if log_reference > 709.782712893384 {
      return Err(UnbalancedSinkhornError::NumericalFailure)
    }
    if log_reference < -744.4400719213812 {
      return Ok(0.0)
    }
    return Ok(@math.exp(log_reference))
  }
  let reference = if log_reference < -744.4400719213812 {
    0.0
  } else if log_reference > 709.782712893384 {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  } else {
    @math.exp(log_reference)
  }
  let delta = @math.ln(mass) - log_reference
  if !is_finite(delta) {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  let divergence = if abs(delta) < 0.01 {
    let delta2 = delta * delta
    let h = delta2 *
      (
        0.5 +
        delta *
        (
          1.0 / 3.0 +
          delta *
          (
            1.0 / 8.0 +
            delta * (1.0 / 30.0 + delta * (1.0 / 144.0 + delta / 840.0))
          )
        )
      )
    reference * h
  } else {
    mass * delta - mass + reference
  }
  if !is_finite(divergence) {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  Ok(if divergence < 0.0 { 0.0 } else { divergence })
}

///|
fn unbalanced_product2(
  left : Double,
  right : Double,
) -> Result[Double, UnbalancedSinkhornError] {
  unbalanced_product3(left, right, 1.0)
}

///|
fn unbalanced_product3(
  first : Double,
  second : Double,
  third : Double,
) -> Result[Double, UnbalancedSinkhornError] {
  if first == 0.0 || second == 0.0 || third == 0.0 {
    return Ok(0.0)
  }
  let direct = first * second * third
  if is_finite(direct) && direct > 0.0 {
    return Ok(direct)
  }
  let log_value = @math.ln(first) + @math.ln(second) + @math.ln(third)
  if !is_finite(log_value) {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  if log_value > 709.782712893384 {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  if log_value < -744.4400719213812 {
    return Ok(0.0)
  }
  let value = @math.exp(log_value)
  if !is_finite(value) {
    return Err(UnbalancedSinkhornError::NumericalFailure)
  }
  Ok(value)
}

///|
fn unbalanced_zero_plan(rows : Int, columns : Int) -> Array[Array[Double]] {
  let plan : Array[Array[Double]] = []
  for _ in 0..<rows {
    plan.push(Array::make(columns, 0.0))
  }
  plan
}
