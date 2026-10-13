///|
/// Status of a completed balanced Sinkhorn run.
pub(all) enum SinkhornTermination {
  Converged
  IterationLimit
}

///|
/// Parameters for the balanced log-domain Sinkhorn solver.
/// `tolerance` is an absolute maximum marginal residual.
pub(all) struct SinkhornOptions {
  epsilon : Double
  tolerance : Double
  max_iterations : Int
}

///|
/// Opaque dual-potential seed returned by a balanced solve.
///
/// It can initialize another solve with compatible support dimensions. Values
/// are expressed in cost units, so they can be reused with a different epsilon.
pub struct SinkhornWarmStart {
  source_potential : Array[Double]
  target_potential : Array[Double]
}

///|
/// A transport plan and diagnostics for the regularized problem
/// `<P, C> + epsilon * sum(P * (log(P) - 1))`.
pub(all) struct SinkhornResult {
  coupling : Array[Array[Double]]
  transport_cost : Double
  entropy_regularizer : Double
  regularized_objective : Double
  marginal_residual : Double
  iterations : Int
  termination : SinkhornTermination
}

///|
/// Input and numerical errors reported by `balanced_sinkhorn`.
pub(all) enum SinkhornError {
  EmptySourceMarginal
  EmptyTargetMarginal
  CostRowCountMismatch
  CostColumnCountMismatch(Int)
  NegativeSourceMass(Int)
  NegativeTargetMass(Int)
  NonFiniteSourceMass(Int)
  NonFiniteTargetMass(Int)
  NonFiniteTotalMass
  UnequalTotalMass
  NegativeCost(Int, Int)
  NonFiniteCost(Int, Int)
  InvalidEpsilon
  InvalidTolerance
  InvalidIterationLimit
  InvalidWarmStart
  NumericalFailure
}

///|
/// Solve balanced entropy-regularized optimal transport in the log domain.
///
/// Marginal totals must agree within `1.0e-12` relative error. Costs must be
/// finite and nonnegative; zero marginal entries are allowed and produce zero
/// rows or columns in the coupling. The reported residual is the maximum
/// absolute row or column marginal error against the supplied marginals.
pub fn balanced_sinkhorn(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : SinkhornOptions,
) -> Result[SinkhornResult, SinkhornError] {
  match balanced_sinkhorn_internal(source, target, cost, options, None, false) {
    Err(error) => Err(error)
    Ok((result, _)) => Ok(result)
  }
}

///|
/// Solve balanced transport using a prior solve's opaque dual-potential seed.
///
/// Returns both the ordinary diagnostics and a seed for another solve. A warm
/// start only changes the initial iterate; the requested tolerance, iteration
/// limit, and termination checks still apply to the new problem.
pub fn balanced_sinkhorn_with_warm_start(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : SinkhornOptions,
  warm_start : SinkhornWarmStart?,
) -> Result[(SinkhornResult, SinkhornWarmStart), SinkhornError] {
  match
    balanced_sinkhorn_internal(source, target, cost, options, warm_start, true) {
    Err(error) => Err(error)
    Ok((result, Some(seed))) => Ok((result, seed))
    Ok(_) => Err(SinkhornError::NumericalFailure)
  }
}

///|
fn balanced_sinkhorn_internal(
  source : Array[Double],
  target : Array[Double],
  cost : Array[Array[Double]],
  options : SinkhornOptions,
  warm_start : SinkhornWarmStart?,
  capture_warm_start : Bool,
) -> Result[(SinkhornResult, SinkhornWarmStart?), SinkhornError] {
  let rows = source.length()
  let columns = target.length()
  if rows == 0 {
    return Err(SinkhornError::EmptySourceMarginal)
  }
  if columns == 0 {
    return Err(SinkhornError::EmptyTargetMarginal)
  }
  if cost.length() != rows {
    return Err(SinkhornError::CostRowCountMismatch)
  }
  if !is_finite(options.epsilon) || options.epsilon <= 0.0 {
    return Err(SinkhornError::InvalidEpsilon)
  }
  if !is_finite(options.tolerance) || options.tolerance <= 0.0 {
    return Err(SinkhornError::InvalidTolerance)
  }
  if options.max_iterations <= 0 {
    return Err(SinkhornError::InvalidIterationLimit)
  }

  let mut source_total = 0.0
  for i in 0..<rows {
    let mass = source[i]
    if !is_finite(mass) {
      return Err(SinkhornError::NonFiniteSourceMass(i))
    }
    if mass < 0.0 {
      return Err(SinkhornError::NegativeSourceMass(i))
    }
    source_total = source_total + mass
    if !is_finite(source_total) {
      return Err(SinkhornError::NonFiniteTotalMass)
    }
  }

  let mut target_total = 0.0
  for j in 0..<columns {
    let mass = target[j]
    if !is_finite(mass) {
      return Err(SinkhornError::NonFiniteTargetMass(j))
    }
    if mass < 0.0 {
      return Err(SinkhornError::NegativeTargetMass(j))
    }
    target_total = target_total + mass
    if !is_finite(target_total) {
      return Err(SinkhornError::NonFiniteTotalMass)
    }
  }
  if source_total > 0.0 || target_total > 0.0 {
    let scale = if source_total > target_total {
      source_total
    } else {
      target_total
    }
    if abs(source_total - target_total) > 1.0e-12 * scale {
      return Err(SinkhornError::UnequalTotalMass)
    }
  }

  for i in 0..<rows {
    if cost[i].length() != columns {
      return Err(SinkhornError::CostColumnCountMismatch(i))
    }
    for j in 0..<columns {
      let value = cost[i][j]
      if !is_finite(value) {
        return Err(SinkhornError::NonFiniteCost(i, j))
      }
      if value < 0.0 {
        return Err(SinkhornError::NegativeCost(i, j))
      }
    }
  }

  match warm_start {
    None => ()
    Some(seed) => {
      if seed.source_potential.length() != rows ||
        seed.target_potential.length() != columns {
        return Err(SinkhornError::InvalidWarmStart)
      }
      for i in 0..<rows {
        if !is_finite(seed.source_potential[i]) {
          return Err(SinkhornError::InvalidWarmStart)
        }
      }
      for j in 0..<columns {
        if !is_finite(seed.target_potential[j]) {
          return Err(SinkhornError::InvalidWarmStart)
        }
      }
    }
  }

  let zero_plan : Array[Array[Double]] = []
  for _ in 0..<rows {
    zero_plan.push(Array::make(columns, 0.0))
  }
  if source_total == 0.0 && target_total == 0.0 {
    let result : SinkhornResult = {
      coupling: zero_plan,
      transport_cost: 0.0,
      entropy_regularizer: 0.0,
      regularized_objective: 0.0,
      marginal_residual: 0.0,
      iterations: 0,
      termination: SinkhornTermination::Converged,
    }
    let seed : SinkhornWarmStart? = if capture_warm_start {
      Some({
        source_potential: Array::make(rows, 0.0),
        target_potential: Array::make(columns, 0.0),
      })
    } else {
      None
    }
    return Ok((result, seed))
  }

  let mut minimum_cost = 0.0
  let mut has_active_cost = false
  for i in 0..<rows {
    if source[i] > 0.0 {
      for j in 0..<columns {
        if target[j] > 0.0 {
          if !has_active_cost || cost[i][j] < minimum_cost {
            minimum_cost = cost[i][j]
            has_active_cost = true
          }
        }
      }
    }
  }
  if !has_active_cost {
    return Err(SinkhornError::NumericalFailure)
  }

  let log_kernel : Array[Array[Double]] = []
  for i in 0..<rows {
    let row = Array::make(columns, 0.0)
    for j in 0..<columns {
      if source[i] > 0.0 && target[j] > 0.0 {
        let scaled_cost = (cost[i][j] - minimum_cost) / options.epsilon
        if !is_finite(scaled_cost) {
          return Err(SinkhornError::NumericalFailure)
        }
        row[j] = -scaled_cost
      }
    }
    log_kernel.push(row)
  }

  let log_source = Array::make(rows, 0.0)
  for i in 0..<rows {
    if source[i] > 0.0 {
      log_source[i] = @math.ln(source[i])
    }
  }
  let log_target = Array::make(columns, 0.0)
  for j in 0..<columns {
    if target[j] > 0.0 {
      log_target[j] = @math.ln(target[j])
    }
  }
  let log_u = Array::make(rows, 0.0)
  let log_v = Array::make(columns, 0.0)
  match warm_start {
    None => ()
    Some(seed) => {
      for i in 0..<rows {
        if source[i] > 0.0 {
          log_u[i] = (seed.source_potential[i] - minimum_cost) / options.epsilon
          if !is_finite(log_u[i]) {
            return Err(SinkhornError::NumericalFailure)
          }
        }
      }
      for j in 0..<columns {
        if target[j] > 0.0 {
          log_v[j] = seed.target_potential[j] / options.epsilon
          if !is_finite(log_v[j]) {
            return Err(SinkhornError::NumericalFailure)
          }
        }
      }
    }
  }
  let mut coupling = zero_plan
  let mut residual = 0.0
  let mut iterations = 0
  let mut termination = SinkhornTermination::IterationLimit

  for iteration in 1..<=options.max_iterations {
    for i in 0..<rows {
      if source[i] > 0.0 {
        let terms : Array[Double] = []
        for j in 0..<columns {
          if target[j] > 0.0 {
            terms.push(log_v[j] + log_kernel[i][j])
          }
        }
        let normalizer = logsumexp(terms)
        log_u[i] = log_source[i] - normalizer
        if !is_finite(log_u[i]) {
          return Err(SinkhornError::NumericalFailure)
        }
      }
    }

    for j in 0..<columns {
      if target[j] > 0.0 {
        let terms : Array[Double] = []
        for i in 0..<rows {
          if source[i] > 0.0 {
            terms.push(log_u[i] + log_kernel[i][j])
          }
        }
        let normalizer = logsumexp(terms)
        log_v[j] = log_target[j] - normalizer
        if !is_finite(log_v[j]) {
          return Err(SinkhornError::NumericalFailure)
        }
      }
    }

    coupling = Array::make(rows, Array::make(columns, 0.0))
    // Replace each row explicitly so rows do not alias the same inner array.
    let next_coupling : Array[Array[Double]] = []
    for i in 0..<rows {
      let row = Array::make(columns, 0.0)
      for j in 0..<columns {
        if source[i] > 0.0 && target[j] > 0.0 {
          let log_mass = log_u[i] + log_kernel[i][j] + log_v[j]
          if !is_finite(log_mass) {
            return Err(SinkhornError::NumericalFailure)
          }
          row[j] = @math.exp(log_mass)
          if !is_finite(row[j]) {
            return Err(SinkhornError::NumericalFailure)
          }
        }
      }
      next_coupling.push(row)
    }
    coupling = next_coupling
    residual = marginal_residual(coupling, source, target)
    iterations = iteration
    if !is_finite(residual) {
      return Err(SinkhornError::NumericalFailure)
    }
    if residual <= options.tolerance {
      termination = SinkhornTermination::Converged
      break
    }
  }

  let (transport_cost, entropy_regularizer) = balanced_objective_components(
    coupling,
    cost,
    options.epsilon,
  )
  let objective = transport_cost + entropy_regularizer
  if !is_finite(transport_cost) ||
    !is_finite(entropy_regularizer) ||
    !is_finite(objective) {
    return Err(SinkhornError::NumericalFailure)
  }
  let seed : SinkhornWarmStart? = if capture_warm_start {
    let source_potential = Array::make(rows, 0.0)
    let target_potential = Array::make(columns, 0.0)
    for i in 0..<rows {
      if source[i] > 0.0 {
        source_potential[i] = options.epsilon * log_u[i] + minimum_cost
        if !is_finite(source_potential[i]) {
          return Err(SinkhornError::NumericalFailure)
        }
      }
    }
    for j in 0..<columns {
      if target[j] > 0.0 {
        target_potential[j] = options.epsilon * log_v[j]
        if !is_finite(target_potential[j]) {
          return Err(SinkhornError::NumericalFailure)
        }
      }
    }
    Some({ source_potential, target_potential, })
  } else {
    None
  }
  let result : SinkhornResult = {
    coupling,
    transport_cost,
    entropy_regularizer,
    regularized_objective: objective,
    marginal_residual: residual,
    iterations,
    termination,
  }
  Ok((result, seed))
}

///|
fn is_finite(value : Double) -> Bool {
  !value.is_nan() && !value.is_inf()
}

///|
fn abs(value : Double) -> Double {
  if value < 0.0 {
    -value
  } else {
    value
  }
}

///|
fn logsumexp(values : Array[Double]) -> Double {
  let mut maximum = values[0]
  for i in 1..<values.length() {
    if values[i] > maximum {
      maximum = values[i]
    }
  }
  let mut sum = 0.0
  for i in 0..<values.length() {
    sum = sum + @math.exp(values[i] - maximum)
  }
  maximum + @math.ln(sum)
}

///|
fn marginal_residual(
  coupling : Array[Array[Double]],
  source : Array[Double],
  target : Array[Double],
) -> Double {
  let rows = source.length()
  let columns = target.length()
  let mut maximum = 0.0
  for i in 0..<rows {
    let mut sum = 0.0
    for j in 0..<columns {
      sum = sum + coupling[i][j]
    }
    let error = abs(sum - source[i])
    if error > maximum {
      maximum = error
    }
  }
  for j in 0..<columns {
    let mut sum = 0.0
    for i in 0..<rows {
      sum = sum + coupling[i][j]
    }
    let error = abs(sum - target[j])
    if error > maximum {
      maximum = error
    }
  }
  maximum
}

///|
fn balanced_objective_components(
  coupling : Array[Array[Double]],
  cost : Array[Array[Double]],
  epsilon : Double,
) -> (Double, Double) {
  let mut transport_cost = 0.0
  let mut entropy_regularizer = 0.0
  for i in 0..<coupling.length() {
    for j in 0..<coupling[i].length() {
      let mass = coupling[i][j]
      if mass > 0.0 {
        transport_cost = transport_cost + mass * cost[i][j]
        entropy_regularizer = entropy_regularizer +
          epsilon * mass * (@math.ln(mass) - 1.0)
      }
    }
  }
  (transport_cost, entropy_regularizer)
}
