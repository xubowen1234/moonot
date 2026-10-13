///|
/// Shape and numeric errors from generic coupling-plan utilities.
pub(all) enum CouplingError {
  EmptyCoupling
  EmptyCouplingRow(Int)
  RaggedCouplingRow(Int, Int, Int)
  NonFiniteCouplingEntry(Int, Int)
  NegativeCouplingEntry(Int, Int)
  CostRowCountMismatch
  CostColumnCountMismatch(Int)
  NonFiniteCost(Int, Int)
  NegativeCost(Int, Int)
  NumericalOverflow(Int, Int)
}

///|
/// Row and column sums of a dense coupling matrix.
pub(all) struct CouplingMarginals {
  source : Array[Double]
  target : Array[Double]
}

///|
/// Validate a non-empty dense coupling and return its column count.
fn validate_coupling(
  coupling : Array[Array[Double]],
) -> Result[Int, CouplingError] {
  if coupling.length() == 0 {
    return Err(CouplingError::EmptyCoupling)
  }
  let columns = coupling[0].length()
  if columns == 0 {
    return Err(CouplingError::EmptyCouplingRow(0))
  }
  for i in 0..<coupling.length() {
    if coupling[i].length() != columns {
      return Err(
        CouplingError::RaggedCouplingRow(i, columns, coupling[i].length()),
      )
    }
    for j in 0..<columns {
      let value = coupling[i][j]
      if !is_finite(value) {
        return Err(CouplingError::NonFiniteCouplingEntry(i, j))
      }
      if value < 0.0 {
        return Err(CouplingError::NegativeCouplingEntry(i, j))
      }
    }
  }
  Ok(columns)
}

///|
/// Compute source-row and target-column mass totals of a coupling plan.
///
/// The input must be a non-empty, rectangular matrix of finite nonnegative
/// entries. If summation exceeds the finite `Double` range, a typed numerical
/// error is returned.
pub fn coupling_marginals(
  coupling : Array[Array[Double]],
) -> Result[CouplingMarginals, CouplingError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(error)
    Ok(value) => value
  }
  let source = Array::make(coupling.length(), 0.0)
  let target = Array::make(columns, 0.0)
  for i in 0..<coupling.length() {
    for j in 0..<columns {
      source[i] = source[i] + coupling[i][j]
      if !is_finite(source[i]) {
        return Err(CouplingError::NumericalOverflow(i, j))
      }
      target[j] = target[j] + coupling[i][j]
      if !is_finite(target[j]) {
        return Err(CouplingError::NumericalOverflow(i, j))
      }
    }
  }
  Ok({ source, target, })
}

///|
/// Compute the unregularized transport cost `<P, C>` of a dense coupling.
///
/// The plan and cost matrix must be non-empty rectangular matrices with the
/// same shape. Coupling entries and costs must be finite and nonnegative. This
/// diagnostic is distinct from the entropy-regularized objective returned by
/// the solver.
pub fn coupling_transport_cost(
  coupling : Array[Array[Double]],
  cost : Array[Array[Double]],
) -> Result[Double, CouplingError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(error)
    Ok(value) => value
  }
  if cost.length() != coupling.length() {
    return Err(CouplingError::CostRowCountMismatch)
  }
  let mut total = 0.0
  for i in 0..<coupling.length() {
    if cost[i].length() != columns {
      return Err(CouplingError::CostColumnCountMismatch(i))
    }
    for j in 0..<columns {
      let value = cost[i][j]
      if !is_finite(value) {
        return Err(CouplingError::NonFiniteCost(i, j))
      }
      if value < 0.0 {
        return Err(CouplingError::NegativeCost(i, j))
      }
      let term = coupling[i][j] * value
      if !is_finite(term) {
        return Err(CouplingError::NumericalOverflow(i, j))
      }
      total = total + term
      if !is_finite(total) {
        return Err(CouplingError::NumericalOverflow(i, j))
      }
    }
  }
  Ok(total)
}
