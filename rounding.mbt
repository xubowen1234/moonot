///|
/// A nonnegative rounded plan and its actual maximum absolute marginal error.
/// This is a feasibility diagnostic, not an optimality or convergence claim.
pub(all) struct CouplingRoundingResult {
  coupling : Array[Array[Double]]
  marginal_residual : Double
}

///|
/// Input, arithmetic, and final-feasibility failures from coupling rounding.
/// Indices are zero-based. FeasibilityFailure contains the measured residual.
pub(all) enum CouplingRoundingError {
  InvalidCoupling(CouplingError)
  SourceCountMismatch
  TargetCountMismatch
  NonFiniteSourceMass(Int)
  NonFiniteTargetMass(Int)
  NegativeSourceMass(Int)
  NegativeTargetMass(Int)
  NonFiniteSourceTotal
  NonFiniteTargetTotal
  UnequalTotalMass
  InvalidTolerance
  NumericalFailure
  FeasibilityFailure(Double)
}

///|
/// Round a dense nonnegative coupling toward supplied balanced marginals.
///
/// Clip rows by min(source[i] / row_sum[i], 1), then columns by
/// min(target[j] / column_sum[j], 1), leaving zero sums unchanged. Add a
/// nonnegative rank-one repair from the resulting positive marginal deficits.
/// Its denominator is the larger deficit total; for exactly balanced arithmetic
/// both totals agree. No caller inputs are changed or normalized.
///
/// Marginal totals must be finite and agree within 1.0e-12 relative to their
/// maximum (both zero is valid). The finite positive tolerance bounds the final
/// maximum absolute marginal residual against the original marginals. A larger
/// residual is an explicit FeasibilityFailure, even for nearly equal totals.
/// Both-zero marginals return an all-zero plan after input validation.
pub fn round_balanced_coupling(
  coupling : Array[Array[Double]],
  source : Array[Double],
  target : Array[Double],
  tolerance : Double,
) -> Result[CouplingRoundingResult, CouplingRoundingError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(CouplingRoundingError::InvalidCoupling(error))
    Ok(value) => value
  }
  let rows = coupling.length()
  if source.length() != rows {
    return Err(CouplingRoundingError::SourceCountMismatch)
  }
  if target.length() != columns {
    return Err(CouplingRoundingError::TargetCountMismatch)
  }
  if !is_finite(tolerance) || tolerance <= 0.0 {
    return Err(CouplingRoundingError::InvalidTolerance)
  }
  let mut source_total = 0.0
  for i in 0..<rows {
    if !is_finite(source[i]) {
      return Err(CouplingRoundingError::NonFiniteSourceMass(i))
    }
    if source[i] < 0.0 {
      return Err(CouplingRoundingError::NegativeSourceMass(i))
    }
    source_total = source_total + source[i]
    if !is_finite(source_total) {
      return Err(CouplingRoundingError::NonFiniteSourceTotal)
    }
  }
  let mut target_total = 0.0
  for j in 0..<columns {
    if !is_finite(target[j]) {
      return Err(CouplingRoundingError::NonFiniteTargetMass(j))
    }
    if target[j] < 0.0 {
      return Err(CouplingRoundingError::NegativeTargetMass(j))
    }
    target_total = target_total + target[j]
    if !is_finite(target_total) {
      return Err(CouplingRoundingError::NonFiniteTargetTotal)
    }
  }
  let total_scale = if source_total > target_total {
    source_total
  } else {
    target_total
  }
  if total_scale > 0.0 &&
    abs(source_total - target_total) / total_scale > 1.0e-12 {
    return Err(CouplingRoundingError::UnequalTotalMass)
  }
  let rounded : Array[Array[Double]] = []
  for i in 0..<rows {
    let row = Array::make(columns, 0.0)
    if total_scale > 0.0 {
      for j in 0..<columns {
        row[j] = coupling[i][j]
      }
      rounding_clip_vector(row, source[i])
    }
    rounded.push(row)
  }
  if total_scale == 0.0 {
    return Ok({ coupling: rounded, marginal_residual: 0.0, })
  }
  for j in 0..<columns {
    let column = Array::make(rows, 0.0)
    for i in 0..<rows {
      column[i] = rounded[i][j]
    }
    rounding_clip_vector(column, target[j])
    for i in 0..<rows {
      rounded[i][j] = column[i]
    }
  }
  let clipped = match coupling_marginals(rounded) {
    Err(_) => return Err(CouplingRoundingError::NumericalFailure)
    Ok(value) => value
  }
  let source_deficit = Array::make(rows, 0.0)
  let target_deficit = Array::make(columns, 0.0)
  let mut source_deficit_total = 0.0
  let mut target_deficit_total = 0.0
  for i in 0..<rows {
    if source[i] > clipped.source[i] {
      source_deficit[i] = source[i] - clipped.source[i]
    }
    source_deficit_total = source_deficit_total + source_deficit[i]
  }
  for j in 0..<columns {
    if target[j] > clipped.target[j] {
      target_deficit[j] = target[j] - clipped.target[j]
    }
    target_deficit_total = target_deficit_total + target_deficit[j]
  }
  if !is_finite(source_deficit_total) || !is_finite(target_deficit_total) {
    return Err(CouplingRoundingError::NumericalFailure)
  }
  let deficit_total = if source_deficit_total > target_deficit_total {
    source_deficit_total
  } else {
    target_deficit_total
  }
  if deficit_total > 0.0 {
    for i in 0..<rows {
      for j in 0..<columns {
        // Normalize the larger deficit first. Never form the potentially
        // overflowing/underflowing product source_deficit * target_deficit.
        let repair = if source_deficit[i] <= target_deficit[j] {
          source_deficit[i] * (target_deficit[j] / deficit_total)
        } else {
          target_deficit[j] * (source_deficit[i] / deficit_total)
        }
        rounded[i][j] = rounded[i][j] + repair
        if !is_finite(rounded[i][j]) {
          return Err(CouplingRoundingError::NumericalFailure)
        }
      }
    }
  }
  // Checked sums also detect any overflow in the actual returned plan.
  let actual = match coupling_marginals(rounded) {
    Err(_) => return Err(CouplingRoundingError::NumericalFailure)
    Ok(value) => value
  }
  let mut residual = 0.0
  for i in 0..<rows {
    let error = abs(actual.source[i] - source[i])
    if error > residual {
      residual = error
    }
  }
  for j in 0..<columns {
    let error = abs(actual.target[j] - target[j])
    if error > residual {
      residual = error
    }
  }
  if !is_finite(residual) {
    return Err(CouplingRoundingError::NumericalFailure)
  }
  if residual > tolerance {
    return Err(CouplingRoundingError::FeasibilityFailure(residual))
  }
  Ok({ coupling: rounded, marginal_residual: residual, })
}

///|
/// Mathematically min(capacity / sum(values), 1) times each entry.
/// Max-scaled summation avoids overflow even when the original sum is infinite.
/// When clipping, normalized shares times capacity avoid an underflowed factor.
fn rounding_clip_vector(values : Array[Double], capacity : Double) -> Unit {
  let mut maximum = 0.0
  for value in values {
    if value > maximum {
      maximum = value
    }
  }
  if maximum == 0.0 {
    return
  }
  let mut scaled_sum = 0.0
  for value in values {
    scaled_sum = scaled_sum + value / maximum
  }
  // Compare without constructing maximum * scaled_sum or capacity / maximum.
  if maximum > capacity / scaled_sum {
    let factor = capacity / maximum / scaled_sum
    for index in 0..<values.length() {
      // A representable factor preserves tiny entries in a large-mass row.
      // If the factor vanishes, normalize first to retain representable mass.
      values[index] = if factor > 0.0 {
        values[index] * factor
      } else {
        values[index] / maximum / scaled_sum * capacity
      }
    }
  }
}
