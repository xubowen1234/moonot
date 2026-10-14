///|
/// Validation and arithmetic errors for Markov gluing of two dense plans.
/// Indices are zero-based; bridge dimensions are counts.
pub(all) enum CouplingCompositionError {
  InvalidTolerance
  InvalidLeft(CouplingError)
  InvalidRight(CouplingError)
  BridgeDimensionMismatch(Int, Int)
  BridgeMassMismatch(Int, Double, Double)
  UndefinedBridgeConditional(Int)
  NumericalOverflow(Int, Int, Int)
  InvalidOutput(CouplingError)
}

///|
/// Compose left[source, bridge] and right[bridge, target] by Markov gluing.
///
/// Return sum_j left[i][j] * (right[j][k] / right_row_mass[j]). Both plans
/// must be finite nonnegative non-empty rectangular matrices with finite
/// marginals and matching bridge dimensions. Each bridge-mass difference
/// must be at most the finite strictly positive absolute tolerance. Positive
/// left bridge mass with a zero right row is always rejected; shared zero
/// bridges are skipped. Inputs are never mutated or globally normalized.
///
/// In exact arithmetic, source marginals are preserved; target marginals are
/// preserved for exact bridge matches. For near matches the target L1 error
/// is at most the sum of absolute bridge discrepancies (hence bridge_count
/// times tolerance), before floating-point rounding. No optimality is implied.
pub fn coupling_compose(
  left : Array[Array[Double]],
  right : Array[Array[Double]],
  tolerance : Double,
) -> Result[Array[Array[Double]], CouplingCompositionError] {
  if !is_finite(tolerance) || tolerance <= 0.0 {
    return Err(CouplingCompositionError::InvalidTolerance)
  }
  let bridges = match validate_coupling(left) {
    Err(error) => return Err(CouplingCompositionError::InvalidLeft(error))
    Ok(value) => value
  }
  let targets = match validate_coupling(right) {
    Err(error) => return Err(CouplingCompositionError::InvalidRight(error))
    Ok(value) => value
  }
  if bridges != right.length() {
    return Err(
      CouplingCompositionError::BridgeDimensionMismatch(bridges, right.length()),
    )
  }
  let left_marginals = match coupling_marginals(left) {
    Err(error) => return Err(CouplingCompositionError::InvalidLeft(error))
    Ok(value) => value
  }
  let right_marginals = match coupling_marginals(right) {
    Err(error) => return Err(CouplingCompositionError::InvalidRight(error))
    Ok(value) => value
  }
  for j in 0..<bridges {
    let left_mass = left_marginals.target[j]
    let right_mass = right_marginals.source[j]
    if left_mass > 0.0 && right_mass == 0.0 {
      return Err(CouplingCompositionError::UndefinedBridgeConditional(j))
    }
    if abs(left_mass - right_mass) > tolerance {
      return Err(
        CouplingCompositionError::BridgeMassMismatch(j, left_mass, right_mass),
      )
    }
  }
  let composed : Array[Array[Double]] = []
  for i in 0..<left.length() {
    let row = Array::make(targets, 0.0)
    for k in 0..<targets {
      let mut total = 0.0
      let mut correction = 0.0
      for j in 0..<bridges {
        // This also skips shared zero bridges without dividing by zero.
        if left[i][j] == 0.0 || right[j][k] == 0.0 {
          continue
        }
        // Every positive left mass has a positive right row after validation.
        // Divide first: the conditional is in [0,1], whereas left*right
        // could overflow even for a perfectly representable composition.
        let conditional = right[j][k] / right_marginals.source[j]
        let term = left[i][j] * conditional
        let adjusted = term - correction
        let next = total + adjusted
        let remainder = next - total - adjusted
        if !is_finite(conditional) ||
          !is_finite(term) ||
          !is_finite(adjusted) ||
          !is_finite(next) ||
          !is_finite(remainder) {
          return Err(CouplingCompositionError::NumericalOverflow(i, j, k))
        }
        total = next
        correction = remainder
      }
      row[k] = total
    }
    composed.push(row)
  }
  // Finite individual entries alone do not ensure finite output marginals,
  // especially when tolerance permits reweighting the right bridge masses.
  match coupling_marginals(composed) {
    Err(error) => Err(CouplingCompositionError::InvalidOutput(error))
    Ok(_) => Ok(composed)
  }
}
