///|
/// Validation failures for conditional barycentric projections from a coupling.
pub(all) enum CouplingProjectionError {
  InvalidCoupling(CouplingError)
  TargetValueCountMismatch
  EmptyTargetFeatureVector
  RaggedTargetValuesRow(Int, Int, Int)
  NonFiniteTargetValue(Int, Int)
  ZeroMassSourceRow(Int)
  NumericalOverflow(Int, Int)
}

///|
/// Map each source row to the conditional weighted mean of target values.
///
/// `target_values[j]` is a finite, non-empty feature vector associated with
/// target column `j`. Every coupling row must have positive mass; a zero-mass
/// row has no conditional barycentric projection and is reported explicitly.
/// The operation preserves feature dimension but not source-row masses.
pub fn coupling_barycentric_projection(
  coupling : Array[Array[Double]],
  target_values : Array[Array[Double]],
) -> Result[Array[Array[Double]], CouplingProjectionError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(CouplingProjectionError::InvalidCoupling(error))
    Ok(value) => value
  }
  if target_values.length() != columns {
    return Err(CouplingProjectionError::TargetValueCountMismatch)
  }
  let dimensions = target_values[0].length()
  if dimensions == 0 {
    return Err(CouplingProjectionError::EmptyTargetFeatureVector)
  }
  for j in 0..<columns {
    if target_values[j].length() != dimensions {
      return Err(
        CouplingProjectionError::RaggedTargetValuesRow(
          j,
          dimensions,
          target_values[j].length(),
        ),
      )
    }
    for d in 0..<dimensions {
      if !is_finite(target_values[j][d]) {
        return Err(CouplingProjectionError::NonFiniteTargetValue(j, d))
      }
    }
  }

  let marginals = match coupling_marginals(coupling) {
    Err(error) => return Err(CouplingProjectionError::InvalidCoupling(error))
    Ok(value) => value
  }
  let projection : Array[Array[Double]] = []
  for i in 0..<coupling.length() {
    let row_mass = marginals.source[i]
    if row_mass == 0.0 {
      return Err(CouplingProjectionError::ZeroMassSourceRow(i))
    }
    let projected = Array::make(dimensions, 0.0)
    for d in 0..<dimensions {
      let significands = Array::make(columns, 0.0)
      let exponents = Array::make(columns, 0)
      let mut maximum_exponent = -4096
      let mut coordinate_bound = 0.0
      for j in 0..<columns {
        // Validate every coordinate above, but never perform arithmetic on
        // a zero-mass contributor. In particular, 0 * infinity is not zero.
        if coupling[i][j] > 0.0 && target_values[j][d] != 0.0 {
          let magnitude = abs(target_values[j][d])
          if magnitude > coordinate_bound {
            coordinate_bound = magnitude
          }
          let (significand, exponent) = product_quotient_parts(
            coupling[i][j],
            magnitude,
            row_mass,
          )
          significands[j] = if target_values[j][d] < 0.0 {
            -significand
          } else {
            significand
          }
          exponents[j] = exponent
          if exponent > maximum_exponent {
            maximum_exponent = exponent
          }
        }
      }
      if coordinate_bound > 0.0 {
        let mut total = 0.0
        let mut correction = 0.0
        for j in 0..<columns {
          if significands[j] != 0.0 {
            let term = @math.scalbn(
              significands[j],
              exponents[j] - maximum_exponent,
            )
            let adjusted = term - correction
            let next = total + adjusted
            correction = next - total - adjusted
            total = next
          }
        }
        // Restore the exponent only after summation: a tiny conditional
        // weight can still have a representable contribution at a large
        // coordinate. The exact convex combination is within this bound;
        // clamp only floating-point overshoot at that mathematical boundary.
        let mean = @math.scalbn(total, maximum_exponent)
        projected[d] = if mean > coordinate_bound {
          coordinate_bound
        } else if mean < -coordinate_bound {
          -coordinate_bound
        } else {
          mean
        }
        if !is_finite(projected[d]) {
          return Err(CouplingProjectionError::NumericalOverflow(i, d))
        }
      }
    }
    projection.push(projected)
  }
  Ok(projection)
}
