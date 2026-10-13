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
      let mut scale = 0.0
      for j in 0..<columns {
        if coupling[i][j] > 0.0 {
          let magnitude = abs(target_values[j][d])
          if magnitude > scale {
            scale = magnitude
          }
        }
      }
      if scale > 0.0 {
        let mut normalized_mean = 0.0
        for j in 0..<columns {
          let weight = coupling[i][j] / row_mass
          let normalized_value = target_values[j][d] / scale
          normalized_mean = normalized_mean + weight * normalized_value
          if !is_finite(normalized_mean) {
            return Err(CouplingProjectionError::NumericalOverflow(i, d))
          }
        }
        // A normalized convex combination is in [-1, 1]. Clamp only the
        // rounding error at that known mathematical boundary.
        let bounded_mean = if normalized_mean > 1.0 {
          1.0
        } else if normalized_mean < -1.0 {
          -1.0
        } else {
          normalized_mean
        }
        projected[d] = bounded_mean * scale
        if !is_finite(projected[d]) {
          return Err(CouplingProjectionError::NumericalOverflow(i, d))
        }
      }
    }
    projection.push(projected)
  }
  Ok(projection)
}
