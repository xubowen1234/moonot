///|
/// Errors for diagonal feature-weighted squared Euclidean costs.
pub(all) enum WeightedPointCostError {
  InvalidPoints(PointCostError)
  WeightDimensionMismatch
  NonFiniteWeight(Int)
  NegativeWeight(Int)
  NumericalOverflow(Int, Int)
}

///|
/// Compute sum_d weights[d] * (source[i][d] - target[j][d])^2.
/// Zero weights intentionally ignore a coordinate (a semimetric), but every
/// input coordinate must still be finite. Inputs are never mutated.
pub fn weighted_squared_euclidean_cost_matrix(
  source_points : Array[Array[Double]],
  target_points : Array[Array[Double]],
  weights : Array[Double],
) -> Result[Array[Array[Double]], WeightedPointCostError] {
  let dimensions = match validate_point_supports(source_points, target_points) {
    Err(error) => return Err(WeightedPointCostError::InvalidPoints(error))
    Ok(value) => value
  }
  if weights.length() != dimensions {
    return Err(WeightedPointCostError::WeightDimensionMismatch)
  }
  let scales = Array::make(dimensions, 0.0)
  for d in 0..<dimensions {
    if !is_finite(weights[d]) {
      return Err(WeightedPointCostError::NonFiniteWeight(d))
    }
    if weights[d] < 0.0 {
      return Err(WeightedPointCostError::NegativeWeight(d))
    }
    scales[d] = weights[d].sqrt()
  }
  let costs : Array[Array[Double]] = []
  for i in 0..<source_points.length() {
    let row = Array::make(target_points.length(), 0.0)
    for j in 0..<target_points.length() {
      let mut norm = 0.0
      for d in 0..<dimensions {
        if scales[d] > 0.0 {
          let delta = source_points[i][d] - target_points[j][d]
          // Rescale before subtraction only if the unscaled difference
          // overflows; tiny weights can still make that distance finite.
          let scaled = if is_finite(delta) {
            scales[d] * delta
          } else {
            scales[d] * source_points[i][d] - scales[d] * target_points[j][d]
          }
          if !is_finite(scaled) {
            return Err(WeightedPointCostError::NumericalOverflow(i, j))
          }
          norm = @math.hypot(norm, scaled)
        }
      }
      let cost = norm * norm
      if !is_finite(cost) {
        return Err(WeightedPointCostError::NumericalOverflow(i, j))
      }
      row[j] = cost
    }
    costs.push(row)
  }
  Ok(costs)
}
