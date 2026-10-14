///|
/// Shape and numeric errors from point-cloud cost-matrix construction.
pub(all) enum PointCostError {
  EmptySourceSupport
  EmptyTargetSupport
  EmptySourcePoint(Int)
  EmptyTargetPoint(Int)
  SourcePointDimensionMismatch(Int, Int, Int)
  TargetPointDimensionMismatch(Int, Int, Int)
  PointDimensionMismatch(Int, Int)
  NonFiniteSourceCoordinate(Int, Int)
  NonFiniteTargetCoordinate(Int, Int)
  NumericalOverflow(Int, Int)
}

///|
priv enum PointCostMetric {
  SquaredEuclidean
  Euclidean
  Manhattan
}

///|
/// Build squared-Euclidean costs between two finite point supports.
///
/// Each row is one point. The returned matrix has source points on rows and
/// target points on columns. Both supports and every point must be non-empty;
/// all points must have the same dimension and finite coordinates. A cost
/// whose subtraction, square, or sum is not representable returns
/// `NumericalOverflow(source_index, target_index)` rather than an infinite
/// entry.
pub fn squared_euclidean_cost_matrix(
  source_points : Array[Array[Double]],
  target_points : Array[Array[Double]],
) -> Result[Array[Array[Double]], PointCostError] {
  point_cost_matrix(
    PointCostMetric::SquaredEuclidean,
    source_points,
    target_points,
  )
}

///|
/// Build Euclidean-distance costs between two finite point supports.
///
/// Validation and orientation follow `squared_euclidean_cost_matrix`; each
/// entry is the Euclidean norm of the coordinate-difference vector. Repeated
/// hypot accumulation avoids squaring large finite coordinates unnecessarily.
pub fn euclidean_cost_matrix(
  source_points : Array[Array[Double]],
  target_points : Array[Array[Double]],
) -> Result[Array[Array[Double]], PointCostError] {
  point_cost_matrix(PointCostMetric::Euclidean, source_points, target_points)
}

///|
/// Build Manhattan-distance costs between two finite point supports.
///
/// Validation and orientation follow `squared_euclidean_cost_matrix`; each
/// entry is the sum of absolute coordinate differences.
pub fn manhattan_cost_matrix(
  source_points : Array[Array[Double]],
  target_points : Array[Array[Double]],
) -> Result[Array[Array[Double]], PointCostError] {
  point_cost_matrix(PointCostMetric::Manhattan, source_points, target_points)
}

///|
fn point_cost_matrix(
  metric : PointCostMetric,
  source_points : Array[Array[Double]],
  target_points : Array[Array[Double]],
) -> Result[Array[Array[Double]], PointCostError] {
  let source_dimension = match
    validate_point_supports(source_points, target_points) {
    Err(error) => return Err(error)
    Ok(dimension) => dimension
  }

  let costs : Array[Array[Double]] = []
  for i in 0..<source_points.length() {
    let row = Array::make(target_points.length(), 0.0)
    for j in 0..<target_points.length() {
      let mut value = 0.0
      for d in 0..<source_dimension {
        let delta = abs(source_points[i][d] - target_points[j][d])
        if !is_finite(delta) {
          return Err(PointCostError::NumericalOverflow(i, j))
        }
        match metric {
          PointCostMetric::Manhattan => value = value + delta
          PointCostMetric::Euclidean => value = @math.hypot(value, delta)
          PointCostMetric::SquaredEuclidean => value = value + delta * delta
        }
        if !is_finite(value) {
          return Err(PointCostError::NumericalOverflow(i, j))
        }
      }
      row[j] = value
    }
    costs.push(row)
  }
  Ok(costs)
}

///|
fn validate_point_supports(
  source_points : Array[Array[Double]],
  target_points : Array[Array[Double]],
) -> Result[Int, PointCostError] {
  if source_points.length() == 0 {
    return Err(PointCostError::EmptySourceSupport)
  }
  if target_points.length() == 0 {
    return Err(PointCostError::EmptyTargetSupport)
  }

  let source_dimension = source_points[0].length()
  if source_dimension == 0 {
    return Err(PointCostError::EmptySourcePoint(0))
  }
  for i in 0..<source_points.length() {
    if source_points[i].length() != source_dimension {
      return Err(
        PointCostError::SourcePointDimensionMismatch(
          i,
          source_dimension,
          source_points[i].length(),
        ),
      )
    }
    for d in 0..<source_dimension {
      if !is_finite(source_points[i][d]) {
        return Err(PointCostError::NonFiniteSourceCoordinate(i, d))
      }
    }
  }

  let target_dimension = target_points[0].length()
  if target_dimension == 0 {
    return Err(PointCostError::EmptyTargetPoint(0))
  }
  for j in 0..<target_points.length() {
    if target_points[j].length() != target_dimension {
      return Err(
        PointCostError::TargetPointDimensionMismatch(
          j,
          target_dimension,
          target_points[j].length(),
        ),
      )
    }
    for d in 0..<target_dimension {
      if !is_finite(target_points[j][d]) {
        return Err(PointCostError::NonFiniteTargetCoordinate(j, d))
      }
    }
  }
  if source_dimension != target_dimension {
    return Err(
      PointCostError::PointDimensionMismatch(source_dimension, target_dimension),
    )
  }

  Ok(source_dimension)
}
