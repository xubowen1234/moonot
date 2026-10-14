///|
test "weighted ground costs preserve rectangular orientation and feature units" {
  let x = [[0.0, 0.0], [3.0, 4.0]]
  let y = [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]
  match @moonot.weighted_squared_euclidean_cost_matrix(x, y, [4.0, 0.25]) {
    Err(_) => assert_true(false)
    Ok(c) => {
      let expected = [[0.0, 5.0, 40.0], [40.0, 17.0, 0.0]]
      for i in 0..<2 {
        for j in 0..<3 {
          assert_true((c[i][j] - expected[i][j]).abs() < 1.0e-12)
        }
      }
    }
  }
  assert_eq(x, [[0.0, 0.0], [3.0, 4.0]])
  assert_eq(y, [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
}

///|
test "weighted zero coordinates avoid meaningless overflow" {
  match
    @moonot.weighted_squared_euclidean_cost_matrix([[1.0e308]], [[-1.0e308]], [
      0.0,
    ]) {
    Ok(c) => assert_eq(c, [[0.0]])
    Err(_) => assert_true(false)
  }
  match
    @moonot.weighted_squared_euclidean_cost_matrix([[1.0e308]], [[-1.0e308]], [
      1.0e-320,
    ]) {
    Err(_) => assert_true(false)
    Ok(c) => assert_true(c[0][0] > 3.9e296 && c[0][0] < 4.1e296)
  }
  assert_true(
    @moonot.weighted_squared_euclidean_cost_matrix([[1.0e308]], [[-1.0e308]], [
      1.0,
    ])
    is Err(@moonot.WeightedPointCostError::NumericalOverflow(0, 0)),
  )
}

///|
test "weighted costs validate points even if weight is zero" {
  let inf = @math.exp(1000.0)
  assert_true(
    @moonot.weighted_squared_euclidean_cost_matrix([[inf]], [[0.0]], [0.0])
    is Err(
      @moonot.WeightedPointCostError::InvalidPoints(
        @moonot.PointCostError::NonFiniteSourceCoordinate(0, 0)
      )
    ),
  )
  assert_true(
    @moonot.weighted_squared_euclidean_cost_matrix([[0.0]], [[0.0]], [])
    is Err(@moonot.WeightedPointCostError::WeightDimensionMismatch),
  )
  assert_true(
    @moonot.weighted_squared_euclidean_cost_matrix([[0.0]], [[0.0]], [-1.0])
    is Err(@moonot.WeightedPointCostError::NegativeWeight(0)),
  )
  assert_true(
    @moonot.weighted_squared_euclidean_cost_matrix([[0.0]], [[0.0]], [inf])
    is Err(@moonot.WeightedPointCostError::NonFiniteWeight(0)),
  )
  assert_true(
    @moonot.weighted_squared_euclidean_cost_matrix([[]], [[0.0]], [1.0])
    is Err(
      @moonot.WeightedPointCostError::InvalidPoints(
        @moonot.PointCostError::EmptySourcePoint(0)
      )
    ),
  )
}

///|
test "unit feature weights agree with ordinary squared Euclidean costs" {
  let x = [[-2.0, 5.0], [3.0, 4.0]]
  let y = [[1.0, 2.0]]
  match
    (
      @moonot.weighted_squared_euclidean_cost_matrix(x, y, [1.0, 1.0]),
      @moonot.squared_euclidean_cost_matrix(x, y),
    ) {
    (Ok(a), Ok(b)) =>
      for i in 0..<2 {
        assert_true((a[i][0] - b[i][0]).abs() < 1.0e-12)
      }
    _ => assert_true(false)
  }
}
