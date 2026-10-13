///|
test "point cost builders return rectangular source-by-target matrices" {
  let source = [[0.0, 0.0], [3.0, 4.0]]
  let target = [[0.0, 0.0], [0.0, 2.0], [3.0, 4.0]]
  match @moonot.squared_euclidean_cost_matrix(source, target) {
    Err(_) => assert_true(false)
    Ok(cost) => assert_eq(cost, [[0.0, 4.0, 25.0], [25.0, 13.0, 0.0]])
  }
  match @moonot.euclidean_cost_matrix(source, target) {
    Err(_) => assert_true(false)
    Ok(cost) => {
      assert_eq(cost[0][1], 2.0)
      assert_true(cost[1][1] > 3.6055512754639 && cost[1][1] < 3.6055512754641)
    }
  }
  match @moonot.euclidean_cost_matrix([[1.0e200]], [[0.0]]) {
    Err(_) => assert_true(false)
    Ok(cost) => assert_eq(cost[0][0], 1.0e200)
  }
  match @moonot.manhattan_cost_matrix(source, target) {
    Err(_) => assert_true(false)
    Ok(cost) => assert_eq(cost, [[0.0, 2.0, 7.0], [7.0, 5.0, 0.0]])
  }
}

///|
test "point cost builders reject empty supports and zero-dimensional points" {
  assert_true(
    @moonot.squared_euclidean_cost_matrix([], [[0.0]])
    is Err(@moonot.PointCostError::EmptySourceSupport),
  )
  assert_true(
    @moonot.euclidean_cost_matrix([[0.0]], [])
    is Err(@moonot.PointCostError::EmptyTargetSupport),
  )
  assert_true(
    @moonot.manhattan_cost_matrix([[]], [[0.0]])
    is Err(@moonot.PointCostError::EmptySourcePoint(0)),
  )
  assert_true(
    @moonot.manhattan_cost_matrix([[0.0]], [[]])
    is Err(@moonot.PointCostError::EmptyTargetPoint(0)),
  )
}

///|
test "point cost builders reject ragged and incompatible point dimensions" {
  assert_true(
    @moonot.squared_euclidean_cost_matrix([[0.0], [1.0, 2.0]], [[0.0]])
    is Err(@moonot.PointCostError::SourcePointDimensionMismatch(1, 1, 2)),
  )
  assert_true(
    @moonot.euclidean_cost_matrix([[0.0]], [[0.0], [1.0, 2.0]])
    is Err(@moonot.PointCostError::TargetPointDimensionMismatch(1, 1, 2)),
  )
  assert_true(
    @moonot.manhattan_cost_matrix([[0.0, 1.0]], [[0.0]])
    is Err(@moonot.PointCostError::PointDimensionMismatch(2, 1)),
  )
}

///|
test "point cost builders identify non-finite input coordinates" {
  let infinity = @math.exp(1000.0)
  let nan = infinity - infinity
  assert_true(
    @moonot.squared_euclidean_cost_matrix([[0.0, nan]], [[0.0, 0.0]])
    is Err(@moonot.PointCostError::NonFiniteSourceCoordinate(0, 1)),
  )
  assert_true(
    @moonot.euclidean_cost_matrix([[0.0]], [[infinity]])
    is Err(@moonot.PointCostError::NonFiniteTargetCoordinate(0, 0)),
  )
}

///|
test "point cost builders report unrepresentable distances" {
  assert_true(
    @moonot.squared_euclidean_cost_matrix([[1.0e308]], [[-1.0e308]])
    is Err(@moonot.PointCostError::NumericalOverflow(0, 0)),
  )
  assert_true(
    @moonot.manhattan_cost_matrix([[1.0e308, 1.0e308]], [[0.0, 0.0]])
    is Err(@moonot.PointCostError::NumericalOverflow(0, 0)),
  )
}
