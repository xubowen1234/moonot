///|
fn projection_close(actual : Double, expected : Double) -> Bool {
  let difference = if actual > expected {
    actual - expected
  } else {
    expected - actual
  }
  difference <= 1.0e-12
}

///|
test "barycentric projection computes conditional vector means" {
  match
    @moonot.coupling_barycentric_projection([[0.25, 0.75], [0.5, 0.5]], [
      [0.0, 10.0],
      [10.0, 30.0],
    ]) {
    Err(_) => assert_true(false)
    Ok(projected) => {
      assert_true(projection_close(projected[0][0], 7.5))
      assert_true(projection_close(projected[0][1], 25.0))
      assert_true(projection_close(projected[1][0], 5.0))
      assert_true(projection_close(projected[1][1], 20.0))
    }
  }
}

///|
test "barycentric projection is invariant to positive row scaling" {
  let values = [[-4.0], [2.0], [10.0]]
  match @moonot.coupling_barycentric_projection([[1.0, 2.0, 1.0]], values) {
    Err(_) => assert_true(false)
    Ok(first) =>
      match
        @moonot.coupling_barycentric_projection(
          [[1.0e100, 2.0e100, 1.0e100]],
          values,
        ) {
        Err(_) => assert_true(false)
        Ok(scaled) => assert_true(projection_close(first[0][0], scaled[0][0]))
      }
  }
}

///|
test "barycentric projection rejects zero source mass rows" {
  assert_true(
    @moonot.coupling_barycentric_projection([[0.0, 0.0], [0.25, 0.75]], [
      [0.0],
      [1.0],
    ])
    is Err(@moonot.CouplingProjectionError::ZeroMassSourceRow(0)),
  )
}

///|
test "barycentric projection validates target feature shape and values" {
  assert_true(
    @moonot.coupling_barycentric_projection([[1.0]], [])
    is Err(@moonot.CouplingProjectionError::TargetValueCountMismatch),
  )
  assert_true(
    @moonot.coupling_barycentric_projection([[1.0]], [[]])
    is Err(@moonot.CouplingProjectionError::EmptyTargetFeatureVector),
  )
  assert_true(
    @moonot.coupling_barycentric_projection([[0.5, 0.5]], [[0.0], [1.0, 2.0]])
    is Err(@moonot.CouplingProjectionError::RaggedTargetValuesRow(1, 1, 2)),
  )
  let infinity = @math.exp(1000.0)
  assert_true(
    @moonot.coupling_barycentric_projection([[1.0]], [[infinity]])
    is Err(@moonot.CouplingProjectionError::NonFiniteTargetValue(0, 0)),
  )
}

///|
test "barycentric projection preserves zero mass target columns" {
  match
    @moonot.coupling_barycentric_projection([[0.25, 0.75, 0.0]], [
      [0.0],
      [4.0],
      [1.0e308],
    ]) {
    Err(_) => assert_true(false)
    Ok(projected) => assert_true(projection_close(projected[0][0], 3.0))
  }
}

///|
test "barycentric projection remains finite at extreme coordinate scales" {
  match
    @moonot.coupling_barycentric_projection([[0.5, 0.5]], [[1.0e308], [1.0e308]]) {
    Err(_) => assert_true(false)
    Ok(projected) => assert_true(projected[0][0] == 1.0e308)
  }
  match
    @moonot.coupling_barycentric_projection([[0.5, 0.5]], [
      [-1.0e308],
      [1.0e308],
    ]) {
    Err(_) => assert_true(false)
    Ok(projected) =>
      assert_true(projected[0][0] >= -1.0e292 && projected[0][0] <= 1.0e292)
  }
}
