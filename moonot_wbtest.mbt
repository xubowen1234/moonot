///|
fn wb_approximately_equal(
  actual : Double,
  expected : Double,
  tolerance : Double,
) -> Bool {
  let difference = if actual > expected {
    actual - expected
  } else {
    expected - actual
  }
  difference <= tolerance
}

///|
test "logsumexp remains stable for large common offsets" {
  let result = logsumexp([1000.0, 999.0])
  let expected = 1000.0 + @math.ln(1.0 + @math.exp(-1.0))
  assert_true(wb_approximately_equal(result, expected, 1.0e-12))
}

///|
test "marginal residual is the maximum absolute component error" {
  let residual = marginal_residual([[0.4, 0.1], [0.1, 0.4]], [0.5, 0.5], [
    0.49, 0.51,
  ])
  assert_true(wb_approximately_equal(residual, 0.01, 1.0e-12))
}
