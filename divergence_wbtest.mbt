///|
test "scaled divergence sum detects an unrepresentable final result" {
  match sinkhorn_divergence_combine(1.0e308, -1.0e308, -1.0e308) {
    Ok(_) => assert_true(false)
    Err(SinkhornDivergenceError::NumericalFailure) => assert_true(true)
    Err(_) => assert_true(false)
  }
}

///|
test "scaled divergence sum avoids overflowing canceling intermediates" {
  match sinkhorn_divergence_combine(1.0e308, 1.0e308, -1.0e308) {
    Err(_) => assert_true(false)
    Ok(value) => assert_eq(value, 1.0e308)
  }
}
