///|
test "generalized I-divergence honors both zero-reference conventions" {
  match unbalanced_kl_from_log_reference(0.0, @math.ln(3.0)) {
    Err(_) => assert_true(false)
    Ok(value) => assert_true(wb_approximately_equal(value, 3.0, 1.0e-14))
  }
  match unbalanced_kl_from_log_reference(2.0, @math.ln(2.0)) {
    Err(_) => assert_true(false)
    Ok(value) => assert_eq(value, 0.0)
  }
}

///|
test "generalized I-divergence near equality avoids cancellation" {
  match unbalanced_kl_from_log_reference(1.00000001, 0.0) {
    Err(_) => assert_true(false)
    Ok(value) => {
      assert_true(value >= 0.0)
      assert_true(wb_approximately_equal(value, 5.0e-17, 1.0e-24))
    }
  }
}
