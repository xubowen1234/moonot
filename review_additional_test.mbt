///|
test "review composition preserves an accumulated minimum subnormal" {
  let mass = 5.0e-324
  let left = [[mass, mass], [mass, mass]]
  let right = [[mass, mass], [mass, mass]]
  let result = match @moonot.coupling_compose(left, right, mass) {
    Ok(plan) => plan
    Err(_) => abort("unexpected composition error")
  }
  for row in result {
    for value in row {
      assert_eq(value, mass)
    }
  }
}

///|
test "review composition sums four subnormal bridge contributions before rounding" {
  let mass = 5.0e-324
  let left = [[mass, mass, mass, mass], [mass, mass, mass, mass]]
  let right = [[mass, mass], [mass, mass], [mass, mass], [mass, mass]]
  match @moonot.coupling_compose(left, right, mass) {
    Err(_) => assert_true(false)
    Ok(plan) => {
      assert_eq(plan, [[mass * 2.0, mass * 2.0], [mass * 2.0, mass * 2.0]])
      plan[0][0] = 0.0
      assert_eq(plan[1][0], mass * 2.0)
      assert_eq(left[0][0], mass)
      assert_eq(right[0][0], mass)
    }
  }
}
