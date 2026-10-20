///|
test "review projection ignores finite inactive coordinates at tiny active scale" {
  match
    @moonot.coupling_barycentric_projection([[1.0, 0.0]], [
      [1.0e-300],
      [1.0e300],
    ]) {
    Err(_) => assert_true(false)
    Ok(p) => assert_true((p[0][0] / 1.0e-300 - 1.0).abs() < 1.0e-12)
  }
}

///|
test "review projection retains representable means across dynamic range" {
  match
    @moonot.coupling_barycentric_projection([[1.0e-200, 1.0e200]], [
      [1.0e200],
      [0.0],
    ]) {
    Err(_) => assert_true(false)
    Ok(p) => assert_true((p[0][0] / 1.0e-200 - 1.0).abs() < 1.0e-12)
  }
}

///|
test "review composition retains representable tiny target mass" {
  match @moonot.coupling_compose([[1.0e200]], [[1.0e-200, 1.0e200]], 1.0e-12) {
    Err(_) => assert_true(false)
    Ok(p) => {
      assert_true((p[0][0] / 1.0e-200 - 1.0).abs() < 1.0e-12)
      assert_eq(p[0][1], 1.0e200)
    }
  }
}

///|
test "review projection handles subnormal row mass without normalization loss" {
  match
    @moonot.coupling_barycentric_projection([[5.0e-324, 5.0e-324]], [
      [2.0],
      [-1.0],
    ]) {
    Err(_) => assert_true(false)
    Ok(p) => assert_eq(p, [[0.5]])
  }
}

///|
test "review projection retains signs dimensions and scale invariance" {
  let values = [[-4.0, 2.0e200], [4.0, -2.0e200], [-1.0e300, 1.0e300]]
  let coupling = [[1.0e-200, 3.0e-200, 0.0], [1.0e100, 3.0e100, 0.0]]
  match @moonot.coupling_barycentric_projection(coupling, values) {
    Err(_) => assert_true(false)
    Ok(p) => {
      for row in p {
        assert_true((row[0] - 2.0).abs() < 1.0e-12)
        assert_true((row[1] / -1.0e200 - 1.0).abs() < 1.0e-12)
      }
      p[0][0] = 99.0
      assert_true((p[1][0] - 2.0).abs() < 1.0e-12)
    }
  }
  assert_eq(coupling, [[1.0e-200, 3.0e-200, 0.0], [1.0e100, 3.0e100, 0.0]])
  assert_eq(values, [[-4.0, 2.0e200], [4.0, -2.0e200], [-1.0e300, 1.0e300]])
}

///|
test "review inactive coordinates remain subject to validation" {
  assert_true(
    @moonot.coupling_barycentric_projection([[1.0, 0.0]], [
      [1.0e-300],
      [1.0 / 0.0],
    ])
    is Err(@moonot.CouplingProjectionError::NonFiniteTargetValue(1, 0)),
  )
  assert_true(
    @moonot.coupling_barycentric_projection([[0.0, 0.0]], [[1.0], [2.0]])
    is Err(@moonot.CouplingProjectionError::ZeroMassSourceRow(0)),
  )
}

///|
test "review composition preserves subnormal mass and skips empty bridges" {
  match
    @moonot.coupling_compose(
      [[5.0e-324, 0.0]],
      [[5.0e-324, 0.0], [0.0, 0.0]],
      1.0e-12,
    ) {
    Err(_) => assert_true(false)
    Ok(p) => assert_eq(p, [[5.0e-324, 0.0]])
  }
}

///|
test "review composition avoids intermediate overflow without mutating plans" {
  let left = [[1.0e308]]
  let right = [[1.0e308, 1.0e-200]]
  match @moonot.coupling_compose(left, right, 1.0e-12) {
    Err(_) => assert_true(false)
    Ok(p) => {
      assert_true((p[0][0] / 1.0e308 - 1.0).abs() < 1.0e-12)
      assert_true((p[0][1] / 1.0e-200 - 1.0).abs() < 1.0e-12)
      p[0][0] = 99.0
    }
  }
  assert_eq(left, [[1.0e308]])
  assert_eq(right, [[1.0e308, 1.0e-200]])
}

///|
test "review composition keeps bridge mismatch and undefined conditional errors" {
  assert_true(
    @moonot.coupling_compose([[1.0]], [[2.0]], 1.0e-12)
    is Err(@moonot.CouplingCompositionError::BridgeMassMismatch(0, _, _)),
  )
  assert_true(
    @moonot.coupling_compose([[5.0e-324]], [[0.0]], 1.0e-12)
    is Err(@moonot.CouplingCompositionError::UndefinedBridgeConditional(0)),
  )
}
