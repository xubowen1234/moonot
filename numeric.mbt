///|
/// Split a positive finite product quotient without creating its potentially
/// overflowing product or underflowing ratio. Callers validate positivity and
/// finiteness. Binary exponent scaling also handles subnormal inputs.
fn product_quotient_parts(
  left : Double,
  right : Double,
  denominator : Double,
) -> (Double, Int) {
  let left_exponent = @math.log2(left).floor().to_int()
  let right_exponent = @math.log2(right).floor().to_int()
  let denominator_exponent = @math.log2(denominator).floor().to_int()
  let significand = @math.scalbn(left, -left_exponent) *
    @math.scalbn(right, -right_exponent) /
    @math.scalbn(denominator, -denominator_exponent)
  (significand, left_exponent + right_exponent - denominator_exponent)
}
