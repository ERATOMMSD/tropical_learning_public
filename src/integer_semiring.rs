use crate::semiring::Semiring;
impl Semiring for i64{
    fn zero() -> Self{
        return 0;
    }
    fn one() -> Self{
        return 1;
    }
    fn dist(a: &Self, b: &Self) -> f64{
        return (a - b).abs() as f64;
    }
    fn to_f64(a: &Self) -> f64{
        return *a as f64;
    }
    fn from_f64(a: f64) -> Self{
        return a.round() as Self;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hoge(){
        let x: i64 = Semiring::one();
        assert_eq!(x, 1);
    }
}
