use std::ops;
use crate::util;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
pub trait Semiring: Sized + ops::Add<Output=Self> + ops::Mul<Output=Self> + ops::AddAssign + ops::MulAssign + Clone + std::fmt::Debug + std::fmt::Display + Serialize + DeserializeOwned + std::marker::Send{
    fn zero() -> Self;
    fn one() -> Self;
    fn dist(a: &Self, b: &Self) -> f64;
    fn max_dist(a: &Vec<Self>, b: &Vec<Self>) -> f64{
        assert_eq!(a.len(), b.len());
        (0..a.len()).fold(f64::NEG_INFINITY, |acc, i|{util::psuedo_max(acc, Self::dist(&a[i], &b[i]))})
    }   
    fn to_f64(a: &Self) -> f64;
    fn from_f64(a: f64) -> Self;
}
