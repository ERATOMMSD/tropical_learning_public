#![macro_use]
use super::{SolvableSemiring};
use crate::matrix::MatrixBehavior;
use crate::semiring::Semiring;
use fmt::Display;
use num::{Num};
use std::default;
use std::fmt; // To use abs
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
pub trait MaxPlusNumSemiringBase: Num + Display + Clone + std::fmt::Debug + std::cmp::PartialOrd + std::cmp::PartialEq + Serialize + DeserializeOwned + std::marker::Send {
    fn to_f64(a: Self) -> f64;
    fn from_f64(a: f64) -> Self;
}
impl MaxPlusNumSemiringBase for i64 {
    fn to_f64(a: Self) -> f64{
        return a as f64;
    }
    fn from_f64(a: f64) -> Self{
        return a.round() as i64;
    }
}
impl MaxPlusNumSemiringBase for f64 {
    fn to_f64(a: Self) -> f64{
        return a;
    }
    fn from_f64(a: f64) -> Self{
        return a;
    }
}
#[serde(bound = "T: MaxPlusNumSemiringBase")] 
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize,  Deserialize)]
pub enum MaxPlusNumSemiring<T: MaxPlusNumSemiringBase> {
    NegInf,
    Raw(T),
}
impl<T: MaxPlusNumSemiringBase> fmt::Display for MaxPlusNumSemiring<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaxPlusNumSemiring::Raw(v) => write!(f, "{}", v),
            MaxPlusNumSemiring::NegInf => write!(f, "-∞"),
        }
    }
}
impl<T: MaxPlusNumSemiringBase> default::Default for MaxPlusNumSemiring<T> {
    fn default() -> Self {
        MaxPlusNumSemiring::NegInf
    }
}
impl<T: MaxPlusNumSemiringBase> std::ops::Add for MaxPlusNumSemiring<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (MaxPlusNumSemiring::Raw(l), MaxPlusNumSemiring::Raw(r)) => {
                MaxPlusNumSemiring::Raw(if l > r{
                    l
                }else{
                    r
                })
            }
            (MaxPlusNumSemiring::Raw(l), MaxPlusNumSemiring::NegInf) => MaxPlusNumSemiring::Raw(l),
            (MaxPlusNumSemiring::NegInf, MaxPlusNumSemiring::Raw(r)) => MaxPlusNumSemiring::Raw(r),
            _ => MaxPlusNumSemiring::NegInf,
        }
    }
}
impl<T: MaxPlusNumSemiringBase> std::ops::AddAssign for MaxPlusNumSemiring<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}
impl<T: MaxPlusNumSemiringBase> std::ops::Mul for MaxPlusNumSemiring<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (MaxPlusNumSemiring::Raw(l), MaxPlusNumSemiring::Raw(r)) => {
                MaxPlusNumSemiring::Raw(l + r)
            }
            _ => MaxPlusNumSemiring::NegInf,
        }
    }
}
impl<T: MaxPlusNumSemiringBase> std::ops::MulAssign for MaxPlusNumSemiring<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}
impl<T: MaxPlusNumSemiringBase> Semiring for MaxPlusNumSemiring<T> {
    fn zero() -> Self {
        MaxPlusNumSemiring::NegInf
    }
    fn one() -> Self {
        MaxPlusNumSemiring::Raw(T::zero())
    }
    fn dist(a: &Self, b: &Self) -> f64{
        match (a, b){
            (Self::Raw(a), Self::Raw(b)) => {
                return (T::to_f64(a.clone()) - T::to_f64(b.clone())).abs();
            },
            (Self::NegInf, Self::NegInf) => {
                0.0
            }
            _ => {
                return f64::INFINITY;
            }
        }
    }
    fn to_f64(a: &Self) -> f64{
        match a{
            Self::NegInf => panic!(),
            Self::Raw(r) => T::to_f64(r.clone())
        }
    }
    fn from_f64(a: f64) -> Self{
        Self::Raw(T::from_f64(a))
    }
}
#[derive(Debug)]
enum ErrorSolving {
    NegInfRowAppeared,
}
impl fmt::Display for ErrorSolving {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSolving::NegInfRowAppeared => write!(f, "A row only filled with -∞ appeared!"),
        }
    }
}
impl std::error::Error for ErrorSolving {}
impl<T: MaxPlusNumSemiringBase> MaxPlusNumSemiring<T> {
    fn get_principal_solution_dropping_row(a: &impl MatrixBehavior<Self>, b: &Vec<Self>, drop_row_id: Option<usize>) -> Vec<Self> {
        assert_eq!(a.width(), b.len());
        assert!(b.len() > 0);
        if a.height() == 0 || a.height() == 1 && drop_row_id != None{
            panic!();
        }
        let x_bar: Vec<Self> = (0..a.height())
            .map(|row| {
                if drop_row_id == Some(row){
                    return Self::zero();
                }
                let row = a.row(row);
                let shifts: Vec<Self> = b
                    .iter()
                    .zip(row.iter())
                    .map(|(bj, aij)| match (bj, aij) {
                        (Self::Raw(bj), Self::Raw(aij)) => {
                            Some(Self::Raw(bj.clone() - aij.clone()))
                        }
                        (Self::NegInf, Self::Raw(_)) => Some(Self::NegInf),
                        (_, Self::NegInf) => None,
                    })
                    .filter(|v| v.as_ref() != None)
                    .map(|v| match v {
                        None => panic!("cannot reach here!"),
                        Some(v) => v,
                    }).collect();
                if shifts.is_empty(){
                    return Self::zero();
                }else{
                    return shifts
                    .iter()
                    .fold(shifts[0].clone(), |i, j|{
                        if i < *j{
                            i
                        }else{
                            j.clone()
                        }
                    })
                }
            })
            .collect();
        assert_eq!(x_bar.len(), a.height());
        return x_bar;
    }
    fn get_principal_solution(a: &impl MatrixBehavior<Self>, b: &Vec<Self>) -> Vec<Self> {
        Self::get_principal_solution_dropping_row(a, b, None)
    }
}
impl<T: MaxPlusNumSemiringBase> SolvableSemiring for MaxPlusNumSemiring<T> {
    fn solve(a: &impl MatrixBehavior<Self>, b: &Vec<Self>, eps: Option<f64>) -> Option<Vec<Self>> {
        Self::solve_dropping_row(a, b, None, eps)
    }
    fn solve_dropping_row(a: &impl MatrixBehavior<Self>, b: &Vec<Self>, drop_row_id: Option<usize>, eps: Option<f64>) -> Option<Vec<Self>>{
        if a.height() == 0 || a.height() == 1 && drop_row_id != None{
            return None
        }
        let x_bar = Self::get_principal_solution_dropping_row(a, b, drop_row_id);
        let xa = a.mult_vec_left(&x_bar);
        match eps{
            None => {
                if xa == *b {
                    return Some(x_bar);
                } else {
                    return None;
                }
            },
            Some(eps) =>{
                let max_diff = MaxPlusNumSemiring::max_dist(b, &xa);
                let shifting = Self::Raw(T::from_f64(max_diff / 2.0));
                let ans: Vec<Self> = x_bar.iter().map(|i|{
                    i.clone() * shifting.clone()
                }).collect();
                let xa_eps = a.mult_vec_left(&ans);
                if MaxPlusNumSemiring::max_dist(b, &xa_eps) <= eps{
                    return Some(ans);
                }else{
                    return None;
                }
            }
        }
    }
    fn need_enclose_column() -> bool{
        return true;
    }
    fn need_reduce_rows() -> bool{
        return true;
    }
}
#[macro_export]
macro_rules! make_mp_vec {
    ( $t: ty, $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push(MaxPlusNumSemiring::<$t>::Raw($x));
            )*
            temp_vec
        }
    };
}
#[macro_export]
macro_rules! make_mp_mat {
    ( $t: ty, $($( $x:expr ),*);*) => {
        {
            let mut temp_mat = Vec::new();
            $(
                temp_mat.push(make_mp_vec![$t, $($x),*]);
            )*
            temp_mat 
        }
    };
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::make_mat;
    fn embed(v: i64) -> MaxPlusNumSemiring<i64> {
        MaxPlusNumSemiring::Raw(v)
    }
    #[test]
    fn add_on_base() {
        let mut x = embed(2);
        x += embed(3);
        assert_eq!(x, embed(3));
        assert_eq!(embed(2) + embed(3), embed(3));
    }
    #[test]
    fn add_infinite() {
        assert_eq!(embed(-2) + MaxPlusNumSemiring::NegInf, embed(-2));
    }
    #[test]
    fn mult_on_base() {
        assert_eq!(embed(2) * embed(3), embed(5));
    }
    #[test]
    fn mult_infinite() {
        assert_eq!(
            embed(-2) * MaxPlusNumSemiring::NegInf,
            MaxPlusNumSemiring::NegInf
        );
        assert_eq!(
            MaxPlusNumSemiring::<i64>::NegInf * MaxPlusNumSemiring::NegInf,
            MaxPlusNumSemiring::NegInf
        );
    }
    #[test]
    fn mult_mat() {
        let x = make_vec![MaxPlusNumSemiring<i64>, r(3), r(1), r(-2)];
        let a = make_mat![MaxPlusNumSemiring<i64>,
        r(-2), r(-5), eps(), r(-3), r(1);
        r(2), r(-3), eps(), r(-3), r(4);
        r(2), r(-2), r(3), r(2), eps()];
        let exp = make_vec![MaxPlusNumSemiring<i64>, r(3), r(-2), r(1), r(0), r(5)];
        let res = a.mult_vec_left(&x);
        assert_eq!(exp, res);
    }
    #[test]
    fn ord() {
        assert_eq!(
            MaxPlusNumSemiring::<i64>::NegInf,
            MaxPlusNumSemiring::NegInf
        );
        assert!(MaxPlusNumSemiring::<i64>::NegInf <= MaxPlusNumSemiring::NegInf);
        assert!(MaxPlusNumSemiring::<i64>::NegInf <= MaxPlusNumSemiring::Raw(4));
        assert!(MaxPlusNumSemiring::Raw(1) <= MaxPlusNumSemiring::Raw(4));
        assert!(!(MaxPlusNumSemiring::Raw(1) <= MaxPlusNumSemiring::Raw(-4)));
        assert!(!(MaxPlusNumSemiring::Raw(1) <= MaxPlusNumSemiring::NegInf));
    }
    fn r(v: i64) -> MaxPlusNumSemiring<i64> {
        return MaxPlusNumSemiring::Raw(v);
    }
    fn eps() -> MaxPlusNumSemiring<i64> {
        return MaxPlusNumSemiring::NegInf;
    }
    fn fr(v: f64) -> MaxPlusNumSemiring<f64> {
        return MaxPlusNumSemiring::Raw(v);
    }
    fn feps() -> MaxPlusNumSemiring<f64> {
        return MaxPlusNumSemiring::NegInf;
    }
    #[test]
    fn test1_solve() {
        let a = make_mat![MaxPlusNumSemiring<i64>,
        r(-2), r(-5), eps(), r(-3), r(1);
        r(2), r(-3), eps(), r(-3), r(4);
        r(2), r(-2), r(3), r(2), eps()];
        let b = make_vec![MaxPlusNumSemiring<i64>, r(3), r(-2), r(1), r(0), r(5)];
        let x = MaxPlusNumSemiring::<i64>::get_principal_solution(&a, &b);
        let exp = make_vec![MaxPlusNumSemiring<i64>, r(3), r(1), r(-2)];
        assert_eq!(x, exp);
        let x = MaxPlusNumSemiring::<i64>::solve(&a, &b, None);
        assert_eq!(x, Some(exp));
    }
    #[test]
    fn test2_solve() {
        let a = make_mat![MaxPlusNumSemiring<i64>,
        r(1), r(2), r(3);
        r(1), r(0), r(1)];
        let b = make_vec![MaxPlusNumSemiring<i64>, r(12), r(12), r(13)];
        let x = MaxPlusNumSemiring::<i64>::get_principal_solution(&a, &b);
        let exp = make_vec![MaxPlusNumSemiring<i64>, r(10), r(11)];
        assert_eq!(x, exp);
        let x = MaxPlusNumSemiring::<i64>::solve(&a, &b, None);
        assert_eq!(x, Some(exp));
    }
    #[test]
    fn test3_solve() {
        let a = make_mat![MaxPlusNumSemiring<i64>,
        r(1), r(2), r(3);
        r(0), r(1), r(0)];
        let b = make_vec![MaxPlusNumSemiring<i64>, r(12), r(12), r(13)];
        let x = MaxPlusNumSemiring::<i64>::get_principal_solution(&a, &b);
        let exp = make_vec![MaxPlusNumSemiring<i64>, r(10), r(11)];
        assert_eq!(x, exp);
        let x = MaxPlusNumSemiring::<i64>::solve(&a, &b, None);
        assert_eq!(x, None);
    }
    #[test]
    fn test4_solve() {
        let a = make_mat![MaxPlusNumSemiring<i64>,
        r(2), r(2), r(2);
        r(1), eps(), r(2)];
        let b = make_vec![MaxPlusNumSemiring<i64>, r(1), eps(), r(1)];
        let x = MaxPlusNumSemiring::<i64>::get_principal_solution(&a, &b);
        assert_eq!(x, make_vec![MaxPlusNumSemiring<i64>, eps(), r(-1)]);
        assert_eq!(MaxPlusNumSemiring::<i64>::solve(&a, &b, None), None);
    }
    #[test]
    fn test5_solve() {
        let a = make_mat![MaxPlusNumSemiring<i64>,
        eps(), eps(), eps();
        eps(), eps(), eps()];
        let b = make_vec![MaxPlusNumSemiring<i64>, r(1), eps(), r(1)];
        let x = MaxPlusNumSemiring::<i64>::get_principal_solution(&a, &b);
        assert_eq!(x, make_vec![MaxPlusNumSemiring<i64>, eps(), eps()]);
        assert_eq!(MaxPlusNumSemiring::<i64>::solve(&a, &b, None), None);
    }
    #[test]
    #[should_panic]
    fn test5_2_solve() {
        let a = make_mat![MaxPlusNumSemiring<i64>,
        eps(), eps(), eps();
        eps(), eps(), eps()];
        let b = make_vec![MaxPlusNumSemiring<i64>, r(1), eps(), r(1)];
        let _ = MaxPlusNumSemiring::<i64>::solve(&a, &b, None).unwrap();
    }
    #[test]
    fn test6_solve(){
        let b = make_vec![MaxPlusNumSemiring<i64>,
        r(11), r(22), r(19), r(16), r(20), r(10), r(14), r(0), r(17), r(4), r(1)];
        let a = make_mat![MaxPlusNumSemiring<i64>,
        r(14), r(25), r(22), r(19), r(23), r(13), r(17), r(1), r(20), r(7), r(4);
        r(1), r(15), r(14), r(22), r(12), r(16), r(6), r(9), r(9), r(11), r(10)
        ];
        let res = MaxPlusNumSemiring::<i64>::solve(&a, &b, None);
        assert_eq!(res, Some(vec![r(-3), r(-9)]));
    }
    #[test]
    fn test_approx_solve(){
        let b = make_vec![MaxPlusNumSemiring<i64>,
        r(0), r(4)];
        let a = make_mat![MaxPlusNumSemiring<i64>,
        r(0), r(0)
        ];
        let res = MaxPlusNumSemiring::<i64>::solve(&a, &b, None);
        assert_eq!(res, None);
        let res = MaxPlusNumSemiring::<i64>::solve(&a, &b, Some(1.0));
        assert_eq!(res, None);
        let res = MaxPlusNumSemiring::<i64>::solve(&a, &b, Some(2.0));
        assert_eq!(res, Some(vec![r(2)]));
        let res = MaxPlusNumSemiring::<i64>::solve(&a, &b, Some(3.0));
        assert_eq!(res, Some(vec![r(2)]));
    }
    #[test]
    fn test_dist(){
        dbg!(MaxPlusNumSemiring::<f64>::dist(&fr(1.0), &fr(2.0)));
        dbg!(MaxPlusNumSemiring::<f64>::dist(&feps(), &fr(2.0)));
        dbg!(MaxPlusNumSemiring::<f64>::dist(&feps(), &feps()));
    }
    #[test]
    fn test_dist2(){
        dbg!(MaxPlusNumSemiring::<f64>::max_dist(&vec![fr(1.0), fr(2.0)], &vec![fr(1.0), fr(2.0)]));
        dbg!(MaxPlusNumSemiring::<f64>::max_dist(&vec![fr(1.1), fr(2.0)], &vec![fr(1.0), fr(2.0)]));
        dbg!(MaxPlusNumSemiring::<f64>::max_dist(&vec![feps(), fr(2.0)], &vec![fr(1.0), fr(2.0)]));
        dbg!(MaxPlusNumSemiring::<i64>::max_dist(&vec![eps(), r(-8), r(-3), r(2), r(15)], (&vec![eps(), r(-8), r(-3), r(2), r(15)])));
        assert!(f64::INFINITY > 1.0);
    }
}
