#![macro_use]
use super::{SolvableSemiring};
use crate::matrix::MatrixBehavior;
use crate::semiring::Semiring;
use std::fmt; // To use abs
extern crate nalgebra as na;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealField{
    Raw(f64)
}
impl RealField{
    pub fn unwrap(&self) -> f64{
        match &self{
            RealField::Raw(v) => v.clone()
        }
    }
}
impl std::cmp::PartialEq for RealField{
    fn eq(&self, _other: &Self) -> bool {
        panic!();
    }
}
impl std::cmp::Eq for RealField{
}
impl std::hash::Hash for RealField{
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        panic!();
    }
}
impl fmt::Display for RealField{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self{
            Self::Raw(v) => write!(f, "{}", v)
        }
    }
}
impl std::ops::Add for RealField{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (RealField::Raw(l), RealField::Raw(r)) => {
                RealField::Raw(r+l)
            }
        }
    }
}
impl std::ops::AddAssign for RealField {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}
impl std::ops::Mul for RealField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (RealField::Raw(l), RealField::Raw(r)) => {
                RealField::Raw(l * r)
            }
        }
    }
}
impl std::ops::MulAssign for RealField {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}
impl Semiring for RealField {
    fn zero() -> Self {
        RealField::Raw(0.0)
    }
    fn one() -> Self {
        RealField::Raw(1.0)
    }
    fn dist(a: &Self, b: &Self) -> f64{
        match (a, b){
            (RealField::Raw(a), RealField::Raw(b)) => {
                (a - b).abs()
            }
        }
    }
    fn to_f64(a: &Self) -> f64{
        a.unwrap()
    }
    fn from_f64(a: f64) -> Self{
        Self::Raw(a)
    }
}
#[derive(Debug)]
enum ErrorSolving {
}
impl fmt::Display for ErrorSolving {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!();
    }
}
impl std::error::Error for ErrorSolving {}
impl SolvableSemiring for RealField {
    fn solve(a: &impl MatrixBehavior<Self>, b: &Vec<Self>, tol: Option<f64>) -> Option<Vec<Self>> {
        Self::solve_dropping_row(a, b, None, tol)
    }
    fn solve_dropping_row(a: &impl MatrixBehavior<Self>, b: &Vec<Self>, drop_row_id: Option<usize>, tol: Option<f64>) -> Option<Vec<Self>>{
        let a_t = na::DMatrix::from_fn(a.height(), a.width(), |i, j| {
            if let Some(drop_row_id) = drop_row_id{
                if i == drop_row_id{
                    return 0.0;
                }
            }
            return a.at(i, j).unwrap()
        }).transpose();
        let b_t = na::DVector::from_fn(b.len(), |i, _|{b[i].unwrap()});
        let x = a_t.clone().svd(true, true).solve(&b_t, 1e-9).unwrap();
        let ax = a_t * x.clone();
        let diff = ax - b_t;
        if diff.iter().any(|v|{v.abs() > tol.unwrap_or(1e-9)}){
            return None;
        }
        let v: Vec<Self> = x.iter().map(|i| {Self::Raw(i.clone())}).collect();
        return Some(v);
    }
    fn need_enclose_column() -> bool{
        return false;
    }
    fn need_reduce_rows() -> bool{
        return false;
    }
}
#[macro_export]
macro_rules! make_r_vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push(RealField::Raw($x));
            )*
            temp_vec
        }
    };
}
#[macro_export]
macro_rules! make_r_mat {
    ($($( $x:expr ),*);*) => {
        {
            let mut temp_mat = Vec::new();
            $(
                temp_mat.push(make_r_vec![$($x),*]);
            )*
            temp_mat 
        }
    };
}
pub fn find_diff_systems_real<A>(a: &dyn Fn(&Vec<A>) -> RealField, b: &dyn Fn(&Vec<A>) -> RealField, alphabet: &Vec<A>, sample_size: usize, max_len: usize, rng: &mut dyn rand::RngCore, eps: f64)-> Option<(Vec<A>, RealField, RealField)>
where A: crate::Character{
    for w in crate::util::make_random_strings(sample_size, max_len, alphabet, rng){
        let a_val = a(&w);
        let b_val = b(&w);
        if (a_val.unwrap() - b_val.unwrap()).abs() > eps{
            return Some((w, a_val, b_val));
        }
    }
    return None;
}
#[cfg(test)]
mod tests {
    use super::*;
    fn embed(v: f64) -> RealField {
        RealField::Raw(v)
    }
    fn assert_almost_eq(a: &Vec<RealField>, b: &Vec<RealField>, eps: f64){
        assert_eq!(a.len(), b.len());
        let x = a.iter().zip(b).all(|(i, j)| {(i.unwrap()-j.unwrap()).abs() < eps});
        if !x{
            dbg!(a);
            dbg!(b);
            panic!("different vectors");
        }
    }
    #[test]
    fn solve1() {
        let a = make_r_mat![1.0, 2.0; 1.0, 4.0];
        let b = make_r_vec![8.0, 26.0];
        let x = RealField::solve(&a, &b, Some(1e-5));
        let exp = Some(make_r_vec![3.0, 5.0]);
        assert_almost_eq(&x.unwrap(), &exp.unwrap(), 1e-5);
    }
    #[test]
    fn solve2() {
        let a = make_r_mat![2.0, 4.0, 2.0, 3.0;
        4.0, 10.0, 6.0, 7.0;
        2.0, 3.0, 1.0, 1.0;
        2.0, 3.0, 1.0, 4.0];
        let b = make_r_vec![8.0, 17.0, 9.0, 11.0];
        let x = RealField::solve(&a, &b, Some(1e-5));
        assert_almost_eq(&a.mult_vec_left(&x.unwrap()), &b, 1e-5);
    }
    #[test]
    fn solve3() {
        let a = make_r_mat![1.0, 0.0, 2.0;
        0.0, 1.0, 4.0;
        2.0, -1.0, 0.0];
        let b = make_r_vec![0.0, 1.0, 4.0];
        let x = RealField::solve_dropping_row(&a, &b, Some(1), Some(1e-5));
        assert_ne!(x, None);
    }
    #[test]
    fn solve4() {
        let a = make_r_mat![1.0, 0.0, 2.0;
        2.0, -1.0, 0.0];
        let b = make_r_vec![1.0, 0.0, 2.0];
        let x = RealField::solve_dropping_row(&a, &b, Some(1), Some(1e-5));
        assert_ne!(x, None);
    }
    #[test]
    fn approx1() {
        let a = make_r_mat![1.0, 1.0, 1.0, 1.0;
        1.0, 3.0, 4.0, 6.0];
        let b = make_r_vec![1.0, 3.0/2.0, 5.0, 11.0/2.0];
        let x = RealField::solve_dropping_row(&a, &b, None, Some(10.0)).unwrap();
        assert_almost_eq(&x, &make_r_vec![-0.25, 1.0], 1e-5);
    }
}
