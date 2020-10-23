#![allow(dead_code)]
pub mod semiring;
mod matrix;
pub mod max_plus_semiring;
mod integer_semiring;
pub mod real_field;
pub mod wfa;
pub mod util;
pub mod learning;
pub mod reader;
mod stopwatch;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
pub mod extension;
pub trait Character: std::hash::Hash + std::cmp::Eq + std::fmt::Display + Sized + Copy + std::fmt::Debug + Serialize + DeserializeOwned + std::marker::Send{
}
impl Character for char{
}
pub struct SolveApproxResult<T: SolvableSemiring> {
    res: Vec<T>,
    tol: f32,
}
pub trait SolvableSemiring: semiring::Semiring {
    fn solve_dropping_row(a: &impl matrix::MatrixBehavior<Self>, b: &Vec<Self>, drop_row_id: Option<usize>, eps: Option<f64>) -> Option<Vec<Self>>;
    fn solve(a: &impl matrix::MatrixBehavior<Self>, b: &Vec<Self>, eps: Option<f64>) -> Option<Vec<Self>>{
        SolvableSemiring::solve_dropping_row(a, b, None, eps)
    }
    fn solve_xab(a: &impl matrix::MatrixBehavior<Self>, b: &impl matrix::MatrixBehavior<Self>, eps: Option<f64>) -> Option<Vec<Vec<Self>>>{
        assert_eq!(a.width(), b.width());
        let d = b.height();
        let mut res = Vec::<Vec<Self>>::new();
        for rid in 0..d{
            let sol = SolvableSemiring::solve(a, b.row(rid), eps)?;
            res.push(sol);
        }
        Some(res)
    }
    fn solve_axb(a: &impl matrix::MatrixBehavior<Self>, b: &impl matrix::MatrixBehavior<Self>, eps: Option<f64>) -> Option<Vec<Vec<Self>>>{
        assert_eq!(a.height(), b.height());
        let res = SolvableSemiring::solve_xab(&a.transpose(), &b.transpose(), eps)?;
        Some(matrix::MatrixBehavior::transpose(&res))
    }
    fn need_enclose_column() -> bool;
    fn need_reduce_rows() -> bool;
}
