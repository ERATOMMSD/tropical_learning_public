#![macro_use]
use crate::semiring::Semiring;
pub type Matrix<T> = Vec<Vec<T>>;
pub trait MatrixBehavior<T>
where T: Clone{
    fn row(&self, ind: usize) -> &Vec<T>;
    fn height(&self) -> usize;
    fn width(&self) -> usize;
    fn mult_vec_left(&self, row: &Vec<T>) -> Vec<T>;
    fn zero(height: usize, width: usize) -> Self;
    fn from_f(height: usize, width: usize, f: &dyn Fn(usize, usize) -> T) -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn dot(&self, rhs: &Self) -> Self;
    fn print(&self) -> String;
    fn transpose(&self) -> Self;
    fn at(&self, i: usize, j: usize) -> T{
        return self.row(i)[j].clone();
    }
}
pub trait VectorBehavior<T> {
    fn scale(&self, coef: &T) -> Self;
    fn plus(&self, rhs: &Self) -> Self;
    fn zero(len: usize) -> Self;
    fn dot(&self, rhs: &Self) -> T;
    fn print(&self) -> String;
}
impl<T: Semiring> VectorBehavior<T> for Vec<T> {
    fn scale(&self, coef: &T) -> Self {
        return self.iter().map(|v| coef.clone()*v.clone()).collect();
    }
    fn plus(&self, rhs: &Self) -> Self {
        assert_eq!(self.len(), rhs.len());
        return self
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
    }
    fn zero(len: usize) -> Self {
        let mut res: Self = Vec::new();
        res.resize(len, T::zero());
        return res;
    }
    fn dot(&self, rhs: &Self) -> T{
        assert_eq!(self.len(), rhs.len());
        return self.iter().zip(rhs.iter())
            .map(|(a, b)| a.clone()*b.clone())
            .fold(T::zero(), |acc, x| acc.clone() + x.clone());
    }
    fn print(&self) -> String{
        self.iter().map(|i| {format!("{}", &i)}).collect::<Vec<String>>().join("\t")
    }
}
impl<T: Semiring> MatrixBehavior<T> for Matrix<T> {
    fn row(&self, ind: usize) -> & Vec<T> {
        &self[ind]
    }
    fn width(&self) -> usize {
        match self.get(0) {
            None => 0,
            Some(v) => v.len(),
        }
    }
    fn height(&self) -> usize {
        self.len()
    }
    fn mult_vec_left(&self, row: &Vec<T>) -> Vec<T> {
        assert!(row.len() == self.height());
        let mult_rows = row.iter().zip(self.iter()).map(|(ci, ri)| ri.scale(&ci));
        let init = Vec::<T>::zero(self.width());
        let res = mult_rows.fold(init, |b, ri| {
            return b.plus(&ri);
        });
        return res;
    }
    fn zero(height: usize, width: usize) -> Self{
        let mut res: Self = Vec::new();
        res.resize(height, Vec::<T>::zero(width));
        return res;
    }
    fn from_f(height: usize, width: usize, f: &dyn Fn(usize, usize) -> T) -> Self{
        let mut res = Self::zero(height, width);
        for i_row in 0..height{
            for i_col in 0..width{
                res[i_row][i_col] = f(i_row, i_col);
            }
        }
        return res;
    }
    fn add(&self, rhs: &Self) -> Self{
        assert_eq!((self.height(), self.width()), (rhs.height(), rhs.width()));
        let res = Self::from_f(self.height(), self.width(), &|i, j|{
            self[i][j].clone() + rhs[i][j].clone()
        });
        res
    }
    fn dot(&self, rhs: &Self) -> Self{
        assert_eq!(self.width(), rhs.height());
        let res = Self::from_f(self.height(), self.width(), &|i, j|{
            let mut temp = T::zero();
            for k in 0..self.width(){
                temp += self[i][k].clone()*rhs[k][j].clone();
            }
            temp
        });
        res
    }
    fn print(&self) -> String{
        self.iter().map(|i| {i.print()}).collect::<Vec<String>>().join("\n")
    }
    fn transpose(&self) -> Self{
        Self::from_f(self.width(), self.height(), &|i, j|{self[j][i].clone()})
    }
}
#[macro_export]
macro_rules! make_vec {
    ( $t:ty, $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
#[macro_export]
macro_rules! make_mat {
    ( $t:ty, $($( $x:expr ),*);*) => {
        {
            let mut temp_mat = Vec::new();
            $(
                temp_mat.push(make_vec![$t, $($x),*]);
            )*
            temp_mat 
        }
    };
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;
    #[test]
    fn scale_vec() {
        let x = make_vec![i64, 1, 2, 3];
        let y = x.scale(&2);
        let z = make_vec![i64, 2, 4, 6];
        assert_eq!(y, z);
    }
    #[test]
    fn plus_vec() {
        let x = make_vec![i64, 1, 2, 3];
        let y = make_vec![i64, 8, -1, 2];
        let z = make_vec![i64, 9, 1, 5];
        assert_eq!(x.plus(&y), z);
    }
    #[test]
    fn zero_vec(){
        let x = make_vec![i64, 0, 0, 0];
        let y = Vec::<i64>::zero(3);
        assert_eq!(x, y);
    }
    #[test]
    fn row_mat(){
        let x = make_mat![i64, 1, 2, 3; 4, 5, 6; 7, 8, 9];
        assert_eq!(x.row(1), &make_vec![i64, 4, 5, 6]);
    }
    #[test]
    fn height_mat(){
        let x = make_mat![i64, 1, 2, 3; 4, 5, 6];
        assert_eq!(x.height(), 2);
    }
    #[test]
    fn width_mat(){
        let x = make_mat![i64, 1, 2, 3; 4, 5, 6];
        assert_eq!(x.width(), 3);
    }
    #[test]
    fn mult_vec_left_mat(){
        let x = make_mat![i64, 1, 2, 3; 4, 5, 6];
        let r = make_vec![i64, -1, 2];
        let res = x.mult_vec_left(&r);
        let exp = make_vec![i64, 7, 8, 9];
        assert_eq!(res, exp);
    }
    #[test]
    fn from_f(){
        let x = Matrix::<i64>::from_f(3, 2, &|i, j|{
            (i+2*j).try_into().unwrap() 
        });
        let y = make_mat![i64, 0, 2; 1, 3; 2, 4];
        assert_eq!(x, y);
    }
    #[test]
    fn add_mat(){
        let x = make_mat![i64, 1, 2, 3; 4, 5, 6];
        let y = make_mat![i64, 1, -2, -3; 3, 2, 1];
        let res = x.add(&y);
        let exp = make_mat![i64, 2, 0, 0; 7, 7, 7];
        assert_eq!(res, exp);
    }
    #[test]
    fn dot_mat(){
        let x = make_mat![i64, 5, 6; 7, 8];
        let y = make_mat![i64, 1, 2; 3, 4];
        let res = x.dot(&y);
        let exp = make_mat![i64, 23, 34; 31, 46];
        assert_eq!(res, exp);
    }
    #[test]
    fn dot_vec(){
        let x = make_vec![i64, 5, 6, 7, 8];
        let y = make_vec![i64, 1, 2, 3, 4];
        let res = x.dot(&y);
        let exp = 5 + 12 + 21 + 32;
        assert_eq!(res, exp);
    }
    #[test]
    fn transpose(){
        let x = make_mat![i64, 1, 2, 3; 4, 5, 6; 7, 8, 9];
        let res = x.transpose();
        let exp = make_mat![i64, 1, 4, 7; 2, 5, 8; 3, 6, 9];
        assert_eq!(res, exp);
    }
}
