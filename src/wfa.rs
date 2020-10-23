use crate::matrix::Matrix;
use crate::matrix::{MatrixBehavior, VectorBehavior};
use crate::semiring;
use crate::Character;
use std::collections::HashMap;
use std::iter::Iterator;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "A: Character")] 
pub struct WFA<A: Character, S: semiring::Semiring> {
    pub ini: Vec<S>,
    pub fin: Vec<S>,
    pub trans: HashMap<A, Matrix<S>>,
}
impl<A, S> std::fmt::Debug for WFA<A, S>
where
A: Character, S: semiring::Semiring
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "")?;
        writeln!(f, "ini:\t{}", self.ini.print())?;
        writeln!(f, "fin:\t{}", self.fin.print())?;
        for (a, t) in self.trans.iter(){
            writeln!(f, "---trans for {}---", a)?;
            writeln!(f, "{}", t.print())?;
            writeln!(f, "---trans for {}---", a)?;
        }
        return Ok(());
    }
}
impl<A: Character, S: semiring::Semiring> WFA<A, S> {
    pub fn get_config(&self, word: impl Iterator<Item = A>) -> Vec<S> {
        let mut res = self.ini.clone();
        for c in word {
            let trans = self
                .trans
                .get(&c)
                .expect(&format!("A character {} is read!", c));
            res = trans.mult_vec_left(&res);
        }
        return res;
    }
    pub fn run(&self, word: impl Iterator<Item = A>) -> S {
        return self.get_config(word).dot(&self.fin);
    }
    pub fn get_alphabet(&self) -> Vec<A>{
        return self.trans.keys().cloned().collect();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn run() {
        let w = WFA {
            ini: vec![1, 3, 4],
            fin: vec![2, 1, 1],
            trans: crate::map! {
                'a' => make_mat![i64, 0, 0, 3; 0, 0, 3; 1, 0, 0],
                'b' => make_mat![i64, 0, 1, 0; 2, 0, 0; 0, 0, 4]
            },
        };
        let res = w.run("ab".chars());
        let exp = 52;
        assert_eq!(res, exp);
    }
}
