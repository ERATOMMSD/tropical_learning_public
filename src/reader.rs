use crate::semiring;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
#[derive(Debug)]
pub enum WFAReadingError {
    IOError(String),
    SerdeError(String),
    DimOfFinalIsWrong(usize),
    HeightOfTransIsWrong(char, usize),
    WidthOfTransIsWrong(char, usize),
}
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct RawWFA {
    ini: Vec<f64>,
    fin: Vec<f64>,
    trans: HashMap<char, Vec<Vec<f64>>>,
}
pub fn convert_into_proper_wfa<S>(rwfa: &RawWFA) -> crate::wfa::WFA<char, S>
where
    S: semiring::Semiring,
{
    let wfa = crate::wfa::WFA::<char, S> {
        ini: rwfa.ini.iter().map(|v| S::from_f64(*v)).collect(),
        fin: rwfa.fin.iter().map(|v| S::from_f64(*v)).collect(),
        trans: rwfa
            .trans
            .iter()
            .map(|(k, v)| {
                let m: Vec<Vec<S>> = v
                    .iter()
                    .map(|r| r.iter().map(|v| S::from_f64(*v)).collect())
                    .collect();
                (*k, m)
            })
            .into_iter()
            .collect(),
    };
    return wfa;
}
fn parse_as_raw_wfa(cont: &str) -> Result<RawWFA, WFAReadingError> {
    let parsed: serde_json::Result<RawWFA> = serde_json::from_str(cont);
    match &parsed {
        Ok(rwfa) => {
            let d = rwfa.ini.len();
            if rwfa.fin.len() != d {
                return Err(WFAReadingError::DimOfFinalIsWrong(d));
            }
            for (a, m) in rwfa.trans.iter() {
                if m.len() != d {
                    return Err(WFAReadingError::HeightOfTransIsWrong(*a, m.len()));
                }
                for r in m {
                    if r.len() != d {
                        return Err(WFAReadingError::WidthOfTransIsWrong(*a, r.len()));
                    }
                }
            }
            return Ok(rwfa.clone());
        }
        Err(e) => {
            let x = e.to_string();
            return Err(WFAReadingError::SerdeError(x));
        }
    }
}
pub fn read_file_as_wfa(path: &str) -> Result<RawWFA, WFAReadingError> {
    match &mut File::open(path) {
        Err(e) => return Err(WFAReadingError::IOError(e.to_string())),
        Ok(f) => {
            let mut cont = String::new();
            let res = f.read_to_string(&mut cont);
            match res {
                Err(e) => return Err(WFAReadingError::IOError(e.to_string())),
                Ok(_size) => {
                    return parse_as_raw_wfa(&cont);
                }
            }
        }
    }
}
pub fn read_file_as_words(path: &str) -> Vec<Vec<char>> {
    match &mut File::open(path) {
        Err(_e) => panic!("failed to read the eqq file."),
        Ok(f) => {
            let mut contents = String::new();
            let res = f.read_to_string(&mut contents);
            match res {
                Err(_e) => panic!("failed to read the eqq file."),
                Ok(_size) => {
                    let mut words: Vec<Vec<char>> = contents.split('\n').map(|x| x.to_string().chars().collect()).collect();
                    words.remove(words.len() - 1);
                    return words;
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test1() {
        let x = read_file_as_wfa("test_wfa.json").unwrap();
        dbg!(&x);
    }
    #[test]
    fn test2() {
        let x = read_file_as_wfa("test_wfa.json").unwrap();
        let y = convert_into_proper_wfa::<crate::max_plus_semiring::MaxPlusNumSemiring<i64>>(&x);
        dbg!(&y);
    }
}
