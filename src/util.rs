#![macro_use]
use rand;
use rand::Rng;
use crate::Character;
#[macro_export]
macro_rules! map(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::HashMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);
fn random_choose<T>(v: &Vec<T>, rng: &mut dyn rand::RngCore) -> T where T: Copy{
    return v[rng.gen_range(0, v.len())];
}
fn make_random_string<T>(len: usize, alphabet: &Vec<T>, rng: &mut dyn rand::RngCore) -> Vec<T> where T: Character{
    return (0..len).map(|_| random_choose(alphabet, rng)).collect();
}
pub fn make_random_strings<T>(num: usize, max_len: usize, alphabet: &Vec<T>, rng: &mut dyn rand::RngCore) -> Vec<Vec<T>> where T: Character{
    return (0..num).map(|_| make_random_string(rng.gen_range(0, max_len+1), alphabet, rng)).collect();
}
pub fn find_diff_systems<A, S>(a: &dyn Fn(&Vec<A>) -> S, b: &dyn Fn(&Vec<A>) -> S, alphabet: &Vec<A>, sample_size: usize, max_len: usize, eps: f64, rng: &mut dyn rand::RngCore)-> Option<(Vec<A>, S, S)>
where A: crate::Character, S: crate::semiring::Semiring {
    for w in make_random_strings(sample_size, max_len, alphabet, rng){
        let a_val = a(&w);
        let b_val = b(&w);
        if S::dist(&a_val, &b_val) > eps{
            return Some((w, a_val, b_val));
        }
    }
    return None;
}
pub fn join_vec<T>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> where T: Clone{
    let mut res: Vec<T> = a.to_vec();
    for c in b.iter(){
        res.push(c.clone());
    }
    return res;
}
pub fn psuedo_max<T>(a: T, b: T) -> T
where
T: PartialOrd
{
    if a > b{
        a
    }else{
        b
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::iter::FromIterator;
    #[test]
    fn test() {
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        let res = String::from_iter(make_random_string(5, &"abc".chars().collect(), &mut rng));
        assert_eq!(res, "bacbb");
        let res = String::from_iter(make_random_string(5, &"abc".chars().collect(), &mut rng));
        assert_eq!(res, "bbbac");
    }
    #[test]
    fn test_make_random_strings() {
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        let res : Vec<String> = make_random_strings(3, 10, &"abc".chars().collect(), &mut rng).iter()
            .map(|v| String::from_iter(v)).collect();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], "acbb");
        assert_eq!(res[1], "bacbb");
        assert_eq!(res[2], "");
    }
    #[test]
    fn test_find_diff_systems() {
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        let sys_a = |w: &Vec<char>|{
            if String::from_iter(w.iter()) == "aba"{
                return 1;
            }else{
                return 0;
            }};
        let sys_b = |_: &Vec<char>|{return 0};
        let alph = vec!['a', 'b'];
        assert_eq!(find_diff_systems(&sys_a, &sys_a, &alph, 100, 4, 1e-5, &mut rng), None);
        assert_eq!(find_diff_systems(&sys_b, &sys_b, &alph, 100, 4, 1e-5, &mut rng), None);
        assert_eq!(find_diff_systems(&sys_a, &sys_b, &alph, 100, 4, 1e-5, &mut rng), Some((vec!['a', 'b', 'a'], 1, 0)));
    }
    #[test]
    fn test_join_vec() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5];
        let res = join_vec(&a, &b);
        assert_eq!(res, vec![1, 2, 3, 4, 5]);
    }
}
