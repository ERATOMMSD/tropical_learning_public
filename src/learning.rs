use crate::matrix::{MatrixBehavior, VectorBehavior};
use crate::util::join_vec;
use crate::wfa::WFA;
use crate::Character;
use crate::SolvableSemiring;
use bimap::BiMap;
use std::cmp::{max, min};
use std::iter::Iterator;
use std::time::{Duration, SystemTime};
use crate::stopwatch::Stopwatch;
use chrono::prelude::*;
use std::thread;
use std::fs::File;
use std::io::prelude::*;
use std::collections::HashMap;
#[derive(Debug)]
pub struct SubData<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    pub systime: SystemTime,
    pub ls: Option::<LearningState<A, S>>,
    pub warnings: Vec<String>,
    pub eqq_counterexamples: Vec<Vec<A>>,
    pub num_construction: usize,
    pub sw_enclose_row: Stopwatch,
    pub sw_enclose_column: Stopwatch,
    pub sw_construction: Stopwatch,
    pub constructed_at: Vec<u64>,
    pub autom_save_handles: Vec<Option<thread::JoinHandle<()>>>
}
impl<A, S> SubData<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    pub fn new() -> Self{
        return SubData::<A, S>{
            systime: SystemTime::now(),
            ls: None,
            warnings: vec![],
            eqq_counterexamples: vec![],
            num_construction: 0,
            sw_enclose_row: Stopwatch::new(),
            sw_enclose_column: Stopwatch::new(),
            sw_construction: Stopwatch::new(),
            constructed_at: Vec::<u64>::new(),
            autom_save_handles: Vec::new()
        };
    }
    fn check_timeout(&self, opt: &LearningOption) -> Result<(), ExtractionError<A, S>>{
        if let Some(to) = opt.timeout{
            if self.systime.elapsed().unwrap() > to{
                return Err(ExtractionError::TimedOutInside);
            }
        }
        return Ok(());
    }
    pub fn stop_all_sw(&mut self){
        self.sw_enclose_row.stop();
        self.sw_enclose_column.stop();
        self.sw_construction.stop();
    }
    fn process_constructed_wfa(&mut self, wfa: &WFA<A, S>){
        let filename = format!("result/wfa{}.json", self.num_construction);
        let serialized = serde_json::to_string(&wfa).unwrap();
        let handle = thread::spawn(move ||{
            let mut f = File::create(filename).expect("cannot open result.json");
            write!(f, "{}", serialized).expect("failed to write in result.json");
        });
        self.autom_save_handles.push(Some(handle));
        self.num_construction += 1;
        self.constructed_at.push(self.systime.elapsed().unwrap().as_secs());
    }
    pub fn wait_for_writings(&mut self){
        for a in &mut self.autom_save_handles{
            match std::mem::replace(a, None){
                Some(a) => a.join().unwrap(),
                None => ()
            }
        }
    }
}
#[derive(Clone)]
pub struct LearningState<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    pub pres: BiMap<Vec<A>, usize>,
    pub sufs: BiMap<Vec<A>, usize>,
    pub table: Vec<Vec<S>>,
    pub alphabet: Vec<A>,
}
impl<A, S> std::fmt::Debug for LearningState<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "hoge")?;
        writeln!(f, "pres: {:?}", self.pres)?;
        writeln!(f, "sufs: {:?}", self.sufs)?;
        writeln!(f, "---table below---")?;
        writeln!(f, "{}", self.table.print())?;
        writeln!(f, "---table above---")?;
        return Ok(());
    }
}
#[derive(Debug)]
pub enum ExtractionError<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    InfiniteEnclosing,
    LimitExceeded(crate::wfa::WFA<A, S>),
    TimedOut((Option::<crate::wfa::WFA<A, S>>, SubData<A, S>)),
    TimedOutInside,
}
pub enum ExtractionStrategy {
    Naive,
    MaxConfig,
    ColumnBased,
}
pub struct DetectingRepeatEncloseOption {
    threshold: usize,
    cache_size: usize,
    window_size: usize, 
}
pub enum AddWordMode {
    Row,
    Column
}
pub struct LearningOption {
    pub iter_limit: Option<usize>,
    pub reduce_rows: bool,
    pub add_column_when_unsolvable: bool,
    pub enclose_row: bool,
    pub enclose_column: bool,
    pub consistensify: bool,
    pub extraction_strategy: ExtractionStrategy,
    pub add_word_mode: AddWordMode,
    pub detect_repeat_enclose: Option<DetectingRepeatEncloseOption>,
    pub timeout: Option<Duration>,
    pub solving_tol: Option<f64>
}
impl Default for LearningOption {
    fn default() -> Self {
        Self {
            iter_limit: None,
            reduce_rows: false,
            add_column_when_unsolvable: false,
            enclose_row: true,
            enclose_column: false,
            consistensify: false,
            extraction_strategy: ExtractionStrategy::Naive,
            detect_repeat_enclose: None,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: None
        }
    }
}
fn init<A, S>(
    pres: &Vec<Vec<A>>,
    sufs: &Vec<Vec<A>>,
    memq: &dyn Fn(&Vec<A>) -> S,
    alphabet: &Vec<A>,
) -> LearningState<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut pres2 = BiMap::<Vec<A>, usize>::new();
    for i in 0..pres.len() {
        pres2.insert_no_overwrite(pres[i].clone(), i).unwrap();
    }
    let mut sufs2 = BiMap::<Vec<A>, usize>::new();
    for i in 0..sufs.len() {
        sufs2.insert_no_overwrite(sufs[i].clone(), i).unwrap();
    }
    let table2: Vec<Vec<S>> = pres
        .iter()
        .map({
            |row| {
                sufs.iter()
                    .map(move |col| memq(&join_vec(row, col)))
                    .collect()
            }
        })
        .collect();
    return LearningState {
        pres: pres2,
        sufs: sufs2,
        table: table2,
        alphabet: alphabet.clone(),
    };
}
fn get_row_from_learning_state<A, S>(
    ls: &LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    pre: &Vec<A>,
) -> Vec<S>
where
    A: Character,
    S: SolvableSemiring,
{
    (0..ls.sufs.len())
        .map(|id_suf| memq(&join_vec(&pre, &ls.sufs.get_by_right(&id_suf).unwrap())))
        .collect()
}
fn get_column_from_learning_state<A, S>(
    ls: &LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    suf: &Vec<A>,
) -> Vec<S>
where
    A: Character,
    S: SolvableSemiring,
{
    (0..ls.pres.len())
        .map(|id_pre| memq(&join_vec(&ls.pres.get_by_right(&id_pre).unwrap(), &suf)))
        .collect()
}
fn remove_one_unnecessary_row<A, S>(ls: &mut LearningState<A, S>, tol: Option<f64>) -> Option<(Vec<A>, Vec<S>)>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut unnec_id: Option<usize> = None;
    for row_id in 0..ls.pres.len() {
        let sol = SolvableSemiring::solve_dropping_row(&ls.table, &ls.table[row_id], Some(row_id), tol);
        if let Some(_) = sol {
            unnec_id = Some(row_id);
            break;
        }
    }
    if let Some(unnec_id) = unnec_id {
        let last_id = ls.pres.len() - 1;
        let removed_word = ls.pres.remove_by_right(&unnec_id).unwrap().0;
        let removed_row = ls.table.remove(unnec_id);
        for i in (unnec_id + 1)..(last_id + 1) {
            let word = ls.pres.get_by_right(&i).unwrap().clone();
            ls.pres.remove_by_right(&i);
            ls.pres.insert_no_overwrite(word.clone(), i - 1).unwrap();
        }
        return Some((removed_word, removed_row));
    } else {
        return None;
    }
}
fn remove_unnecessary_rows<A, S>(ls: &mut LearningState<A, S>, tol: Option<f64>)
where
    A: Character,
    S: SolvableSemiring,
{
    loop {
        let res = remove_one_unnecessary_row(ls, tol);
        match res {
            None => {
                break;
            }
            Some(_) => {}
        }
    }
}
fn enclose<A, S>(
    ls: &mut LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
    subdata: &mut SubData<A, S>
) -> Result<bool, ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    if !opt.enclose_row {
        return Ok(false);
    }
    subdata.sw_enclose_row.start();
    let mut modified = false;
    loop {
        let mut checking = 0;
        let mut adding: Option<(Vec<A>, Vec<S>)> = None;
        for id_pre in checking..ls.pres.len() {
            let pre = ls.pres.get_by_right(&id_pre).unwrap();
            for c in &ls.alphabet {
                let next_pre = join_vec(pre, &vec![*c]);
                if ls.pres.contains_left(&next_pre) {
                    continue;
                }
                let next_row: Vec<S> = get_row_from_learning_state(ls, memq, &next_pre);
                match SolvableSemiring::solve(&ls.table, &next_row, opt.solving_tol) {
                    None => {
                        adding = Some((next_pre, next_row));
                        break;
                    }
                    Some(_x) => {
                    }
                }
            }
            checking += 1;
        }
        match adding {
            None => break,
            Some((next_pre, next_row)) => {
                modified = true;
                for c in &next_pre {
                    print!("{}", c);
                }
                println!("! added");
                ls.pres
                    .insert_no_overwrite(next_pre.to_vec(), ls.pres.len())
                    .unwrap();
                ls.table.push(next_row.clone());
            }
        }
        subdata.check_timeout(opt)?;
    }
    subdata.sw_enclose_row.stop();
    return Ok(modified);
}
fn enclose_column<A, S>(
    ls: &mut LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
    subdata: &mut SubData<A, S>
) -> Result<bool, ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    if !opt.enclose_column {
        return Ok(false);
    }
    subdata.sw_enclose_column.start();
    let mut modified = false;
    loop {
        let mut checking = 0;
        let mut adding: Option<(Vec<A>, Vec<S>)> = None;
        for id_suf in checking..ls.sufs.len() {
            let suf = ls.sufs.get_by_right(&id_suf).unwrap();
            for c in &ls.alphabet {
                let next_suf = join_vec(&vec![*c], suf);
                if ls.sufs.contains_left(&next_suf) {
                    continue;
                }
                let next_col: Vec<S> = get_column_from_learning_state(ls, memq, &next_suf);
                match SolvableSemiring::solve(&ls.table.transpose(), &next_col, opt.solving_tol) {
                    None => {
                        adding = Some((next_suf, next_col));
                        break;
                    }
                    Some(_x) => {
                    }
                }
            }
            checking += 1;
        }
        match adding {
            None => break,
            Some((next_suf, next_col)) => {
                modified = true;
                for c in &next_suf {
                    print!("{}", c);
                }
                println!("!!");
                print!("{}", next_col.print());
                ls.sufs
                    .insert_no_overwrite(next_suf.to_vec(), ls.sufs.len())
                    .unwrap();
                for rid in 0..ls.pres.len() {
                    ls.table[rid].push(memq(&join_vec(
                        &ls.pres.get_by_right(&rid).unwrap(),
                        &next_suf,
                    )));
                }
            }
        }
        subdata.check_timeout(opt)?;
    }
    subdata.sw_enclose_column.stop();
    return Ok(modified);
}
fn add_column_when_unsolvable<A, S>(
    ls: &mut LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
) -> bool
where
    A: Character,
    S: SolvableSemiring,
{
    if !opt.add_column_when_unsolvable {
        return false;
    }
    for a in ls.alphabet.iter() {
        dbg!(a);
        let table_next: Vec<Vec<S>> = (0..ls.pres.len())
            .map(|rid| {
                get_row_by_prefix(
                    ls,
                    &join_vec(ls.pres.get_by_right(&rid).unwrap(), &vec![*a]),
                    memq,
                )
            })
            .collect();
        let sol = SolvableSemiring::solve_axb(&ls.table, &table_next, opt.solving_tol);
        match sol{
            None => {
                dbg!(a);
                let sufs_num_old = ls.sufs.len();
                let mut num_added = 0;
                let x: Vec<Vec<A>> = ls.sufs.left_values().cloned().collect();
                for (_, suf) in x.iter().enumerate() {
                    let new_suf = join_vec(&vec![*a], &suf);
                    if !ls.sufs.contains_left(&new_suf) {
                        ls.sufs
                            .insert_no_overwrite(new_suf, num_added + sufs_num_old)
                            .unwrap();
                        num_added += 1;
                    }
                }
                let sufs_num_new = sufs_num_old + num_added;
                dbg!(&num_added);
                for sid in sufs_num_old..sufs_num_new {
                    dbg!(&sid);
                    dbg!(&ls.sufs);
                    let suf = ls.sufs.get_by_right(&sid).unwrap();
                    for (pre, pid) in &ls.pres {
                        ls.table[*pid].push(memq(&join_vec(pre, &suf)));
                    }
                }
                return true;
            }
            _ => {}
        }
    }
    return false;
}
fn get_row_by_prefix<A, S>(
    ls: &LearningState<A, S>,
    pre: &Vec<A>,
    memq: &dyn Fn(&Vec<A>) -> S,
) -> Vec<S>
where
    A: Character,
    S: SolvableSemiring,
{
    (0..ls.sufs.len())
        .map(|cid| {
            let w = join_vec(pre, &ls.sufs.get_by_right(&cid).unwrap());
            memq(&w)
        })
        .collect()
}
fn consistensify<A, S>(
    ls: &mut LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
) -> bool
where
    A: Character,
    S: SolvableSemiring,
{
    dbg!("started consistensify");
    if !opt.consistensify {
        return false;
    }
    if ls.pres.len() <= 1 {
        return false;
    }
    let mut modified = false;
    for row_id in 0..ls.pres.len() {
        let sol = SolvableSemiring::solve_dropping_row(&ls.table, &ls.table[row_id], Some(row_id), opt.solving_tol);
        if let Some(coef) = sol {
            for a in &ls.alphabet {
                let next_table: Vec<Vec<S>> = (0..ls.pres.len())
                    .map(|rid| {
                        let next_word = join_vec(&ls.pres.get_by_right(&rid).unwrap(), &vec![*a]);
                        get_row_by_prefix(ls, &next_word, memq)
                    })
                    .collect();
                let next_row = next_table[row_id].clone();
                let left = next_row;
                let right = next_table.mult_vec_left(&coef);
                if S::max_dist(&left, &right) > opt.solving_tol.unwrap_or(0.0) {
                    modified = true;
                    for i in 0..left.len() {
                        if S::dist(&left[i], &right[i]) > opt.solving_tol.unwrap_or(0.0) {
                            let sufs_num_old = ls.sufs.len();
                            let suf = ls.sufs.get_by_right(&i).unwrap().clone();
                            let new_suf = join_vec(&vec![*a], &suf);
                            if !ls.sufs.contains_left(&new_suf) {
                                ls.sufs
                                    .insert_no_overwrite(new_suf.clone(), sufs_num_old)
                                    .unwrap();
                                for (pre, pid) in &ls.pres {
                                    ls.table[*pid].push(memq(&join_vec(pre, &new_suf)));
                                }
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    return modified;
}
fn add_columns<A, S>(ls: &mut LearningState<A, S>, suf: &Vec<A>, memq: &dyn Fn(&Vec<A>) -> S, opt: &LearningOption, subdata: &mut SubData<A, S>) -> Result<(), ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    let num_sufs_old = ls.sufs.len();
    for i in (0..(suf.len() + 1)).rev() {
        let suf: Vec<A> = suf[(suf.len() - i)..suf.len()].iter().cloned().collect();
        if ls.sufs.contains_left(&suf) {
            break;
        } else {
            ls.sufs.insert_no_overwrite(suf, ls.sufs.len()).unwrap();
        }
    }
    let num_sufs_new = ls.sufs.len();
    for i_pre in 0..ls.pres.len() {
        for i_suf in num_sufs_old..num_sufs_new {
            ls.table[i_pre].push(memq(&join_vec(
                &ls.pres.get_by_right(&i_pre).unwrap(),
                &ls.sufs.get_by_right(&i_suf).unwrap(),
            )));
            subdata.check_timeout(opt)?;
        }
        assert_eq!(ls.table[i_pre].len(), ls.sufs.len());
    }
    return Ok(());
}
fn add_rows<A, S>(ls: &mut LearningState<A, S>, pre: &Vec<A>, memq: &dyn Fn(&Vec<A>) -> S, opt: &LearningOption, subdata: &mut SubData<A, S>)
-> Result<(), ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    let num_pres_old = ls.pres.len();
    for i in (0..(pre.len() + 1)).rev() {
        let pre: Vec<A> = pre[0..i].iter().cloned().collect();
        if ls.pres.contains_left(&pre) {
            break;
        } else {
            ls.pres.insert_no_overwrite(pre, ls.pres.len()).unwrap();
        }
    }
    let num_pres_new = ls.pres.len();
    for i_pre in num_pres_old..num_pres_new {
        ls.table.push(
            (0..ls.sufs.len())
                .map(|i_suf| {
                    memq(&join_vec(
                        &ls.pres.get_by_right(&i_pre).unwrap(),
                        &ls.sufs.get_by_right(&i_suf).unwrap(),
                    ))
                })
                .collect(),
        );
        subdata.check_timeout(opt)?;
    }
    return Ok(());
}
fn add_word_pre<A, S>(ls: &mut LearningState<A, S>, word: &Vec<A>, memq: &dyn Fn(&Vec<A>) -> S, opt: &LearningOption, subdata: &mut SubData<A, S>)
-> Result<(), ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut ok = 0;
    let mut ng = word.len() + 1;
    while ng - ok > 1 {
        let mut piv = (ok + ng) / 2;
        piv = max(0, piv);
        piv = min(word.len(), piv);
        if ls
            .pres
            .contains_left(&word[0..piv].iter().cloned().collect())
        {
            ok = piv;
        } else {
            ng = piv;
        }
    }
    if ok == word.len() {
        return Ok(());
    }
    let word_s: Vec<A> = word[(ok + 1)..word.len()].iter().cloned().collect();
    if ls.sufs.contains_left(&word_s) {
        return Ok(());
    }
    add_columns(ls, &word_s, &memq, opt, subdata)?;
    return Ok(());
}
fn add_word_suf<A, S>(ls: &mut LearningState<A, S>, word: &Vec<A>, memq: &dyn Fn(&Vec<A>) -> S, opt: &LearningOption, subdata: &mut SubData<A, S>)
-> Result<(), ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut ok = 0;
    let mut ng = word.len() + 1;
    while ng - ok > 1 {
        let mut piv = (ok + ng) / 2;
        piv = max(0, piv);
        piv = min(word.len(), piv);
        if ls.sufs.contains_left(
            &word[(word.len() - piv)..word.len()]
                .iter()
                .cloned()
                .collect(),
        ) {
            ok = piv;
        } else {
            ng = piv;
        }
    }
    if ok == word.len() {
        return Ok(());
    }
    let word_p: Vec<A> = word[0..(word.len() - ok)].iter().cloned().collect();
    if ls.pres.contains_left(&word_p) {
        return Ok(());
    }
    add_rows(ls, &word_p, &memq, opt, subdata)?;
    return Ok(());
}
fn add_word<A, S>(ls: &mut LearningState<A, S>, word: &Vec<A>, memq: &dyn Fn(&Vec<A>) -> S, opt: &LearningOption, subdata: &mut SubData<A, S>)
-> Result<(), ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    match &opt.add_word_mode{
        AddWordMode::Row => add_word_pre(ls, word, memq, opt, subdata)?,
        AddWordMode::Column => add_word_suf(ls, word, memq, opt, subdata)?
    }   
    return Ok(());
}
fn extract_wfa_naive<A, S>(
    ls: &LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
) -> crate::wfa::WFA<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut ls = ls.clone();
    let bigeps = Some(f64::INFINITY);  
    if opt.reduce_rows {
        remove_unnecessary_rows(&mut ls, opt.solving_tol);
    }
    let dim = ls.pres.len();
    let row_eps: Vec<S> = (0..ls.sufs.len())
        .map(|col_id| memq(ls.sufs.get_by_right(&col_id).unwrap()))
        .collect();
    dbg!("table");
    println!("{}", &ls.table.print());
    let alpha = S::solve(&ls.table, &row_eps, bigeps);
    if let None = alpha {
        dbg!("row_eps");
        println!("{}", row_eps.print());
        panic!();
    }
    let alpha = alpha.unwrap();
    let beta: Vec<S> = ls
        .table
        .iter()
        .map(|r| r[*ls.sufs.get_by_left(&vec![]).unwrap()].clone())
        .collect();
    let mut trans = HashMap::<A, crate::matrix::Matrix<S>>::new();
    for a in ls.alphabet.iter() {
        let mut trans_current: Vec<Vec<S>> = Vec::<Vec<S>>::new();
        for row_id in 0..dim {
            let row_word = ls.pres.get_by_right(&row_id).unwrap();
            let row_next: Vec<S> = (0..ls.sufs.len())
                .map(|col_id| {
                    memq(&join_vec(
                        &join_vec(&row_word, &vec![*a]),
                        ls.sufs.get_by_right(&col_id).unwrap(),
                    ))
                })
                .collect();
            let sol = S::solve(&ls.table, &row_next, bigeps)
                .expect("This has to be solvable.  If not, it is not complete.");
            trans_current.push(sol);
        }
        trans.insert(*a, trans_current);
    }
    return crate::wfa::WFA {
        ini: alpha,
        fin: beta,
        trans: trans,
    };
}
fn extract_wfa_column_based<A, S>(
    ls: &LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
) -> crate::wfa::WFA<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut ls = ls.clone();
    if opt.reduce_rows {
        remove_unnecessary_rows(&mut ls, opt.solving_tol);
    }
    let dim = ls.sufs.len();
    let alpha = ls.table[*ls.pres.get_by_left(&vec![]).unwrap()].clone();
    let beta: Vec<S> = (0..dim)
        .map(|cid| {
            if *ls.sufs.get_by_right(&cid).unwrap() == vec![] {
                S::one()
            } else {
                S::zero()
            }
        })
        .collect();
    let mut trans = HashMap::<A, crate::matrix::Matrix<S>>::new();
    for a in ls.alphabet.iter() {
        dbg!(&a);
        let mut trans_next = Vec::<Vec<S>>::new();
        for row_id in 0..ls.pres.len() {
            let row_word = ls.pres.get_by_right(&row_id).unwrap();
            let row_next: Vec<S> = (0..ls.sufs.len())
                .map(|cid| {
                    memq(&join_vec(
                        &join_vec(&row_word, &vec![*a]),
                        ls.sufs.get_by_right(&cid).unwrap(),
                    ))
                })
                .collect();
            trans_next.push(row_next);
        }
        trans.insert(
            *a,
            SolvableSemiring::solve_axb(&ls.table, &trans_next, opt.solving_tol).unwrap(),
        );
    }
    return crate::wfa::WFA {
        ini: alpha,
        fin: beta,
        trans: trans,
    };
}
fn extract_wfa_maximal<A, S>(
    ls: &LearningState<A, S>,   
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
) -> crate::wfa::WFA<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut ls = ls.clone();
    if opt.reduce_rows {
        remove_unnecessary_rows(&mut ls, opt.solving_tol);
        dbg!(&ls);
    }
    let dim = ls.pres.len();
    let max_conf: Vec<Vec<S>> = ls
        .table
        .iter()
        .map(|r| SolvableSemiring::solve(&ls.table, r, opt.solving_tol).unwrap())
        .collect();
    let get_max_conf_from_pre = |p: Vec<A>| {
        let row_p = (0..ls.sufs.len())
            .map(|col_id| memq(&join_vec(&p, &ls.sufs.get_by_right(&col_id).unwrap())))
            .collect();
        let sol = SolvableSemiring::solve(&ls.table, &row_p, opt.solving_tol)
            .expect("This has to be solvable.  If not, it is not complete.");
        sol
    };
    let alpha = get_max_conf_from_pre(vec![]);
    let final_values: Vec<S> = ls
        .table
        .iter()
        .map(|r| r[*ls.sufs.get_by_left(&vec![]).unwrap()].clone())
        .collect();
    let beta = SolvableSemiring::solve(&max_conf.transpose(), &final_values, opt.solving_tol).unwrap();
    let mut trans = HashMap::<A, crate::matrix::Matrix<S>>::new();
    for a in ls.alphabet.iter() {
        dbg!(&a);
        let mut trans_next = Vec::<Vec<S>>::new();
        for row_id in 0..dim {
            let row_word = ls.pres.get_by_right(&row_id).unwrap();
            let row_next = get_max_conf_from_pre(join_vec(&row_word, &vec![*a]));
            trans_next.push(row_next);
        }
        trans.insert(
            *a,
            SolvableSemiring::solve_axb(&max_conf, &trans_next, opt.solving_tol).unwrap(),
        );
    }
    return crate::wfa::WFA {
        ini: alpha,
        fin: beta,
        trans: trans,
    };
}
fn extract_wfa<A, S>(
    ls: &LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
    subdata: &mut SubData<A, S>
) -> crate::wfa::WFA<A, S>
where
    A: Character,
    S: SolvableSemiring,
{
    subdata.sw_construction.start();
    let res = match opt.extraction_strategy {
        ExtractionStrategy::Naive => extract_wfa_naive(ls, memq, opt),
        ExtractionStrategy::MaxConfig => extract_wfa_maximal(ls, memq, opt),
        ExtractionStrategy::ColumnBased => extract_wfa_column_based(ls, memq, opt)
    };
    subdata.sw_construction.stop();
    return res;
}
fn extract_wfa_full<A, S>(
    ls: &mut LearningState<A, S>,
    memq: &dyn Fn(&Vec<A>) -> S,
    opt: &LearningOption,
    subdata: &mut SubData<A, S>
) -> Result<crate::wfa::WFA<A, S>, ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    loop {
        dbg!(&ls);
        let res_enclose = enclose(ls, &memq, opt, subdata)?;
        dbg!("ran enclose");
        let res_consistensify = consistensify(ls, &memq, opt);
        dbg!("ran consistency");
        let res_add_column_when_unsolvable = add_column_when_unsolvable(ls, &memq, opt);
        dbg!("ran add_column_when_unsolvable");
        let res_enclose_column = enclose_column(ls, &memq, opt, subdata)?;
        dbg!("ran enclose_column");
        dbg!((
            &res_enclose,
            &res_consistensify,
            &res_add_column_when_unsolvable,
            &res_enclose_column
        ));
        if !res_enclose
            && !res_consistensify
            && !res_add_column_when_unsolvable
            && !res_enclose_column
        {
            break;
        }
        subdata.check_timeout(opt)?;
    }
    let res = extract_wfa(ls, memq, opt, subdata);
    subdata.process_constructed_wfa(&res);
    Ok(res)
}
fn trap_timedout<A, S>(e: ExtractionError<A, S>, subdata: SubData<A, S>, extracted_last: Option<WFA<A, S>>) -> ExtractionError<A, S>
where
A: Character,
S: SolvableSemiring,
{
    if let ExtractionError::TimedOutInside = e{
        return ExtractionError::TimedOut((extracted_last, subdata));
    }
    return e;
}
pub fn learn<A, S>(
    alphabet: &Vec<A>,
    memq: &dyn Fn(&Vec<A>) -> S,
    eqq: &mut dyn FnMut(&crate::wfa::WFA<A, S>) -> Option<Vec<A>>,
    opt: &LearningOption
) -> Result<(WFA<A, S>, SubData<A, S>), ExtractionError<A, S>>
where
    A: Character,
    S: SolvableSemiring,
{
    let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, alphabet);
    let mut subdata = SubData::<A, S>::new();
    let mut cnt = 0;
    let mut extracted_last = None;
    loop {
        let extracted = extract_wfa_full(&mut ls, &memq, opt, &mut subdata);
        match extracted{
            Ok(extracted) => {
                let res = eqq(&extracted);
                match res {
                    None => {
                        subdata.ls = Some(ls.clone());
                        return Ok((extracted, subdata));
                    },
                    Some(ce) =>{ 
                        let res = add_word(&mut ls, &ce, &memq, opt, &mut subdata);
                        if let Err(e) = res{
                            return Err(trap_timedout(e, subdata, extracted_last));
                        }
                        if subdata.eqq_counterexamples.contains(&ce){
                            let s = ce.iter().map(|c|{format!("{}", c)}).collect::<Vec<String>>().join("");
                            subdata.warnings.push(format!("DoubleCE: a CE {} was given twice!", &s));
                        }
                        subdata.eqq_counterexamples.push(ce.clone());
                    },
                }
                cnt += 1;
                if let Some(n) = opt.iter_limit {
                    if cnt > n {
                        subdata.ls = Some(ls.clone());
                        return Err(ExtractionError::LimitExceeded(extracted));
                    }
                }
                extracted_last = Some(extracted);
            },
            Err(e) => {
                subdata.ls = Some(ls.clone());
                return Err(trap_timedout(e, subdata, extracted_last));
            }
        }
        if let Err(e) = subdata.check_timeout(opt){
            subdata.ls = Some(ls.clone());
            return Err(trap_timedout(e, subdata, extracted_last));
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::make_mp_mat;
    use crate::make_mp_vec;
    use crate::max_plus_semiring::MaxPlusNumSemiring;
    use crate::real_field::RealField;
    use crate::semiring::Semiring;
    use crate::wfa::WFA;
    use rand::Rng;
    use rand::SeedableRng;
    use std::collections::HashSet;
    #[test]
    fn test_enclose1() {
        let w = WFA {
            ini: make_mp_vec![i64, 1, 3, 4],
            fin: make_mp_vec![i64, 2, 1, 1],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 0, 0, 3; 0, 0, 3; 1, 0, 0],
                'b' => make_mp_mat![i64, 0, 1, 0; 2, 0, 0; 0, 0, 4]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(
            &vec!["".chars().collect()],
            &vec![vec![], vec!['a'], vec!['b']],
            &memq,
            &"ab".chars().collect(),
        );
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        assert_eq!(ls.table.len(), 1);
        assert_eq!(ls.pres.len(), 1);
        let i_empty = *ls.pres.get_by_left(&vec![]).unwrap();
        assert_eq!(ls.table[i_empty], make_mp_vec![i64, 5, 7, 9]);
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        assert_eq!(ls.table.len(), 1);
        assert_eq!(ls.pres.len(), 1);
        let i_empty = *ls.pres.get_by_left(&vec![]).unwrap();
        assert_eq!(ls.table[i_empty], make_mp_vec![i64, 5, 7, 9]);
    }
    #[test]
    fn test_enclose2() {
        let w = WFA {
            ini: make_mp_vec![i64, 6, 11, 1],
            fin: make_mp_vec![i64, 7, 0, 6],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 2, 3, 1; 2, 0, 9; 3, 0, 8],
                'b' => make_mp_mat![i64, 9, 6, 2; 10, 3, 2; 8, 5, 4]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(
            &vec!["".chars().collect()],
            &vec![vec![], vec!['a'], vec!['b']],
            &memq,
            &"ab".chars().collect(),
        );
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        assert_eq!(ls.table.len(), 4);
        assert_eq!(ls.pres.len(), 4);
        let i_empty = *ls.pres.get_by_left(&vec![]).unwrap();
        assert_eq!(ls.table[i_empty], make_mp_vec![i64, 13, 26, 28]);
        let i_a = *ls.pres.get_by_left(&vec!['a']).unwrap();
        assert_eq!(ls.table[i_a], make_mp_vec![i64, 26, 34, 35]);
        let i_ab = *ls.pres.get_by_left(&vec!['a', 'b']).unwrap();
        assert_eq!(ls.table[i_ab], make_mp_vec![i64, 35, 40, 44]);
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        assert_eq!(ls.table.len(), 4);
        assert_eq!(ls.pres.len(), 4);
        let i_empty = *ls.pres.get_by_left(&vec![]).unwrap();
        assert_eq!(ls.table[i_empty], make_mp_vec![i64, 13, 26, 28]);
        let i_a = *ls.pres.get_by_left(&vec!['a']).unwrap();
        assert_eq!(ls.table[i_a], make_mp_vec![i64, 26, 34, 35]);
        let i_ab = *ls.pres.get_by_left(&vec!['a', 'b']).unwrap();
        assert_eq!(ls.table[i_ab], make_mp_vec![i64, 35, 40, 44]);
    }
    #[test]
    fn test_add_word() {
        let w = WFA {
            ini: make_mp_vec![i64, 6, 11, 1],
            fin: make_mp_vec![i64, 7, 0, 6],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 2, 3, 1; 2, 0, 9; 3, 0, 8],
                'b' => make_mp_mat![i64, 9, 6, 2; 10, 3, 2; 8, 5, 4]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &"ab".chars().collect());
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        let pre_result = ls
            .pres
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let pre_expect = [vec![]].iter().cloned().collect::<HashSet<Vec<char>>>();
        assert_eq!(pre_result, pre_expect);
        let suf_result = ls
            .sufs
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let suf_expect = [vec![]].iter().cloned().collect::<HashSet<Vec<char>>>();
        assert_eq!(suf_result, suf_expect);
        add_word_pre(&mut ls, &"aa".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        let pre_result = ls
            .pres
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let pre_expect = [vec![]].iter().cloned().collect::<HashSet<Vec<char>>>();
        assert_eq!(pre_result, pre_expect);
        let suf_result = ls
            .sufs
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let suf_expect = [vec![], vec!['a']]
            .iter()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        assert_eq!(suf_result, suf_expect);
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        let pre_result = ls
            .pres
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let pre_expect = [vec![], vec!['a'], vec!['a', 'b'], vec!['b']]
            .iter()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        assert_eq!(pre_result, pre_expect);
        let suf_result = ls
            .sufs
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let suf_expect = [vec!['a'], vec![]]
            .iter()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        assert_eq!(suf_result, suf_expect);
        add_word_pre(&mut ls, &"abbababa".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        let pre_result = ls
            .pres
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let pre_expect = [vec![], vec!['a'], vec!['a', 'b'], vec!['b']]
            .iter()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        assert_eq!(pre_result, pre_expect);
        let suf_result = ls
            .sufs
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let suf_expect = [
            vec![],
            vec!['a'],
            vec!['a', 'b', 'a', 'b', 'a'],
            vec!['b', 'a', 'b', 'a'],
            vec!['a', 'b', 'a'],
            vec!['b', 'a'],
        ]
        .iter()
        .cloned()
        .collect::<HashSet<Vec<char>>>();
        assert_eq!(suf_result, suf_expect);
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        let pre_result = ls
            .pres
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let pre_expect = [vec![], vec!['a'], vec!['a', 'b'], vec!['b']]
            .iter()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        assert_eq!(pre_result, pre_expect);
        let suf_result = ls
            .sufs
            .left_values()
            .cloned()
            .collect::<HashSet<Vec<char>>>();
        let suf_expect = [
            vec![],
            vec!['a'],
            vec!['a', 'b', 'a', 'b', 'a'],
            vec!['b', 'a', 'b', 'a'],
            vec!['a', 'b', 'a'],
            vec!['b', 'a'],
        ]
        .iter()
        .cloned()
        .collect::<HashSet<Vec<char>>>();
        assert_eq!(suf_result, suf_expect);
    }
    #[test]
    fn test_extract_wfa1() {
        let w = WFA {
            ini: make_mp_vec![i64, 6, 11, 1],
            fin: make_mp_vec![i64, 7, 0, 6],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 2, 3, 1; 2, 0, 9; 3, 0, 8],
                'b' => make_mp_mat![i64, 9, 6, 2; 10, 3, 2; 8, 5, 4]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &"ab".chars().collect());
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"aa".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"abba".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"bababababab".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        dbg!(&ls);
        let extracted = extract_wfa(
            &ls,
            &memq,
            &LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: true,
                ..Default::default()
            },
            &mut subdata
        );
        let ef = |x: &Vec<char>| extracted.run(x.iter().cloned());
        dbg!(&extracted);
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let d =
            crate::util::find_diff_systems(&memq, &ef, &"ab".chars().collect(), 1000, 20, 1e-5, &mut rng);
        assert_eq!(d, None);
    }
    #[test]
    fn test_extract_wfa2() {
        let w = WFA {
            ini: make_mp_vec![i64, 1, 0],
            fin: make_mp_vec![i64, 0, 2],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 2, 3; 3, 0],
                'b' => make_mp_mat![i64, 2, 3; -1, 2]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &"ab".chars().collect());
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"aa".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"abbaba".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        dbg!(&ls);
        let extracted = extract_wfa(
            &ls,
            &memq,
            &LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: true,
                ..Default::default()
            },
            &mut subdata
        );
        let ef = |x: &Vec<char>| extracted.run(x.iter().cloned());
        dbg!(&extracted);
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let d =
            crate::util::find_diff_systems(&memq, &ef, &"ab".chars().collect(), 1000, 20, 1e-5, &mut rng);
        assert_eq!(d, None);
    }
    #[test]
    fn test_extract_wfa3() {
        let w = WFA {
            ini: make_mp_vec![i64, 6, 11, 1],
            fin: make_mp_vec![i64, 7, 0, 6],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 2, 3, 1; 2, 0, 9; 3, 0, 8],
                'b' => make_mp_mat![i64, 9, 6, 2; 10, 3, 2; 8, 5, 4]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &"ab".chars().collect());
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"aa".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"abba".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        add_word_pre(&mut ls, &"abbbbabb".chars().collect(), &memq, &LearningOption::default(), &mut subdata).unwrap();
        enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
        dbg!(&ls);
        let extracted = extract_wfa(
            &ls,
            &memq,
            &LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: true,
                ..Default::default()
            },
            &mut subdata
        );
        let ef = |x: &Vec<char>| extracted.run(x.iter().cloned());
        dbg!(&extracted);
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let d =
            crate::util::find_diff_systems(&memq, &ef, &"ab".chars().collect(), 1000, 20, 1e-5, &mut rng);
        assert_eq!(d, None);
    }
    fn make_random_mpval(rng: &mut dyn rand::RngCore) -> MaxPlusNumSemiring<i64> {
        if rng.gen_ratio(1, 10) {
            MaxPlusNumSemiring::<i64>::NegInf
        } else {
            MaxPlusNumSemiring::<i64>::Raw(rng.gen_range(-5, 5 + 1))
        }
    }
    fn check_sanity<S>(ls: &LearningState<char, S>, wfa_orig: &WFA<char, S>, wfa_ext: &WFA<char, S>)
    where S: SolvableSemiring{
        let mut words = HashSet::<Vec<char>>::new();
        for p in ls.pres.left_values() {
            for s in ls.sufs.left_values() {
                words.insert(join_vec(p, s));
            }
        }
        for w in words{
            let s: String = w.iter().cloned().collect();
            println!("san check: {}", s);
            let exp = wfa_orig.run(w.iter().cloned());
            let res = wfa_ext.run(w.iter().cloned());
            assert!(S::dist(&exp, &res) < 1e-5);
        }
    }
    fn check_sanity_real(ls: &LearningState<char, RealField>, wfa_orig: &WFA<char, RealField>, wfa_ext: &WFA<char, RealField>, eps: f64)
    {
        let mut words = HashSet::<Vec<char>>::new();
        for p in ls.pres.left_values() {
            for s in ls.sufs.left_values() {
                words.insert(join_vec(p, s));
            }
        }
        for w in words{
            let s: String = w.iter().cloned().collect();
            let exp = wfa_orig.run(w.iter().cloned());
            let res = wfa_ext.run(w.iter().cloned());
            println!("san check: {}, {}, {}", s, &exp, &res);
            if (exp.unwrap() - res.unwrap()).abs() > eps{
                panic!();
            }
        }
    }
    #[test]
    fn test_extract_wfa_random1x() {
        let max_iteration = 5000;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let mut detected_inf_encl = 0;
        'main: for _ in 0..10 {
            let d = rng.gen_range(2, 10 + 1);
            let alph_size = rng.gen_range(2, 4 + 1);
            let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
            let ini: Vec<MaxPlusNumSemiring<i64>> =
                (0..d).map(|_| make_random_mpval(&mut rng)).collect();
            let fin: Vec<MaxPlusNumSemiring<i64>> =
                (0..d).map(|_| make_random_mpval(&mut rng)).collect();
            let mut trans = HashMap::<char, crate::matrix::Matrix<MaxPlusNumSemiring<i64>>>::new();
            for a in &alph {
                let m: crate::matrix::Matrix<MaxPlusNumSemiring<i64>> = (0..d)
                    .map(|_| (0..d).map(|_| make_random_mpval(&mut rng)).collect())
                    .collect();
                trans.insert(*a, m);
            }
            let wfa_random = crate::wfa::WFA { ini, fin, trans };
            dbg!(&wfa_random);
            let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
            let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
                crate::util::find_diff_systems(
                    &memq,
                    &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                    &alph,
                    40000,
                    20,
                    1e-5,
                    &mut rng,
                )
            };
            let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
            let opt = LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: false,
                enclose_row: true,
                enclose_column: true,
                detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                    cache_size: alph_size * 10,
                    window_size: alph_size,
                    threshold: 10,
                }),
                extraction_strategy: ExtractionStrategy::Naive,
                add_word_mode: AddWordMode::Row,
                ..Default::default()
            };
            let mut subdata = SubData::new();
            let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
            let mut timeout = true;
            for _ in 0..max_iteration {
                let eqq_res = eqq(&extracted_wfa);
                match eqq_res {
                    None => {
                        timeout = false;
                        break;
                    }
                    Some((ce, _, _)) => {
                        add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                        let res = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata);
                        match res {
                            Ok(a) => {
                                extracted_wfa = a;
                            }
                            Err(e) => match e {
                                ExtractionError::InfiniteEnclosing => {
                                    dbg!("Detected InfiniteEnclosing!");
                                    detected_inf_encl += 1;
                                    continue 'main;
                                }
                                ExtractionError::LimitExceeded(_) => panic!(),
                                ExtractionError::TimedOut(_) => panic!(),
                                ExtractionError::TimedOutInside => panic!()
                            },
                        }
                    }
                }
            }
            dbg!(&ls.pres.len(), &ls.sufs.len());
            if timeout {
                dbg!(ls);
                dbg!(extracted_wfa);
                panic!();
            }
            let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
                crate::util::find_diff_systems(
                    &memq,
                    &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                    &alph,
                    100000,
                    20,
                    1e-5,
                    &mut rng,
                )
            };
            let res = eqq(&extracted_wfa);
            if let Some(ce) = &res {
                let mut words = HashSet::<Vec<char>>::new();
                for p in ls.pres.left_values() {
                    for s in ls.sufs.left_values() {
                        words.insert(join_vec(p, s));
                    }
                }
                if words.contains(&ce.0) {
                    dbg!(&wfa_random);
                    assert_eq!(res, None);
                }
            }
            check_sanity(&ls, &wfa_random, &extracted_wfa);
        }
        assert_eq!(detected_inf_encl, 0);
    }
    #[test]
    fn test_extract_wfa_random2() {
        let max_iteration = 10000;
        let mut detected_inf_encl = 0;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        'main: for _ in 0..10 {
            let d = rng.gen_range(2, 5 + 1);
            let alph_size = rng.gen_range(2, 4 + 1);
            let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
            let opt = LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: false,
                enclose_row: true,  
                enclose_column: true,
                detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                    cache_size: alph_size * 10,
                    window_size: alph_size,
                    threshold: 10,
                }),
                extraction_strategy: ExtractionStrategy::Naive,
                add_word_mode: AddWordMode::Row,
                ..Default::default()
            };
            let ini: Vec<MaxPlusNumSemiring<i64>> =
                (0..d).map(|_| make_random_mpval(&mut rng)).collect();
            let fin: Vec<MaxPlusNumSemiring<i64>> =
                (0..d).map(|_| make_random_mpval(&mut rng)).collect();
            let mut trans = HashMap::<char, crate::matrix::Matrix<MaxPlusNumSemiring<i64>>>::new();
            for a in &alph {
                let m: crate::matrix::Matrix<MaxPlusNumSemiring<i64>> = (0..d)
                    .map(|_| (0..d).map(|_| make_random_mpval(&mut rng)).collect())
                    .collect();
                trans.insert(*a, m);
            }
            let wfa_random = crate::wfa::WFA { ini, fin, trans };
            dbg!(&wfa_random);
            let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
            let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
                crate::util::find_diff_systems(
                    &memq,
                    &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                    &alph,
                    100000,
                    15,
                    1e-5,
                    &mut rng,
                )
            };
            let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
            let mut subdata = SubData::new();
            let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
            let mut timeout = true;
            for _ in 0..max_iteration {
                let eqq_res = eqq(&extracted_wfa);
                match eqq_res {
                    None => {
                        timeout = false;
                        break;
                    }
                    Some((ce, x, y)) => {
                        dbg!((&ce, &x, &y));
                        dbg!(&wfa_random);
                        dbg!(&extracted_wfa);
                        add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                        let res = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata);
                        match res {
                            Ok(a) => {
                                extracted_wfa = a;
                            }
                            Err(e) => match e {
                                ExtractionError::InfiniteEnclosing => {
                                    dbg!("Detected InfiniteEnclosing!");
                                    detected_inf_encl += 1;
                                    continue 'main;
                                }
                                ExtractionError::LimitExceeded(_) => panic!(),
                                ExtractionError::TimedOut(_) => panic!(),
                                ExtractionError::TimedOutInside => panic!()
                            },
                        }
                    }
                }
            }
            dbg!(&ls.pres.len(), &ls.sufs.len());
            if timeout {
                dbg!(ls);
                panic!();
            }
            let res = eqq(&extracted_wfa);
            if let Some(ce) = &res {
                let mut words = HashSet::<Vec<char>>::new();
                for p in ls.pres.left_values() {
                    for s in ls.sufs.left_values() {
                        words.insert(join_vec(p, s));
                    }
                }
                if words.contains(&ce.0) {
                    dbg!(&wfa_random);
                    assert_eq!(res, None);
                }
            }
            let mut words = HashSet::<Vec<char>>::new();
            for p in ls.pres.left_values() {
                for s in ls.sufs.left_values() {
                    words.insert(join_vec(p, s));
                }
            }
            for w in words{
                dbg!(&w);
                let exp = wfa_random.run(w.iter().cloned());
                let res = extracted_wfa.run(w.iter().cloned());
                assert_eq!(exp, res);
            }
            dbg!((extracted_wfa.ini.len(), d));
            check_sanity(&ls, &wfa_random, &extracted_wfa);
        }
        assert_eq!(detected_inf_encl, 0);
    }
    fn embed(v: i64) -> MaxPlusNumSemiring<i64> {
        MaxPlusNumSemiring::Raw(v)
    }
    fn eps() -> MaxPlusNumSemiring<i64> {
        MaxPlusNumSemiring::NegInf
    }
    #[test]
    fn test_extract_wfa_prob1() {
        let max_iteration = 1000;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let alph_size = 5;
        let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
        let ini = vec![eps(), embed(0), embed(-2), embed(-3), embed(-3)];
        let fin = vec![eps(), embed(-4), embed(5), eps(), embed(4)];
        let trans = map![
            'd' => vec![
                vec![eps(), embed(5), embed(-5), embed(-1), embed(2)],
                vec![eps(), embed(-1), embed(-4), embed(2), embed(-1)],
                vec![embed(3), embed(-4), eps(), embed(3), embed(5)],
                vec![embed(-4), embed(-2), eps(), embed(-1), embed(3)],
                vec![embed(-1), embed(4), embed(2), eps(), embed(-5)]
            ],
            'a' => vec![
                vec![embed(0), embed(2), eps(), embed(-5), embed(0)],
                vec![eps(), embed(5), embed(-5), eps(), embed(-5)],
                vec![eps(), embed(5), embed(3), embed(-2), eps()],
                vec![embed(0), eps(), embed(0), eps(), embed(-4)],
                vec![embed(-3), embed(4), eps(), embed(-1), embed(-2)]
            ],
            'c' => vec![
                vec![eps(), eps(), embed(1), embed(-3), embed(-3)],
                vec![embed(-1), embed(0), embed(5), eps(), embed(-1)],
                vec![embed(-5), embed(-3), embed(5), embed(1), embed(-1)],
                vec![eps(), embed(-2), embed(2), embed(-4), eps()],
                vec![embed(-1), eps(), embed(-2), embed(4), embed(-4)]
            ],
            'e' => vec![
                vec![eps(), eps(), embed(-4), embed(2), eps()],
                vec![eps(), embed(-4), eps(), embed(1), embed(-1)],
                vec![embed(-1), embed(-2), embed(-2), embed(2), embed(2)],
                vec![embed(0), embed(3), embed(5), embed(2), embed(-3)],
                vec![embed(4), embed(-3), embed(-3), embed(-2), embed(4)]
            ],
            'b' => vec![
                vec![embed(-1), eps(), embed(-3), eps(), embed(3)],
                vec![embed(4), embed(1), embed(-3), embed(4), embed(3)],
                vec![embed(1), embed(2), embed(4), embed(4), embed(0)],
                vec![embed(-4), eps(), eps(), embed(5), embed(2)],
                vec![embed(5), embed(1), embed(5), embed(5), embed(3)]
            ]
        ];
        let wfa_random = crate::wfa::WFA { ini, fin, trans };
        let opt = LearningOption {
            add_column_when_unsolvable: false,
            consistensify: false,
            detect_repeat_enclose: None,
            enclose_column: true,
            extraction_strategy: ExtractionStrategy::Naive,
            iter_limit: None,
            reduce_rows: true,
            enclose_row: true,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        dbg!(&wfa_random);
        let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                100000,
                20,
                1e-5,
                &mut rng,
            )
        };
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
        let mut subdata = SubData::new();
        let mut extracted_wfa = extract_wfa_full(
            &mut ls,
            &memq,
            &LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: true,
                ..Default::default()
            }, 
            &mut subdata
        )
        .unwrap();
        let mut timeout = true;
        for _ in 0..max_iteration {
            let eqq_res = eqq(&extracted_wfa);
            match eqq_res {
                None => {
                    timeout = false;
                    break;
                }
                Some((ce, _, _)) => {
                    add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                    extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
                }
            }
        }
        dbg!(&ls.pres.len(), &ls.sufs.len());
        if timeout {
            dbg!(ls);
            panic!();
        }
        let res = eqq(&extracted_wfa);
        assert_eq!(res, None);
    }
    #[test]
    fn test_extract_wfa_prob2() {
        let max_iteration = 1000;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let alph_size = 4;
        let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
        let ini = vec![embed(-2), embed(-1), embed(0)];
        let fin = vec![embed(5), embed(0), embed(3)];
        let trans = map![
            'd' => vec![
                vec![embed(3), embed(3), eps()],
                vec![embed(4), embed(-2), embed(5)],
                vec![embed(-1), embed(2), embed(-5)],
            ],
            'a' => vec![
                vec![eps(), embed(0), embed(5)],
                vec![embed(0), embed(-3), embed(0)],
                vec![embed(5), embed(4), embed(-2)],
            ],
            'c' => vec![
                vec![embed(2), embed(-3), embed(4)],
                vec![embed(5), embed(-3), embed(0)],
                vec![embed(-3), eps(), embed(4)],
            ],
            'b' => vec![
                vec![embed(-2),embed(-1), embed(0)],
                vec![embed(3), embed(-2), embed(4)],
                vec![embed(-2), embed(4), embed(-2)],
            ]
        ];
        let opt = LearningOption {
            add_column_when_unsolvable: false,
            consistensify: false,
            detect_repeat_enclose: None,
            enclose_column: true,
            enclose_row: true,
            extraction_strategy: ExtractionStrategy::Naive,
            iter_limit: None,
            reduce_rows: true,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        let wfa_random = crate::wfa::WFA { ini, fin, trans };
        dbg!(&wfa_random);
        let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                10000,
                50,
                1e-5,
                &mut rng,
            )
        };
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
        let mut subdata = SubData::new();
        let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
        let mut timeout = true;
        for _ in 0..max_iteration {
            let eqq_res = eqq(&extracted_wfa);
            match eqq_res {
                None => {
                    timeout = false;
                    break;
                }
                Some((ce, _, _)) => {
                    add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                    extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
                }
            }
        }
        dbg!(&ls);
        dbg!(&ls.pres.len(), &ls.sufs.len());
        assert!(!timeout);
        let res = crate::util::find_diff_systems(
            &memq,
            &(|x: &Vec<char>| extracted_wfa.run(x.iter().cloned())),
            &alph,
            100000,
            500,
            1e-5,
            &mut rng,
        );
        assert_eq!(res, None);
        dbg!(extracted_wfa);
    }
    #[test]
    fn test_extract_wfa_random3x() {
        let max_iteration = 1000;
        let mut detected_inf_encl = 0;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        'main: for _ in 0..10 {
            let d = 2;
            let alph_size = rng.gen_range(2, 3 + 1);
            let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
            let opt = LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: true,
                enclose_row: true,
                enclose_column: true,
                detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                    cache_size: alph_size * 10,
                    window_size: alph_size,
                    threshold: 10,
                }),
                extraction_strategy: ExtractionStrategy::Naive,
                add_word_mode: AddWordMode::Row,
                ..Default::default()
            };
            let ini: Vec<MaxPlusNumSemiring<i64>> =
                (0..d).map(|_| make_random_mpval(&mut rng)).collect();
            let fin: Vec<MaxPlusNumSemiring<i64>> =
                (0..d).map(|_| make_random_mpval(&mut rng)).collect();
            let mut trans = HashMap::<char, crate::matrix::Matrix<MaxPlusNumSemiring<i64>>>::new();
            for a in &alph {
                let m: crate::matrix::Matrix<MaxPlusNumSemiring<i64>> = (0..d)
                    .map(|_| (0..d).map(|_| make_random_mpval(&mut rng)).collect())
                    .collect();
                trans.insert(*a, m);
            }
            let wfa_random = crate::wfa::WFA { ini, fin, trans };
            dbg!(&wfa_random);
            let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
            let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
                crate::util::find_diff_systems(
                    &memq,
                    &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                    &alph,
                    100000,
                    20,
                    1e-5,
                    &mut rng,
                )
            };
            let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
            let mut subdata = SubData::new();
            let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
            let mut timeout = true;
            for i in 0..max_iteration {
                let eqq_res = eqq(&extracted_wfa);
                match eqq_res {
                    None => {
                        timeout = false;
                        break;
                    }
                    Some((ce, _, _)) => {
                        add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                        let res = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata);
                        match res {
                            Ok(a) => {
                                extracted_wfa = a;
                            }
                            Err(e) => match e {
                                ExtractionError::InfiniteEnclosing => {
                                    dbg!("Detected InfiniteEnclosing!");
                                    detected_inf_encl += 1;
                                    continue 'main;
                                }
                                ExtractionError::LimitExceeded(_) => panic!(),
                                ExtractionError::TimedOut(_) => panic!(),
                                ExtractionError::TimedOutInside => panic!()
                            },
                        }
                    }
                }
                dbg!(i);
            }
            dbg!(&ls.pres.len(), &ls.sufs.len());
            if timeout {
                dbg!(ls);
                panic!();
            }
            let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
                crate::util::find_diff_systems(
                    &memq,
                    &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                    &alph,
                    1000000,
                    20,
                    1e-5,
                    &mut rng,
                )
            };
            let res = eqq(&extracted_wfa);
            assert_eq!(res, None);
            dbg!(&extracted_wfa);
            check_sanity(&ls, &wfa_random, &extracted_wfa);
        }
        assert_eq!(detected_inf_encl, 0);
    }
    #[test]
    fn test_extract_wfa_prob3() {
        let max_iteration = 1000;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let alph_size = 2;
        let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
        let ini = vec![embed(-5), embed(4)];
        let fin = vec![embed(-4), embed(4)];
        let trans = map![
            'a' => vec![
                vec![embed(1), embed(-3)],
                vec![eps(), embed(3)],
            ],
            'c' => vec![
                vec![embed(-5), embed(4)],
                vec![embed(2), embed(-5)]
            ],
            'b' => vec![
                vec![eps(), embed(0)],
                vec![embed(-5), eps()],
            ]
        ];
        let wfa_random = crate::wfa::WFA { ini, fin, trans };
        let opt = LearningOption {
            add_column_when_unsolvable: false,
            consistensify: false,
            detect_repeat_enclose: None,
            enclose_row: true,
            enclose_column: true,
            extraction_strategy: ExtractionStrategy::Naive,
            iter_limit: None,
            reduce_rows: true,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        dbg!(&wfa_random);
        let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                10000,
                50,
                1e-5,
                &mut rng,
            )
        };
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
        let mut subdata = SubData::new();
        let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
        let mut timeout = true;
        for _ in 0..max_iteration {
            let eqq_res = eqq(&extracted_wfa);
            match eqq_res {
                None => {
                    timeout = false;
                    break;
                }
                Some((ce, _, _)) => {
                    add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                    extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
                }
            }
        }
        dbg!(&ls);
        dbg!(&ls.pres.len(), &ls.sufs.len());
        assert!(!timeout);
        let res = crate::util::find_diff_systems(
            &memq,
            &(|x: &Vec<char>| extracted_wfa.run(x.iter().cloned())),
            &alph,
            100000,
            500,
            1e-5,
            &mut rng,
        );
        assert_eq!(res, None);
        dbg!(extracted_wfa);
    }
    #[test]
    fn test_remove_unnecessary_rows1() {
        let alph_size = 2;
        let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
        let ini = vec![embed(-5), embed(4)];
        let fin = vec![embed(-4), embed(4)];
        let trans = map![
            'a' => vec![
                vec![embed(1), embed(-3)],
                vec![eps(), embed(3)],
            ],
            'c' => vec![
                vec![embed(-5), embed(4)],
                vec![embed(2), embed(-5)]
            ],
            'b' => vec![
                vec![eps(), embed(0)],
                vec![embed(-5), eps()],
            ]
        ];
        let wfa_random = crate::wfa::WFA { ini, fin, trans };
        let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
        let mut ls = init(
            &vec![vec!['b'], vec!['a'], vec!['a', 'a'], vec!['a', 'a', 'b']],
            &vec![
                vec![],
                vec!['a', 'b', 'c', 'c', 'c', 'a'],
                vec!['b', 'c', 'c', 'c', 'a'],
                vec!['c', 'c', 'c', 'a'],
                vec!['c', 'c', 'a'],
                vec!['c', 'a'],
                vec!['a'],
                vec!['b'],
                vec!['a', 'a'],
                vec!['a', 'a', 'b'],
                vec!['a', 'b'],
            ],
            &memq,
            &alph,
        );
        dbg!(&ls);
        remove_unnecessary_rows(&mut ls, None);
        dbg!(&ls);
        assert_eq!(ls.pres.len(), 2);
    }
    #[test]
    fn test_extract_wfa3_problem_replay1() {
        let w = WFA {
            ini: make_mp_vec![i64, 6, 11, 1],
            fin: make_mp_vec![i64, 7, 0, 6],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 2, 3, 1; 2, 0, 9; 3, 0, 8],
                'b' => make_mp_mat![i64, 9, 6, 2; 10, 3, 2; 8, 5, 4]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &"ab".chars().collect());
        let opt = LearningOption {
            iter_limit: None,
            reduce_rows: false,
            add_column_when_unsolvable: false,
            enclose_column: true,
            consistensify: false,
            extraction_strategy: ExtractionStrategy::Naive,
            detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                cache_size: 2,
                threshold: 100,
                window_size: 2,
            }),
            enclose_row: true,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &opt, &mut subdata).unwrap();
        add_word(&mut ls, &"aa".chars().collect(), &memq, &opt, &mut subdata).unwrap();
        enclose(&mut ls, &memq, &opt, &mut subdata).unwrap();
        add_word(&mut ls, &"abba".chars().collect(), &memq, &opt, &mut subdata).unwrap();
        enclose(&mut ls, &memq, &opt, &mut subdata).unwrap();
        dbg!(&ls);
        let extracted = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
        let ef = |x: &Vec<char>| extracted.run(x.iter().cloned());
        dbg!(&extracted);
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let d =
            crate::util::find_diff_systems(&memq, &ef, &"ab".chars().collect(), 1000, 5, 1e-5, &mut rng);
        assert_eq!(d, None);
    }
    #[test]
    fn test_extract_wfa3_problem_replay2() {
        let w = WFA {
            ini: make_mp_vec![i64, 6, 11, 1],
            fin: make_mp_vec![i64, 7, 0, 6],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 2, 3, 1; 2, 0, 9; 3, 0, 8],
                'b' => make_mp_mat![i64, 9, 6, 2; 10, 3, 2; 8, 5, 4]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &"ab".chars().collect());
        let opt = LearningOption {
            iter_limit: None,
            add_column_when_unsolvable: false,
            reduce_rows: false,
            enclose_column: true,
            consistensify: false,
            detect_repeat_enclose: None,
            extraction_strategy: ExtractionStrategy::Naive,
            enclose_row: true,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        let mut subdata = SubData::new();
        enclose(&mut ls, &memq, &opt, &mut subdata).unwrap();
        add_word(&mut ls, &"aa".chars().collect(), &memq, &opt, &mut subdata).unwrap();
        enclose(&mut ls, &memq, &opt, &mut subdata).unwrap();
        add_word(&mut ls, &"ab".chars().collect(), &memq, &opt, &mut subdata).unwrap();
        enclose(&mut ls, &memq, &opt, &mut subdata).unwrap();
        loop {
            let res_enclose = enclose(&mut ls, &memq, &LearningOption::default(), &mut subdata).unwrap();
            let res_consistensify = consistensify(
                &mut ls,
                &memq,
                &LearningOption {
                    consistensify: true,
                    ..Default::default()
                },
            );
            dbg!((&res_enclose, &res_consistensify));
            if !res_enclose && !res_consistensify {
                break;
            }
        }
        dbg!(&ls);
        let extracted = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
        let ef = |x: &Vec<char>| extracted.run(x.iter().cloned());
        dbg!(&extracted);
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let d =
            crate::util::find_diff_systems(&memq, &ef, &"ab".chars().collect(), 1000, 5, 1e-5, &mut rng);
        let x = init(
            &vec![
                vec![],
                vec!['a'],
                vec!['a', 'b'],
                vec!['b'],
                vec!['a', 'a'],
                vec!['a', 'b', 'a'],
                vec!['a', 'b', 'b'],
                vec!['b', 'a'],
                vec!['b', 'b'],
                vec!['a', 'a', 'a'],
                vec!['a', 'a', 'b'],
            ],
            &vec![vec![], vec!['a']],
            &memq,
            &"ab".chars().collect(),
        );
        dbg!(&x);
        assert_eq!(d, None);
    }
    #[test]
    fn test_extract_random1_problem_replay1() {
        let w = WFA {
            ini: make_mp_vec![i64, 1, 5, 4],
            fin: vec![eps(), embed(0), embed(-5)],
            trans: crate::map! {
                'a' => make_mp_mat![i64, 5, -5, 2; 3, -4, 4; 5, 1, 4],
                'b' => make_mp_mat![i64, -3, -4, -4; 4, 4, 4; 5, 0, 4],
                'c' => vec![vec![embed(0), embed(3), embed(2)],
                    vec![eps(), embed(-1), embed(4)],
                    vec![embed(5), embed(-5), embed(5)]]
            },
        };
        let opt = LearningOption {
            iter_limit: None,
            add_column_when_unsolvable: false,
            reduce_rows: true,
            enclose_column: true,
            consistensify: false,
            extraction_strategy: ExtractionStrategy::Naive,
            detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                threshold: 100,
                window_size: 3,
                cache_size: 3,
            }),
            enclose_row: true,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let pres = vec![
            vec![],
            vec!['a'],
            vec!['a', 'a'],
            vec!['a', 'a', 'a'],
            vec!['a', 'a', 'a', 'a'],
            vec!['a', 'a', 'a', 'a', 'a'],
            vec!['c'],
        ];
        let sufs = vec![
            vec![],
            vec!['a', 'c', 'c', 'b'],
            vec!['c', 'c', 'b'],
            vec!['c', 'b'],
            vec!['b'],
        ];
        let mut ls = init(&pres, &sufs, &memq, &vec!['a', 'b', 'c']);
        let mut subdata = SubData::new();
        let extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
        assert_eq!(
            w.run(vec!['c'].iter().cloned()),
            extracted_wfa.run(vec!['c'].iter().cloned())
        );
        dbg!(extracted_wfa);
    }
    fn test_check_infinity() {
        let w = WFA {
            ini: make_mp_vec![i64, 0, 0, 0],
            fin: vec![embed(0), eps(), eps()],
            trans: crate::map! {
                'a' => vec![vec![embed(0), eps(), eps()],
                    vec![eps(), embed(1), eps()],
                    vec![eps(), eps(), embed(2)]],
                'b' => vec![vec![eps(), eps(), eps()],
                    vec![embed(0), eps(), eps()],
                    vec![eps(), eps(), eps()]],
                'c' => vec![vec![eps(), eps(), eps()],
                    vec![eps(), eps(), eps()],
                    vec![embed(0), eps(), eps()]]
            },
        };
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let pres = vec![vec![]];
        let sufs = vec![vec![], vec!['b'], vec!['c']];
        let mut ls = init(&pres, &sufs, &memq, &vec!['a', 'b', 'c']);
        let opt = LearningOption {
            iter_limit: None,
            add_column_when_unsolvable: false,
            reduce_rows: true,
            enclose_column: true,
            consistensify: false,
            extraction_strategy: ExtractionStrategy::Naive,
            detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                threshold: 100,
                window_size: 3,
                cache_size: 3,
            }),
            enclose_row: true,
            add_word_mode: AddWordMode::Row,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        let mut subdata = SubData::new();
        let extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata);
        match extracted_wfa {
            Ok(_) => panic!(),
            Err(e) => match e {
                ExtractionError::InfiniteEnclosing => {}
                _ => {
                    panic!();
                }
            },
        }
    }
    #[test]
    fn test_extract_wfa_random1_replay1() {
        let max_iteration = 5000;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let mut _detected_inf_encl = 0;
        let d = 3;
        let alph_size = 3;
        let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
        let mut trans = HashMap::<char, crate::matrix::Matrix<MaxPlusNumSemiring<i64>>>::new();
        for a in &alph {
            let m: crate::matrix::Matrix<MaxPlusNumSemiring<i64>> = (0..d)
                .map(|_| (0..d).map(|_| make_random_mpval(&mut rng)).collect())
                .collect();
            trans.insert(*a, m);
        }
        let wfa_random = WFA {
            ini: vec![embed(-3), eps(), embed(-3)],
            fin: vec![eps(), embed(-2), eps()],
            trans: crate::map! {
                'a' => vec![vec![embed(0), embed(-3), embed(-3)],
                    vec![embed(-5), embed(5), embed(4)],
                    vec![embed(-2), embed(-3), embed(-2)]],
                'b' => vec![vec![embed(-2), embed(5), embed(-4)],
                    vec![embed(3), embed(-4), embed(3)],
                    vec![eps(), embed(3), eps()]],
                'c' => vec![vec![embed(-1), embed(4), embed(-4)],
                    vec![embed(0), embed(1), embed(-1)],
                    vec![embed(-4), embed(-5), embed(-1)]]
            },
        };
        dbg!(&wfa_random);
        let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                40000,
                20,
                1e-5,
                &mut rng,
            )
        };
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
        let opt = LearningOption {
            iter_limit: None,
            add_column_when_unsolvable: false,
            reduce_rows: true,
            enclose_column: true,
            detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                threshold: 100,
                window_size: 3,
                cache_size: 3,
            }),
            ..Default::default()
        };
        let mut subdata = SubData::new();
        let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
        let mut timeout = true;
        for _ in 0..max_iteration {
            let eqq_res = eqq(&extracted_wfa);
            match eqq_res {
                None => {
                    timeout = false;
                    break;
                }
                Some((ce, _, _)) => {
                    add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                    let res = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata);
                    match res {
                        Ok(a) => {
                            extracted_wfa = a;
                        }
                        Err(e) => match e {
                            ExtractionError::InfiniteEnclosing => {
                                dbg!("Detected InfiniteEnclosing!");
                                _detected_inf_encl += 1;
                                panic!();
                            }
                            ExtractionError::LimitExceeded(_) => panic!(),
                            ExtractionError::TimedOut(_) => panic!(),
                            ExtractionError::TimedOutInside => panic!()
                        },
                    }
                }
            }
        }
        dbg!(&ls.pres.len(), &ls.sufs.len());
        if timeout {
            dbg!(ls);
            dbg!(extracted_wfa);
            panic!();
        }
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                100000,
                20,
                1e-5,
                &mut rng,
            )
        };
        let res = eqq(&extracted_wfa);
        assert_eq!(res, None);
    }
    use std::fs;
    use std::io::{BufWriter, Write};
    #[test]
    fn test_extract_wfa_random1_replay2() {
        let max_iteration = 5000;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let mut _detected_inf_encl = 0;
        let alph_size = 4;
        let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
        let wfa_random = WFA {
            ini: vec![embed(-4), embed(0), embed(-5), embed(-4), embed(2), embed(-4), eps()],
            fin: vec![embed(-2), embed(-1), embed(4), embed(-5), embed(-5), embed(-1), eps()],
            trans: crate::map! {
                'a' =>vec![vec![embed(2),embed(2),embed(0),eps(),embed(-2),embed(-1),embed(1)],
                vec![eps(),embed(-2),embed(-3),embed(-4),embed(-2),embed(2),embed(-2)],
                vec![embed(-1),embed(-1),embed(4),eps(),embed(-1),embed(1),embed(-3)],
                vec![embed(3),embed(-5),embed(-1),embed(4),embed(-1),embed(-1),embed(-2)],
                vec![embed(-5),embed(3),embed(2),embed(4),embed(2),embed(2),embed(-5)],
                vec![embed(4),eps(),embed(-2),embed(2),embed(3),embed(-2),embed(0)],
                vec![embed(-4),embed(5),embed(-4),embed(4),embed(1),embed(1),embed(-4)]],
                'b' => vec![vec![embed(3),embed(5),embed(3),embed(3),eps(),embed(3),embed(5)],
                vec![embed(0),embed(-4),embed(5),embed(-5),embed(-4),embed(3),embed(-5)],
                vec![embed(1),embed(1),embed(0),embed(-5),embed(5),embed(3),embed(-2)],
                vec![embed(-1),embed(1),embed(2),eps(),embed(-5),embed(1),embed(0)],
                vec![embed(2),embed(-4),embed(-4),embed(3),embed(1),eps(),embed(3)],
                vec![embed(-5),embed(-1),embed(3),embed(-5),embed(4),eps(),eps()],
                vec![embed(2),embed(1),eps(),embed(-3),eps(),embed(5),embed(-5)]],
                'c' => vec![vec![embed(1),embed(0),embed(2),embed(5),embed(4),embed(1),embed(-4)],
                vec![embed(4),embed(3),embed(-5),embed(2),embed(0),embed(1),embed(0)],
                vec![embed(2),embed(-3),embed(-5),embed(2),embed(-5),embed(4),embed(-4)],
                vec![embed(-2),embed(-1),embed(-5),embed(4),embed(-4),eps(),embed(-4)],
                vec![embed(5),embed(5),embed(1),embed(4),embed(0),embed(4),embed(2)],
                vec![embed(-2),embed(-1),embed(-5),embed(3),embed(0),embed(-5),embed(-3)],
                vec![embed(-5),embed(-3),embed(-2),embed(2),eps(),embed(-2),embed(5)]],
                'd' => vec![vec![embed(-4),embed(-4),embed(-2),eps(),embed(-2),embed(5),eps()],
                vec![embed(-1),embed(-4),embed(-1),embed(0),embed(3),eps(),embed(-5)],
                vec![embed(-2),eps(),embed(0),embed(-3),embed(-2),embed(-2),embed(2)],
                vec![embed(-1),embed(-2),eps(),embed(-2),embed(4),embed(5),embed(-1)],
                vec![embed(4),embed(-5),embed(4),embed(-3),embed(-2),embed(-4),embed(0)],
                vec![eps(),embed(-4),embed(0),embed(-2),embed(-2),embed(-2),embed(3)],
                vec![embed(-2),embed(-1),eps(),eps(),embed(-1),embed(-1),embed(-1)]]
            },
        };
        dbg!(&wfa_random);
        let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                40000,
                20,
                1e-5,
                &mut rng,
            )
        };
        let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
        let opt = LearningOption {
            iter_limit: None,
            add_column_when_unsolvable: false,
            reduce_rows: true,
            enclose_row: true,
            enclose_column: true,
            detect_repeat_enclose: Some(DetectingRepeatEncloseOption {
                cache_size: alph_size * 10,
                window_size: alph_size,
                threshold: 10,
            }),
            extraction_strategy: ExtractionStrategy::Naive,
            add_word_mode: AddWordMode::Row,
            consistensify: false,
            timeout: None,
            solving_tol: Some(1e-5)
        };
        let mut subdata = SubData::new();
        let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
        let mut timeout = true;
        for _ in 0..max_iteration {
            let eqq_res = eqq(&extracted_wfa);
            match eqq_res {
                None => {
                    timeout = false;
                    break;
                }
                Some((ce, _, _)) => {
                    add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                    let res = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata);
                    match res {
                        Ok(a) => {
                            extracted_wfa = a;
                        }
                        Err(e) => match e {
                            ExtractionError::InfiniteEnclosing => {
                                dbg!("Detected InfiniteEnclosing!");
                                _detected_inf_encl += 1;
                                panic!();
                            }
                            ExtractionError::LimitExceeded(_) => panic!(),
                            ExtractionError::TimedOut(_) => panic!(),
                            ExtractionError::TimedOutInside => panic!()
                        },
                    }
                }
            }
        }
        dbg!(&ls.pres.len(), &ls.sufs.len());
        if timeout {
            dbg!(ls);
            dbg!(extracted_wfa);
            panic!();
        }
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                100000,
                20,
                1e-5,
                &mut rng,
            )
        };
        let res = eqq(&extracted_wfa);
        if let Some(ce) = &res {
            let mut words = HashSet::<Vec<char>>::new();
            for p in ls.pres.left_values() {
                for s in ls.sufs.left_values() {
                    words.insert(join_vec(p, s));
                }
            }
            if words.contains(&ce.0) {
                dbg!(&wfa_random);
                assert_eq!(res, None);
            }
        }
        let mut f = BufWriter::new(fs::File::create("pres.txt").unwrap());
        for i in 0..ls.pres.len(){
            let x: String = ls.pres.get_by_right(&i).unwrap().into_iter().collect();
            f.write(x.as_bytes()).unwrap();
            f.write(b"\n").unwrap();
        }
        let mut f = BufWriter::new(fs::File::create("sufs.txt").unwrap());
        for i in 0..ls.sufs.len(){
            let x: String = ls.sufs.get_by_right(&i).unwrap().into_iter().collect();
            f.write(x.as_bytes()).unwrap();
            f.write(b"\n").unwrap();
        }
        let mut f = BufWriter::new(fs::File::create("table.csv").unwrap());
        for i in 0..ls.pres.len(){
            for j in 0..ls.sufs.len(){
                let x = ls.table[i][j].clone();
                let y = match x{
                    MaxPlusNumSemiring::NegInf => "-inf".to_string(),
                    MaxPlusNumSemiring::Raw(v) => v.to_string()
                };
                f.write(y.as_bytes()).unwrap();
                f.write(b",").unwrap();
            }
            f.write(b"\n").unwrap();
        }
        assert_eq!(extracted_wfa.run("acdbb".chars()), embed(23));
        check_sanity(&ls, &wfa_random, &extracted_wfa);
    }
    fn make_random_rval(rng: &mut dyn rand::RngCore) -> RealField {
        RealField::Raw(rng.gen_range(0, 10 + 1) as f64 / 10.0)
    }
    fn normalize_real(v: &Vec<RealField>) -> Vec<RealField>{
        let s = v.iter().fold(RealField::zero(), |i, j|{i + j.clone()});
        if s.unwrap() < 1e-3{
            return v.clone();
        }
        return v.iter().map(|val|{RealField::Raw(val.unwrap() / s.unwrap())}).collect();
    }
    #[test]
    fn test_extract_wfa_real_random1x() {
        let eps = 1e-5;
        let max_iteration = 5000;
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let mut detected_inf_encl = 0;
        'main: for _ in 0..10 {
            let d = rng.gen_range(2, 10 + 1);
            let alph_size = rng.gen_range(2, 5 + 1);
            let alph: Vec<char> = "abcde"[0..alph_size].chars().collect();
            let ini: Vec<RealField> =
                normalize_real(&(0..d).map(|_| make_random_rval(&mut rng)).collect());
            let fin: Vec<RealField> =
                (0..d).map(|_| make_random_rval(&mut rng)).collect();
            let mut trans = HashMap::<char, crate::matrix::Matrix<RealField>>::new();
            for a in &alph {
                let m: crate::matrix::Matrix<RealField> = (0..d)
                    .map(|_| normalize_real(&(0..d).map(|_| make_random_rval(&mut rng)).collect()))
                    .collect();
                trans.insert(*a, m);
            }
            let wfa_random = crate::wfa::WFA { ini, fin, trans };
            dbg!(&wfa_random);
            let memq = |x: &Vec<char>| wfa_random.run(x.iter().cloned());
            let mut eqq = |wfa: &WFA<char, RealField>| {
                crate::real_field::find_diff_systems_real(
                    &memq,
                    &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                    &alph,
                    40000,
                    20,
                    &mut rng,
                    eps
                )
            };
            /* learn */
            let mut ls = init(&vec![vec![]], &vec![vec![]], &memq, &alph);
            let opt = LearningOption {
                iter_limit: None,
                add_column_when_unsolvable: false,
                reduce_rows: false,
                enclose_row: true,
                enclose_column: false,
                detect_repeat_enclose:None,
                extraction_strategy: ExtractionStrategy::Naive,
                add_word_mode: AddWordMode::Row,
                ..Default::default()
            };
            let mut subdata = SubData::new();
            let mut extracted_wfa = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata).unwrap();
            let mut timeout = true;
            for _ in 0..max_iteration {
                dbg!(&extracted_wfa);
                let eqq_res = eqq(&extracted_wfa);
                match eqq_res {
                    None => {
                        timeout = false;
                        break;
                    }
                    Some((ce, _, _)) => {
                        add_word(&mut ls, &ce, &memq, &opt, &mut subdata).unwrap();
                        let res = extract_wfa_full(&mut ls, &memq, &opt, &mut subdata);
                        match res {
                            Ok(a) => {
                                extracted_wfa = a;
                            }
                            Err(e) => match e {
                                ExtractionError::InfiniteEnclosing => {
                                    dbg!("Detected InfiniteEnclosing!");
                                    detected_inf_encl += 1;
                                    continue 'main;
                                }
                                ExtractionError::LimitExceeded(_) => panic!(),
                                ExtractionError::TimedOut(_) => panic!(),
                                ExtractionError::TimedOutInside => panic!()
                            },
                        }
                    }
                }
            }
            dbg!(&ls.pres.len(), &ls.sufs.len());
            if timeout {
                dbg!(ls);
                dbg!(extracted_wfa);
                panic!();
            }
            let mut eqq = |wfa: &WFA<char, RealField>| {
                crate::real_field::find_diff_systems_real(
                    &memq,
                    &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                    &alph,
                    100000,
                    20,
                    &mut rng,
                    eps
                )
            };
            let res = eqq(&extracted_wfa);
            if let Some(ce) = &res {
                let mut words = HashSet::<Vec<char>>::new();
                for p in ls.pres.left_values() {
                    for s in ls.sufs.left_values() {
                        words.insert(join_vec(p, s));
                    }
                }
                if words.contains(&ce.0) {
                    dbg!(&wfa_random);
                    assert_eq!(res, None);
                }
            }
            check_sanity_real(&ls, &wfa_random, &extracted_wfa, eps);
        }
        assert_eq!(detected_inf_encl, 0);
    }
    #[test]
    fn test_timeout() {
        let w = WFA {
            ini: make_mp_vec![i64, 0, 0, 0],
            fin: vec![embed(0), eps(), eps()],
            trans: crate::map! {
                'a' => vec![vec![embed(0), eps(), eps()],
                    vec![eps(), embed(1), eps()],
                    vec![eps(), eps(), embed(2)]],
                'b' => vec![vec![eps(), eps(), eps()],
                    vec![embed(0), eps(), eps()],
                    vec![eps(), eps(), eps()]],
                'c' => vec![vec![eps(), eps(), eps()],
                    vec![eps(), eps(), eps()],
                    vec![embed(0), eps(), eps()]]
            },
        };
        let mut rng = Box::new(rand::rngs::SmallRng::from_seed([
            42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        let alph = vec!['a', 'b', 'c'];
        let memq = |x: &Vec<char>| w.run(x.iter().cloned());
        let mut eqq = |wfa: &WFA<char, MaxPlusNumSemiring<i64>>| {
            let res = crate::util::find_diff_systems(
                &memq,
                &(|x: &Vec<char>| wfa.run(x.iter().cloned())),
                &alph,
                100000,
                20,
                1e-5,
                &mut rng,
            );
            match &res{
                None => None,
                Some(v) => Some(v.0.clone())
            }
        };
        let opt = LearningOption {
            iter_limit: None,
            add_column_when_unsolvable: false,
            reduce_rows: false,
            enclose_row: true,
            enclose_column: true,
            detect_repeat_enclose: None,
            extraction_strategy: ExtractionStrategy::Naive,
            add_word_mode: AddWordMode::Row,
            consistensify: false,
            timeout: Some(Duration::new(10, 0)),
            solving_tol: Some(1e-5)
        };
        let l = learn(&alph, &memq, &mut eqq, &opt);
        match &l{
            Err(ExtractionError::TimedOut(w)) => {
                dbg!(w);
            },
            _ => {
                panic!();
            }
        }
    }
}
