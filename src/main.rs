extern crate getopts;
use getopts::Options;
use std::env;
extern crate tropical_learning;
use tropical_learning::semiring;
use std::fs::File;
use std::io::prelude::*;
use tropical_learning::learning;
use std::time::{Duration, SystemTime, SystemTimeError};
use serde::{Deserialize, Serialize};
use tropical_learning::max_plus_semiring::MaxPlusNumSemiring;
use tropical_learning::real_field::RealField;
use tropical_learning::wfa;
use chrono::prelude::*;
enum SemiringType{
    RealField,
    MaxPlusSemiring
}
fn str2sr(s: &str) -> SemiringType{
    match s{
        "R" => SemiringType::RealField,
        "MP" => SemiringType::MaxPlusSemiring,
        _ => panic!("unexpectd semiring type")
    }
}
pub struct ArgOption{
    wfafile: String,
    timeout: Option<usize>,
    eqqfile: String,
    valfile: String,
    tol: f64,
    reduce_rows: bool
}
fn check_sanity<S>(ls: &learning::LearningState<char, S>, wfa: &wfa::WFA<char, S>, tol: f64, rfile: &mut ResultFile)
where S: tropical_learning::SolvableSemiring{
    let mut words = std::collections::HashMap::<Vec<char>, (Vec<char>, Vec<char>)>::new();
    for p in ls.pres.left_values() {
        for s in ls.sufs.left_values() {
            words.insert(tropical_learning::util::join_vec(p, s), (p.clone(), s.clone()));
        }
    }
    for (k, v) in words{
        let exp = ls.table[*ls.pres.get_by_left(&v.0).unwrap()][*ls.sufs.get_by_left(&v.1).unwrap()].clone();
        let res = wfa.run(k.iter().cloned());
        if S::dist(&exp, &res) > tol*2.1 {
            let s: String = k.iter().cloned().collect();
            rfile.error.push(format!("SanityCheck: Insanity ({}, {}, {}) found!", &s, &exp, &res));
        }
    }
}
fn make_eqq_from_file<'a, S1, S2>(path: &String, orig: &'a tropical_learning::wfa::WFA<char, S1>, tol: f64) -> Box<dyn FnMut(&tropical_learning::wfa::WFA<char, S2>) -> Option<Vec<char>> + 'a>
where S1: semiring::Semiring,
S2: semiring::Semiring
{
    let words = tropical_learning::reader::read_file_as_words(path);
    let mut used = std::collections::HashSet::<Vec<char>>::new();
    let f = move |cand: &tropical_learning::wfa::WFA<char, S2>|{
        let mut current_max: Option<(Vec<char>, f64)> = None;
        let mut sum_of_sq_errors = 0.0;
        for w in words.iter(){
            let val_orig = orig.run(w.iter().cloned());
            let val_cand = cand.run(w.iter().cloned());
            let diff = (S1::to_f64(&val_orig) - S2::to_f64(&val_cand)).abs();
            if let Some((_, cm_v)) = &current_max{
                if diff > *cm_v{
                    if !used.contains(w){
                        current_max = Some((w.clone(), diff));
                    }
                }
            }else{
                if !used.contains(w){
                    current_max = Some((w.clone(), diff));
                }
            }
            sum_of_sq_errors += diff*diff;
        }
        let rmse = sum_of_sq_errors / (words.len() as f64);
        if rmse <= tol{
            return None;
        }else{
            if let Some((cm_w, _)) = current_max{
                used.insert(cm_w.clone());
                return Some(cm_w);
            }else{
                return None;
            }
        }
    };
    return Box::new(f);
}
#[derive(Deserialize, Serialize, Clone, Debug)]
struct IntermidWFADatum{
    wfa_size: usize,
    constructed_at: u64,
    accuracy: f64,
    rmse: f64
}
#[derive(Deserialize, Serialize, Clone, Debug)]
struct ResultFile{
    wfa_size: usize,
    elapsed_time: u64,
    accuracy: f64,
    rmse: f64,
    error: Vec<String>,
    wfa: String,
    executions: Vec<(String, f64, f64)>,
    ls: String,
    num_construction: usize,
    time_enclose_row: u128,
    time_enclose_column: u128,
    time_construction: u128,
    intermid_wfa_data: Vec<IntermidWFADatum>
}
impl ResultFile{
    fn new() -> Self{
        ResultFile{
            wfa_size: 0,
            elapsed_time: 0,
            accuracy: -1.0,
            rmse: -1.0,
            error: vec![],
            wfa: "".to_string(),
            executions: vec![],
            ls: String::new(),
            num_construction: 0,
            time_enclose_row: 0,
            time_enclose_column: 0,
            time_construction: 0,
            intermid_wfa_data: Vec::new()
        }
    }
}
fn check_performance<S1, S2>(rfile: &mut ResultFile, elapsed: &Result<Duration, SystemTimeError>, opts: &ArgOption, orig_wfa: &wfa::WFA<char, S1>, extracted: &wfa::WFA<char, S2>, subdata: &learning::SubData<char, S2>)
where S1: tropical_learning::SolvableSemiring,
S2: tropical_learning::SolvableSemiring
{
    rfile.wfa_size = extracted.ini.len();
    rfile.elapsed_time = elapsed.clone().unwrap_or(Duration::new(0, 0)).as_secs();
    let words = tropical_learning::reader::read_file_as_words(&opts.valfile);
    let mut sum_of_sq_errors = 0.0;
    let mut num_correct = 0;
    let numerator = words.len();
    for w in words{
        let val_orig = S1::to_f64(&orig_wfa.run(w.iter().cloned()));
        let val_ext = S2::to_f64(&extracted.run(w.iter().cloned()));
        if (val_orig - val_ext).abs() <= opts.tol{
            num_correct += 1;
        }
        sum_of_sq_errors += (val_orig - val_ext).powf(2.0);
        rfile.executions.push((w.into_iter().collect(), val_orig, val_ext));
    }
    rfile.accuracy = (num_correct as f64)/(numerator as f64);
    rfile.rmse = (sum_of_sq_errors/(numerator as f64)).sqrt();
    rfile.wfa = format!("{:?}", extracted);
    if let Some(ls) = &subdata.ls{
        check_sanity(&ls, &extracted, opts.tol, rfile)
    }else{
        rfile.error.push("SanityCheck: LearningState not found!".to_string());
    }
    rfile.error.push(subdata.warnings.join("\n"));
    rfile.num_construction = subdata.num_construction;
    rfile.time_enclose_column = subdata.sw_enclose_column.get_sec();
    rfile.time_enclose_row = subdata.sw_enclose_row.get_sec();
    rfile.time_construction = subdata.sw_construction.get_sec();
}
fn make_rfile<S1, S2>(rfile: &mut ResultFile, elapsed: &Result<Duration, SystemTimeError>, opts: &ArgOption, orig_wfa: &wfa::WFA<char, S1>, extracted: &wfa::WFA<char, S2>, subdata: &learning::SubData<char, S2>)
where S1: tropical_learning::SolvableSemiring,
S2: tropical_learning::SolvableSemiring
{
    check_performance(rfile, elapsed, opts, orig_wfa, extracted, subdata);
    for i in 0..subdata.num_construction{
        let filename = format!("result/wfa{}.json", i);
        let intermid_wfa: wfa::WFA<char, S2> = match &mut File::open(filename) {
            Err(e) => panic!(),
            Ok(f) => {
                let mut cont = String::new();
                let res = f.read_to_string(&mut cont);
                match res {
                    Err(e) => panic!(),
                    Ok(_size) => {
                        serde_json::from_str(&cont.as_str()).unwrap()
                    }
                }
            }
        };
        let mut rfile_temp = ResultFile::new();
        let subdata_dummy = learning::SubData::new();
        check_performance(&mut rfile_temp, &Ok(std::time::Duration::new(0, 0)), opts, orig_wfa, &intermid_wfa, &subdata_dummy);
        let iwd = IntermidWFADatum{
            wfa_size: intermid_wfa.ini.len(),
            constructed_at: subdata.constructed_at[i],
            accuracy: rfile_temp.accuracy,
            rmse: rfile_temp.rmse
        };
        rfile.intermid_wfa_data.push(iwd);
    }
}
fn learn_wfa2wfa<S1, S2>(opts: &ArgOption) -> String
where S1: tropical_learning::SolvableSemiring,
S2: tropical_learning::SolvableSemiring
{
    let raw_wfa = tropical_learning::reader::read_file_as_wfa(&opts.wfafile).unwrap();
    let orig_wfa = tropical_learning::reader::convert_into_proper_wfa::<S1>(&raw_wfa);
    let memq = |x: &Vec<char>| S2::from_f64(S1::to_f64(&orig_wfa.run(x.iter().cloned())));
    let mut eqq = make_eqq_from_file::<S1, S2>(&opts.eqqfile, &orig_wfa, opts.tol);
    let alphabet = orig_wfa.get_alphabet();
    let lopt = learning::LearningOption{
        add_column_when_unsolvable: false,
        add_word_mode: learning::AddWordMode::Row,
        consistensify: false,
        detect_repeat_enclose: None,
        enclose_column: S2::need_enclose_column(),
        enclose_row: true,
        extraction_strategy: learning::ExtractionStrategy::Naive,
        iter_limit: None,
        reduce_rows: S2::need_reduce_rows() && opts.reduce_rows,
        solving_tol: Some(opts.tol),
        timeout: match opts.timeout{
            None => None,
            Some(timeout) => Some(std::time::Duration::new(timeout as u64, 0))
        } 
    };
    let start_time = SystemTime::now();
    let mut extracted = tropical_learning::learning::learn(&alphabet, &memq, eqq.as_mut(), &lopt);
    let elapsed = start_time.elapsed();
    let mut rfile = ResultFile::new();
    match &mut extracted{
        Ok((extracted, subdata)) => {
            subdata.stop_all_sw();
            subdata.wait_for_writings();
            make_rfile(&mut rfile, &elapsed, opts, &orig_wfa, &extracted, &subdata);
        },
        Err(e) =>{
            if let learning::ExtractionError::TimedOut((extracted, subdata)) = e{
                if let Some(extracted) = extracted{
                    subdata.stop_all_sw();
                    subdata.wait_for_writings();
                    make_rfile(&mut rfile, &elapsed, opts, &orig_wfa, &extracted, &subdata);
                }
                rfile.error.push("ExtractionError::TimedOut occured!".to_string());
            }else{
                rfile.error.push(format!("{:?}", e));
            }
        }
    }
    let s = serde_json::to_string_pretty(&rfile).unwrap();
    return s;
}
fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} FILE [options]", program);
    print!("{}", opts.usage(&brief));
}
pub fn validate_dumped_json<S1, S2>(path: &String, opts: &ArgOption)
where
S1: tropical_learning::SolvableSemiring,
S2: tropical_learning::SolvableSemiring
{
    let extracted = match &mut File::open(path) {
        Err(e) => panic!(),
        Ok(f) => {
            let mut cont = String::new();
            let res = f.read_to_string(&mut cont);
            match res {
                Err(e) => panic!(),
                Ok(_size) => {
                    serde_json::from_str(&cont).unwrap()
                }
            }
        }
    };
    let original = tropical_learning::reader::convert_into_proper_wfa::<S1>(&tropical_learning::reader::read_file_as_wfa(&opts.wfafile).unwrap());
    let mut rfile_temp = ResultFile::new();
        let subdata_dummy = learning::SubData::<char, S2>::new();
        check_performance(&mut rfile_temp, &Ok(std::time::Duration::new(0, 0)), opts, &original, &extracted, &subdata_dummy);
    println!("acc: {}, rmse: {}", rfile_temp.accuracy, rfile_temp.rmse);
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    opts.optflag("", "disable_reduce_rows", "disable the minimization (reducing rows)");
    opts.optopt("", "wfa", "wfa file in json", "WFAJSON");
    opts.optopt("", "isr", "input semiring", "R|MP");
    opts.optopt("", "osr", "output semiring", "R|MP");
    opts.optopt("", "timeout", "timeout (secs).  Disabled if 0 is specified.", "TIMEOUT");
    opts.optopt("", "eqq", "equivalence query file", "EQQFILE");
    opts.optopt("", "val", "valiation file", "VALFILE");
    opts.optopt("", "tol", "tolerance", "TOL");
    opts.optopt("", "ext_wfajson", "work as validation mode.  specifies the dumped wfa", "EXT_WFAJSON");
    let matches = match opts.parse(&args[1..]){
        Ok(m) => {m}
        Err(f) => { panic!(f.to_string())}
    };
    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }
    let wfafile = matches.opt_str("wfa").expect("-wfa is required");
    let input_semiring = str2sr(&matches.opt_str("isr").expect("-isr is required"));
    let output_semiring = str2sr(&matches.opt_str("osr").expect("-osr is required"));
    let timeout: usize = matches.opt_str("timeout").unwrap().parse().unwrap();
    let eqqfile = matches.opt_str("eqq").expect("-eqq is required");
    let valfile = matches.opt_str("val").expect("-val is required");
    let tol: f64 = matches.opt_str("tol").unwrap().parse().unwrap();
    let ext_wfajson = matches.opt_str("ext_wfajson");
    let argopt = ArgOption{
        wfafile: wfafile,
        timeout: if timeout == 0 {
            None
        }else{
            Some(timeout)
        },
        eqqfile: eqqfile,
        valfile: valfile,
        tol: tol,
        reduce_rows: !matches.opt_present("disable_reduce_rows")
    };
    if let Some(ext_wfajson) = ext_wfajson{
        let res = match (input_semiring, output_semiring){
            (SemiringType::MaxPlusSemiring, SemiringType::MaxPlusSemiring) => validate_dumped_json::<MaxPlusNumSemiring<f64>, MaxPlusNumSemiring<f64>>(&ext_wfajson, &argopt),
            (SemiringType::RealField, SemiringType::MaxPlusSemiring) => validate_dumped_json::<RealField, MaxPlusNumSemiring<f64>>(&ext_wfajson, &argopt),
            (SemiringType::MaxPlusSemiring, SemiringType::RealField) => validate_dumped_json::<MaxPlusNumSemiring<f64>, RealField>(&ext_wfajson, &argopt),
            (SemiringType::RealField, SemiringType::RealField) => validate_dumped_json::<RealField, RealField>(&ext_wfajson, &argopt),
        } ;
        return;
    }
    let res = match (input_semiring, output_semiring){
        (SemiringType::MaxPlusSemiring, SemiringType::MaxPlusSemiring) => learn_wfa2wfa::<MaxPlusNumSemiring<f64>, MaxPlusNumSemiring<f64>>(&argopt),
        (SemiringType::RealField, SemiringType::MaxPlusSemiring) => learn_wfa2wfa::<RealField, MaxPlusNumSemiring<f64>>(&argopt),
        (SemiringType::MaxPlusSemiring, SemiringType::RealField) => learn_wfa2wfa::<MaxPlusNumSemiring<f64>, RealField>(&argopt),
        (SemiringType::RealField, SemiringType::RealField) => learn_wfa2wfa::<RealField, RealField>(&argopt),
    } ;
    let result_path = std::path::Path::new("result");
    if !result_path.exists() || !result_path.is_dir(){
        panic!("Directory 'result' not found!");
    }
    let mut f = File::create("result/result.json").expect("cannot open result.json");
    write!(f, "{}", res).expect("failed to write in result.json");
    println!("Completed successfully!");
}
