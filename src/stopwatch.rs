use std::time::{Duration, SystemTime};
#[derive(Debug)]
pub struct Stopwatch{
    starttime: Option<SystemTime>,
    pub total_ms: u128,
}
impl Stopwatch{
    pub fn new() -> Self{
        Self{
            starttime: None,
            total_ms : 0
        }
    }
    pub fn start(&mut self){
        assert_eq!(self.starttime, None);
        self.starttime = Some(SystemTime::now());
    }
    pub fn stop(&mut self){
        if let Some(starttime) = self.starttime{
            let dur = starttime.elapsed().unwrap().as_millis();
            self.starttime = None;
            self.total_ms += dur;
        }
    }
    pub fn get_sec(&self) -> u128{
        return self.total_ms / 1000;
    }
}
