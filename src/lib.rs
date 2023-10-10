pub mod revisions;
pub mod serialization;

#[derive(Default)]
pub struct Overwriter {
    last_len: usize,
}

impl Overwriter {
    pub fn print(&mut self, str: String) {
        use std::io::Write;
        print!("\r");
        for _ in 0..self.last_len {
            print!(" ");
        }
        std::io::stdout().flush().unwrap();
        print!("\r{str}");
        std::io::stdout().flush().unwrap();
        self.last_len = str.len();
    }
}
