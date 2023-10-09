pub mod revisions;
pub mod serialization;

pub fn print_overwrite(str: String) {
    use std::io::Write;

    print!("\r");
    for _ in 0..str.len() + 1 {
        print!(" ");
    }
    std::io::stdout().flush().unwrap();
    print!("\r{str}");
    std::io::stdout().flush().unwrap();
}
