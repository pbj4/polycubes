use clap::Parser;

mod revisions;

fn main() {
    let now = std::time::Instant::now();

    let args = Args::parse();

    if let Some(num_threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
    }

    let n = args.n;

    println!("enumerating up to n = {n}...");

    revisions::latest::solve(n);

    println!("total time: {:?}", now.elapsed());
}

#[derive(Parser)]
struct Args {
    /// target number of cubes to go up to
    n: usize,
    /// number of threads to use
    threads: Option<usize>,
}
