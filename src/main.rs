fn main() {
    let mut args = pico_args::Arguments::from_env();
    let n: usize = args.free_from_str().expect("Error parsing number of cubes");
    let num_threads = args
        .opt_free_from_str()
        .expect("Error parsing number of threads");

    if let Some(num_threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
    }

    let now = std::time::Instant::now();

    println!("enumerating up to n = {n}...");

    polycubes::revisions::latest::solve(n);

    println!("total time: {:?}", now.elapsed());
}
