fn main() {
    #[cfg(feature = "output")]
    panic!("can't run default binary with output feature enabled");

    #[cfg(not(feature = "output"))]
    main_inner();
}

#[cfg(not(feature = "output"))]
fn main_inner() {
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

    println!("enumerating up to n = {n}...");

    let now = std::time::Instant::now();
    let map = polycubes::revisions::latest::solve(n);
    let time = now.elapsed();

    println!("total time: {:?}", time);

    let results = polycubes::serialization::Results::from_map(map);
    let (rs, ps) = results.average_rate(time);

    println!("performance: {} r/s, {} p/s", rs, ps);
}
