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

    let now = std::time::Instant::now();

    println!("enumerating up to n = {n}...");

    let map = polycubes::revisions::latest::solve(n);

    let time = now.elapsed();

    println!("total time: {:?}", time);

    let r: usize = map.values().map(|(r, _)| r).sum();
    let p: usize = map.values().map(|(_, p)| p).sum();

    println!(
        "performance: {} r/s, {} p/s",
        (r as f64 / time.as_secs_f64()) as usize,
        (p as f64 / time.as_secs_f64()) as usize,
    );
}
