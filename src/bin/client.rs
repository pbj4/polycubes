use std::io::Write;

fn main() {
    let mut args = pico_args::Arguments::from_env();
    let url: String = args.free_from_str().expect("Error parsing server url");
    let num_threads = args
        .opt_free_from_str()
        .expect("Error parsing number of threads");

    let url = url.strip_suffix('/').unwrap_or(&url);
    let work_url = url.to_owned() + "/work";
    let result_url = url.to_owned() + "/result";

    if let Some(num_threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
    }

    let mut now = std::time::Instant::now();
    let mut jobs_done = 0;

    let mut delay = std::time::Duration::from_secs(1);
    loop {
        match ureq::get(&work_url).call() {
            Ok(response) => {
                if response.status() == 204 {
                    println!("\nwork finished");
                    break;
                }

                print!(
                    "\rjobs completed: {}, server latency: {:?}        ",
                    jobs_done,
                    now.elapsed(),
                );
                std::io::stdout().lock().flush().unwrap();

                let mut job = Vec::new();
                response.into_reader().read_to_end(&mut job).unwrap();

                let (n, polycube) = polycubes::serialization::job::deserialize(&job);
                let polycube = polycubes::serialization::polycube::deserialize(polycube);

                let map = polycubes::revisions::latest::solve_partial(n, polycube.view());

                let result = polycubes::serialization::result::serialize(polycube.view(), &map);

                now = std::time::Instant::now();

                let mut delay = std::time::Duration::from_secs(1);
                while let Err(err) = ureq::post(&result_url).send_bytes(&result) {
                    eprintln!("\nerror posting results: {err:?}");
                    eprintln!("retrying in {delay:?}");
                    std::thread::sleep(delay);
                    delay *= 2;
                }

                jobs_done += 1;
            }
            Err(err) => {
                eprintln!("\nerror posting results: {err:?}");
                eprintln!("retrying in {delay:?}");
                std::thread::sleep(delay);
                delay *= 2;
            }
        }
    }
}
