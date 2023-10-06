use {polycubes::serialization, std::sync::mpsc};

fn main() {
    let mut args = pico_args::Arguments::from_env();
    let url: String = args.free_from_str().expect("Error parsing server url");
    let num_threads = args
        .opt_free_from_str()
        .expect("Error parsing number of threads");

    let url = url.strip_suffix('/').unwrap_or(&url).to_owned() + "/work";

    if let Some(num_threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
    }

    let (result_tx, work_rx) = connect_network(url, 10);

    while let Ok(work) = work_rx.recv() {
        for polycube in work.jobs {
            let map =
                polycubes::revisions::latest::solve_partial(work.target_n, polycube.de().view());

            let result = serialization::Results::from_map(&map);

            if let Err(_) = result_tx.send((polycube, result.ser())) {
                break;
            }
        }
    }

    println!("work finished")
}

fn connect_network(
    url: String,
    jobs_wanted: usize,
) -> (
    mpsc::Sender<(serialization::SerPolycube, serialization::SerResults)>,
    mpsc::Receiver<serialization::JobResponse>,
) {
    let (result_tx, result_rx) = mpsc::channel();
    let (work_tx, work_rx) = mpsc::sync_channel(3);

    std::thread::spawn(move || loop {
        use nanoserde::{DeBin, SerBin};
        let job_request = serialization::JobRequest {
            jobs_wanted,
            results: result_rx.try_iter().collect(),
        }
        .serialize_bin();

        let mut delay = std::time::Duration::from_secs(1);
        let job_response = loop {
            match ureq::post(&url).send_bytes(&job_request) {
                Ok(response) => {
                    let mut buf = Vec::new();
                    response.into_reader().read_to_end(&mut buf).unwrap();
                    let response = serialization::JobResponse::deserialize_bin(&buf).unwrap();
                    break response;
                }
                Err(err) => {
                    eprintln!("\nerror posting results: {err:?}");
                    eprintln!("retrying in {delay:?}");
                    std::thread::sleep(delay);
                    delay *= 2;
                }
            }
        };

        if job_response.jobs.is_empty() {
            break;
        } else {
            work_tx.send(job_response).unwrap();
        }
    });

    (result_tx, work_rx)
}
