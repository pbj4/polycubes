use {
    nanoserde::{DeBin, SerBin},
    polycubes::serialization::*,
    std::{
        sync::mpsc,
        time::{Duration, Instant},
    },
};

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

    let (result_tx, work_rx) = spawn_server_connection(url);

    while let Ok(work) = work_rx.recv() {
        for polycube in work.jobs {
            let now = Instant::now();
            let map =
                polycubes::revisions::latest::solve_partial(work.target_n, polycube.de().view());
            let elasped = now.elapsed();

            let result = Results::from_map(map);

            if result_tx.send(((polycube, result.ser()), elasped)).is_err() {
                break;
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn spawn_server_connection(
    url: String,
) -> (
    mpsc::Sender<((SerPolycube, SerResults), Duration)>,
    mpsc::Receiver<JobResponse>,
) {
    let (result_tx, result_rx) = mpsc::channel();
    let (work_tx, work_rx) = mpsc::sync_channel(2);

    std::thread::spawn(move || {
        let mut overwriter = polycubes::Overwriter::default();
        let mut jobs_wanted = 1;
        let mut jobs_completed = 0;
        let (mut rs, mut ps) = (0, 0);
        loop {
            let (results, times): (std::collections::HashMap<_, _>, Vec<Duration>) =
                result_rx.try_iter().unzip();
            if !results.is_empty() {
                (rs, ps) = results
                    .values()
                    .map(SerResults::de)
                    .sum::<Results>()
                    .average_rate(times.into_iter().sum::<Duration>());
            }
            jobs_completed += results.len();

            let job_request = JobRequest {
                jobs_wanted,
                results,
            }
            .serialize_bin();

            let request_start = Instant::now();
            let mut delay = Duration::from_secs(1);
            let job_response = loop {
                match attohttpc::post(&url).bytes(&job_request).send() {
                    Ok(response) => {
                        let response =
                            JobResponse::deserialize_bin(&response.bytes().unwrap()).unwrap();
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
            let request_latency = request_start.elapsed();

            if job_response.jobs.is_empty() {
                break;
            }

            let received_jobs = job_response.jobs.len();

            let work_start = Instant::now();
            let blocked = if let Err(std::sync::mpsc::TrySendError::Full(job_response)) =
                work_tx.try_send(job_response)
            {
                work_tx.send(job_response).unwrap();
                true
            } else {
                false
            };
            let work_time = work_start.elapsed();

            overwriter.print(format!(
                "server latency: {:?}, current jobs: {}, total jobs: {}, work time: {:?}, r/s: {}, p/s: {}",
                request_latency, received_jobs, jobs_completed, work_time, rs, ps
            ));

            if work_time < 20 * request_latency || !blocked {
                jobs_wanted = (jobs_wanted * 4).div_ceil(3);
            } else {
                jobs_wanted = (jobs_wanted * 3 / 4).max(1);
            }
        }

        println!("\nwork finished");
        std::process::exit(0);
    });

    (result_tx, work_rx)
}
