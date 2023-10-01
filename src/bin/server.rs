use {
    polycubes::serialization,
    std::io::{BufRead, Write},
    tiny_http::{Method, Response, Server},
};

fn main() {
    println!("opening database");

    let db = sled::open("serverdb").unwrap();
    let meta = db.open_tree("meta").unwrap();

    if !meta.contains_key(DB_FINISH_KEY).unwrap() {
        let mut args = pico_args::Arguments::from_env();
        let listen_addr: String = args.free_from_str().expect("Error parsing listen address");

        let (initial_n, target_n): (usize, usize) = if let (Some(initial_n), Some(target_n)) = (
            meta.get(DB_INITIAL_N_KEY).unwrap(),
            meta.get(DB_TARGET_N_KEY).unwrap(),
        ) {
            print!("loading saved configuration: ");
            let [initial_n]: [u8; 1] = initial_n.as_ref().try_into().unwrap();
            let [target_n]: [u8; 1] = target_n.as_ref().try_into().unwrap();
            (initial_n.into(), target_n.into())
        } else {
            print!("loading configuration from cli: ");
            let initial_n: u8 = args.free_from_str().expect("Error parsing initial_n");
            let target_n: u8 = args.free_from_str().expect("Error parsing target_n");
            meta.insert(DB_INITIAL_N_KEY, &[initial_n]).unwrap();
            meta.insert(DB_TARGET_N_KEY, &[target_n]).unwrap();
            (initial_n.into(), target_n.into())
        };

        println!("initial_n = {initial_n}, target_n = {target_n}");

        if !meta.contains_key(DB_INIT_KEY).unwrap() {
            println!("generating initial polycubes");

            polycubes::revisions::latest::solve_out(initial_n, &|view| {
                db.insert(serialization::polycube::serialize(view), &[])
                    .unwrap();
            });
            db.flush().unwrap();

            meta.insert(DB_INIT_KEY, &[]).unwrap();
        }

        let now = std::time::Instant::now();

        let http = Server::http(&listen_addr).unwrap();

        println!("listening on http://{listen_addr}/ for job requests...");

        let job_iter_producer = || {
            db.iter()
                .map(Result::unwrap)
                .filter_map(|(k, v)| v.is_empty().then_some(k))
        };
        let mut job_iter = job_iter_producer();
        let mut get_job = || {
            if let Some(job) = job_iter.next() {
                Some(job)
            } else {
                job_iter = job_iter_producer();
                job_iter.next()
            }
        };

        let total_jobs = db.iter().count();
        let mut jobs_done = db
            .iter()
            .map(Result::unwrap)
            .filter(|(_, v)| !v.is_empty())
            .count();
        let print_job_progress = |jobs_done| {
            print!("\rjobs processed: {jobs_done}/{total_jobs}");
            std::io::stdout().lock().flush().unwrap();
        };
        print_job_progress(jobs_done);

        while let Ok(mut request) = http.recv() {
            match (request.url(), request.method()) {
                ("/work", Method::Get) => {
                    if let Some(polycube) = get_job() {
                        request
                            .respond(Response::new(
                                200.into(),
                                vec![],
                                serialization::job::serialize(polycube.as_ref(), target_n)
                                    .as_slice(),
                                None,
                                None,
                            ))
                            .unwrap();
                    } else {
                        request.respond(Response::empty(204)).unwrap();
                        break;
                    }
                }
                ("/result", Method::Post) => {
                    let mut result = Vec::new();
                    request.as_reader().read_to_end(&mut result).unwrap();

                    let (polycube, counts) = serialization::result::as_key_value(&result);
                    let old = db.insert(polycube, counts).unwrap().unwrap();

                    if old.is_empty() {
                        jobs_done += 1;
                        print_job_progress(jobs_done);
                    }

                    let status_code = if old.is_empty() || counts == old.as_ref() {
                        200
                    } else {
                        400
                    };

                    request.respond(Response::empty(status_code)).unwrap();
                }
                _ => request.respond(Response::empty(404)).unwrap(),
            }
        }

        let counts = db
            .iter()
            .map(Result::unwrap)
            .fold(vec![(0, 0); target_n], |mut a, (_, c)| {
                let c = serialization::result::deserialize_counts(c.as_ref());
                assert_eq!(a.len(), c.len());

                for ((ar, ap), (cr, cp)) in a.iter_mut().zip(c) {
                    *ar += cr;
                    *ap += cp;
                }

                a
            });

        meta.insert(
            DB_FINISH_KEY,
            serialization::result::serialize_counts(counts.iter().cloned()),
        )
        .unwrap();
        meta.flush().unwrap();

        // keep telling clients work is finished, will run until process exits
        std::thread::spawn(move || {
            while let Ok(request) = http.recv() {
                let status_code = if let ("/work", Method::Get) = (request.url(), request.method())
                {
                    204
                } else if let ("/result", Method::Post) = (request.url(), request.method()) {
                    200
                } else {
                    404
                };

                request.respond(Response::empty(status_code)).unwrap();
            }
        });

        println!();
        println!("finish time: {:?}", now.elapsed())
    }

    println!("results:");

    let counts = serialization::result::deserialize_counts(
        meta.get(DB_FINISH_KEY).unwrap().unwrap().as_ref(),
    );

    let [initial_n]: [u8; 1] = meta
        .get(DB_INITIAL_N_KEY)
        .unwrap()
        .unwrap()
        .as_ref()
        .try_into()
        .unwrap();

    for (i, (r, p)) in counts.iter().enumerate() {
        let i = i + 1;
        if i >= initial_n.into() {
            println!("n: {:?}, r: {:?}, p: {:?}", i, r, p);
        }
    }

    print!("press enter to exit...");
    std::io::stdout().lock().flush().unwrap();
    std::io::stdin()
        .lock()
        .read_line(&mut String::new())
        .unwrap();
}

const DB_INIT_KEY: &str = "db_init";
const DB_FINISH_KEY: &str = "db_finish";
const DB_INITIAL_N_KEY: &str = "db_initial_n";
const DB_TARGET_N_KEY: &str = "db_target_n";
