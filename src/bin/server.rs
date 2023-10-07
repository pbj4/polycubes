use {
    nanoserde::{DeBin, SerBin},
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

        print!("loading configuration from ");
        let (initial_n, target_n) = if let Some(config) = get_config_db(&meta) {
            print!("db: ");
            config
        } else {
            print!("cli: ");
            get_config_cli(&mut args, &meta)
        };

        println!("initial_n = {initial_n}, target_n = {target_n}");

        if !meta.contains_key(DB_INIT_KEY).unwrap() {
            println!("generating initial polycubes");

            polycubes::revisions::latest::solve_out(initial_n, &|view| {
                db.insert(serialization::SerPolycube::ser(view).as_slice(), &[])
                    .unwrap();
            });
            db.flush().unwrap();

            meta.insert(DB_INIT_KEY, &[]).unwrap();
        }

        let now = std::time::Instant::now();

        let http = Server::http(&listen_addr).unwrap();

        println!("listening on http://{listen_addr}/ for job requests...");

        let mut job_finder = JobFinder::new(&db);
        let mut job_tracker = JobTracker::new(&db);
        print!("\r{job_tracker}");
        std::io::stdout().lock().flush().unwrap();

        'handle: while let Ok(mut request) = http.recv() {
            if let ("/work", Method::Post) = (request.url(), request.method()) {
                // handle result
                let mut job_request = Vec::new();
                request.as_reader().read_to_end(&mut job_request).unwrap();
                let job_request = serialization::JobRequest::deserialize_bin(&job_request).unwrap();

                for (polycube, result) in job_request.results {
                    let old = db
                        .insert(polycube.as_slice(), result.as_slice())
                        .unwrap()
                        .unwrap();

                    if old.is_empty() {
                        job_tracker.increment();
                        print!("\r{job_tracker}");
                        std::io::stdout().lock().flush().unwrap();
                    } else if old != result.as_slice() {
                        eprintln!(
                            "conflicting result from {:?} for {:?}, {:?} vs {:?}",
                            request.remote_addr(),
                            polycube.as_slice(),
                            old,
                            result.as_slice(),
                        );
                        request.respond(Response::empty(400)).unwrap();
                        continue 'handle;
                    }
                }

                // assign job
                let job_response = serialization::JobResponse {
                    target_n,
                    jobs: job_finder
                        .assign(job_request.jobs_wanted)
                        .into_iter()
                        .map(|iv| serialization::SerPolycube::from_slice(&iv))
                        .collect(),
                };

                request
                    .respond(Response::new(
                        200.into(),
                        vec![],
                        job_response.serialize_bin().as_slice(),
                        None,
                        None,
                    ))
                    .unwrap();

                if job_finder.finished() || job_tracker.finished() {
                    break 'handle;
                }
            }
        }

        let counts = db
            .iter()
            .map(Result::unwrap)
            .fold(vec![(0, 0); target_n], |mut a, (_, c)| {
                let c = serialization::SerResults::from_slice(&c).de().into_vec();
                assert_eq!(a.len(), c.len());

                for ((ar, ap), (cr, cp)) in a.iter_mut().zip(c) {
                    *ar += cr;
                    *ap += cp;
                }

                a
            });

        meta.insert(
            DB_FINISH_KEY,
            serialization::Results::from_vec(counts)
                .ser()
                .serialize_bin(),
        )
        .unwrap();
        meta.flush().unwrap();

        println!();
        println!("finish time: {:?}", now.elapsed())
    }

    println!("results:");

    let counts =
        serialization::SerResults::deserialize_bin(&meta.get(DB_FINISH_KEY).unwrap().unwrap())
            .unwrap()
            .de()
            .into_vec();

    let (initial_n, _) = get_config_db(&meta).unwrap();

    for (i, (r, p)) in counts.iter().enumerate() {
        let i = i + 1;
        if i >= initial_n {
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

struct JobFinder {
    db: sled::Tree,
    iter: Option<sled::Iter>,
}

impl JobFinder {
    fn new(db: &sled::Tree) -> Self {
        Self {
            db: db.clone(),
            iter: Some(db.iter()),
        }
    }

    fn finished(&self) -> bool {
        self.iter.is_none()
    }

    fn assign(&mut self, max: usize) -> Vec<sled::IVec> {
        let max = max.max(1);

        if let Some(iter) = &mut self.iter {
            let vec: Vec<sled::IVec> = iter
                .by_ref()
                .map(Result::unwrap)
                .filter_map(|(k, v)| v.is_empty().then_some(k))
                .take(max)
                .collect();

            if vec.is_empty() {
                *iter = self.db.iter();

                let vec: Vec<sled::IVec> = iter
                    .by_ref()
                    .map(Result::unwrap)
                    .filter_map(|(k, v)| v.is_empty().then_some(k))
                    .take(max)
                    .collect();

                if vec.is_empty() {
                    self.iter = None;
                }

                vec
            } else {
                vec
            }
        } else {
            vec![]
        }
    }
}

struct JobTracker {
    total: usize,
    completed: usize,
}

impl JobTracker {
    fn new(db: &sled::Tree) -> Self {
        Self {
            total: db.len(),
            completed: db
                .iter()
                .map(Result::unwrap)
                .filter(|(_, v)| !v.is_empty())
                .count(),
        }
    }

    fn increment(&mut self) {
        self.completed += 1;
    }

    fn finished(&self) -> bool {
        self.total == self.completed
    }
}

impl std::fmt::Display for JobTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "jobs processed: {}/{}", self.completed, self.total)
    }
}

fn get_config_db(meta: &sled::Tree) -> Option<(usize, usize)> {
    if let (Some(initial_n), Some(target_n)) = (
        meta.get(DB_INITIAL_N_KEY).unwrap(),
        meta.get(DB_TARGET_N_KEY).unwrap(),
    ) {
        let [initial_n]: [u8; 1] = initial_n.as_ref().try_into().unwrap();
        let [target_n]: [u8; 1] = target_n.as_ref().try_into().unwrap();
        Some((initial_n.into(), target_n.into()))
    } else {
        None
    }
}

fn get_config_cli(args: &mut pico_args::Arguments, meta: &sled::Tree) -> (usize, usize) {
    let initial_n: u8 = args.free_from_str().expect("Error parsing initial_n");
    let target_n: u8 = args.free_from_str().expect("Error parsing target_n");
    meta.insert(DB_INITIAL_N_KEY, &[initial_n]).unwrap();
    meta.insert(DB_TARGET_N_KEY, &[target_n]).unwrap();
    (initial_n.into(), target_n.into())
}
