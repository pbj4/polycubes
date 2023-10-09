use {
    nanoserde::{DeBin, SerBin},
    polycubes::serialization::*,
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
        let Parameters {
            initial_n,
            target_n,
        } = Parameters::from_db(&meta)
            .map(|params| {
                print!("db: ");
                params
            })
            .unwrap_or_else(|| {
                print!("cli: ");
                let params = Parameters::from_cli(&mut args);
                params.set_db(&meta);
                params
            });

        println!("initial_n = {initial_n}, target_n = {target_n}");

        if !meta.contains_key(DB_INIT_KEY).unwrap() {
            println!("generating initial polycubes");

            polycubes::revisions::latest::solve_out(initial_n, &|view| {
                db.insert(SerPolycube::ser(view).as_slice(), &[]).unwrap();
            });
            db.flush().unwrap();

            meta.insert(DB_INIT_KEY, &[]).unwrap();
        }

        let now = std::time::Instant::now();

        let http = Server::http(&listen_addr).unwrap();

        println!("listening on http://{listen_addr}/ for job requests...");

        let job_finder = JobFinder::new(&db);
        let job_tracker = JobTracker::new(&db);
        print!("\r{job_tracker}");
        std::io::stdout().lock().flush().unwrap();

        let finish = spawn_job_server(http, (*db).clone(), target_n, job_finder, job_tracker);

        let mutex = std::sync::Mutex::<()>::default();
        let _guard = finish.wait(mutex.lock().unwrap()).unwrap();

        let results: Results = db
            .iter()
            .map(Result::unwrap)
            .map(|(_, v)| SerResults::from_slice(&v).de())
            .sum();

        meta.insert(DB_FINISH_KEY, results.ser().serialize_bin())
            .unwrap();
        meta.flush().unwrap();

        println!("\nfinish time: {:?}", now.elapsed())
    }

    println!("results:");

    let results = SerResults::deserialize_bin(&meta.get(DB_FINISH_KEY).unwrap().unwrap())
        .unwrap()
        .de();

    let Parameters { initial_n, .. } = Parameters::from_db(&meta).unwrap();

    for (i, (r, p)) in results.counts_slice().iter().enumerate() {
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
const DB_PARAMETERS_KEY: &str = "db_parameters";

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

#[derive(nanoserde::SerBin, nanoserde::DeBin)]
struct Parameters {
    initial_n: usize,
    target_n: usize,
}

impl Parameters {
    fn from_db(meta: &sled::Tree) -> Option<Self> {
        meta.get(DB_PARAMETERS_KEY)
            .unwrap()
            .map(|param| Self::deserialize_bin(&param).unwrap())
    }

    fn from_cli(args: &mut pico_args::Arguments) -> Self {
        let initial_n = args.free_from_str().expect("Error parsing initial_n");
        let target_n = args.free_from_str().expect("Error parsing target_n");
        Self {
            initial_n,
            target_n,
        }
    }

    fn set_db(&self, meta: &sled::Tree) {
        meta.insert(DB_PARAMETERS_KEY, self.serialize_bin())
            .unwrap();
    }
}

fn spawn_job_server(
    http: Server,
    db: sled::Tree,
    target_n: usize,
    mut job_finder: JobFinder,
    mut job_tracker: JobTracker,
) -> std::sync::Arc<std::sync::Condvar> {
    let finish = std::sync::Arc::new(std::sync::Condvar::new());

    {
        let finish = finish.clone();
        std::thread::spawn(move || {
            'handle: while let Ok(mut request) = http.recv() {
                if let ("/work", Method::Post) = (request.url(), request.method()) {
                    // handle result
                    let mut job_request = Vec::new();
                    request.as_reader().read_to_end(&mut job_request).unwrap();
                    let job_request = JobRequest::deserialize_bin(&job_request).unwrap();

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
                    let job_response = JobResponse {
                        target_n,
                        jobs: job_finder
                            .assign(job_request.jobs_wanted)
                            .into_iter()
                            .map(|iv| SerPolycube::from_slice(&iv))
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
                        finish.notify_all();
                    }
                }
            }
        });
    }

    finish
}
