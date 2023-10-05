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
                db.insert(serialization::polycube::serialize(view), &[])
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

        while let Ok(mut request) = http.recv() {
            match (request.url(), request.method()) {
                ("/work", Method::Get) => {
                    if let Some(polycube) = job_finder.next() {
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
                        job_tracker.increment();
                        print!("\r{job_tracker}");
                        std::io::stdout().lock().flush().unwrap();
                    }

                    let status_code = if old.is_empty() || counts == old.as_ref() {
                        200
                    } else {
                        400
                    };

                    request.respond(Response::empty(status_code)).unwrap();
                }
                _ => {}
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
                match (request.url(), request.method()) {
                    ("/work", Method::Get) => request.respond(Response::empty(204)).unwrap(),
                    ("/result", Method::Post) => request.respond(Response::empty(200)).unwrap(),
                    _ => {}
                }
            }
        });

        println!();
        println!("finish time: {:?}", now.elapsed())
    }

    println!("results:");

    let counts = serialization::result::deserialize_counts(
        meta.get(DB_FINISH_KEY).unwrap().unwrap().as_ref(),
    );

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

    fn next_linear(&mut self) -> Option<sled::IVec> {
        self.iter
            .as_mut()?
            .by_ref()
            .map(Result::unwrap)
            .find_map(|(k, v)| v.is_empty().then_some(k))
    }
}

impl Iterator for JobFinder {
    type Item = sled::IVec;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.next_linear() {
            Some(item)
        } else if let Some(iter) = &mut self.iter {
            *iter = self.db.iter();
            self.next_linear()
        } else {
            None
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
