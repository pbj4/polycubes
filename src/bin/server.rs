use {
    nanoserde::{DeBin, SerBin},
    polycubes::serialization::*,
    std::{
        io::{BufRead, Write},
        time::{Duration, Instant, SystemTime},
    },
    tiny_http::{Method, Response, Server},
};

const CLIENT_TIMEOUT: Duration = Duration::from_secs(600);

fn main() {
    println!("opening database...");

    let db = sled::open("serverdb").unwrap();
    let meta = db.open_tree("meta").unwrap();

    if !meta.contains_key(DB_FINISH_KEY).unwrap() {
        let mut args = pico_args::Arguments::from_env();
        let listen_addr: String = args.free_from_str().expect("Error parsing listen address");

        print!("loading configuration from ");
        let start_data = StartData::from_db(&meta)
            .map(|start_data| {
                print!("db: ");
                start_data
            })
            .unwrap_or_else(|| {
                print!("cli: ");
                let start_data = StartData::from_cli(&mut args);
                start_data.set_db(&meta);
                start_data
            });

        println!(
            "initial_n = {}, target_n = {}",
            start_data.initial_n, start_data.target_n
        );

        if !meta.contains_key(DB_INIT_KEY).unwrap() {
            println!("generating initial polycubes...");

            polycubes::revisions::latest::solve_out(start_data.initial_n, &|view| {
                db.insert(SerPolycube::ser(view).as_slice(), &[]).unwrap();
            });
            db.flush().unwrap();

            meta.insert(DB_INIT_KEY, &[]).unwrap();
        }

        let http = Server::http(&listen_addr).unwrap();

        println!("listening on http://{listen_addr}/ for job requests...");

        let format_status =
            move |job_tracker: &JobTracker, client_tracker: &ClientTracker| -> String {
                format!(
                    "time elapsed: {:?}, {job_tracker}, {client_tracker}",
                    start_data.start_time().elapsed().unwrap()
                )
            };

        let job_finder = JobFinder::new(&db);
        let job_tracker = JobTracker::new(&db);
        let client_tracker = ClientTracker::new();
        let mut overwriter = polycubes::Overwriter::default();
        overwriter.print(format_status(&job_tracker, &client_tracker));

        let finish = spawn_job_server(
            http,
            (*db).clone(),
            start_data,
            format_status,
            job_finder,
            job_tracker,
            client_tracker,
            overwriter,
        );

        let mutex = std::sync::Mutex::<()>::default();
        let _guard = finish.wait(mutex.lock().unwrap()).unwrap();

        let results: Results = db
            .iter()
            .map(Result::unwrap)
            .map(|(_, v)| SerResults::from_slice(&v).de())
            .sum();

        meta.insert(
            DB_FINISH_KEY,
            FinishData::from_results(results).serialize_bin(),
        )
        .unwrap();
        meta.flush().unwrap();

        println!();
    }

    let start_data = StartData::from_db(&meta).unwrap();

    let finish_data =
        FinishData::deserialize_bin(&meta.get(DB_FINISH_KEY).unwrap().unwrap()).unwrap();

    println!(
        "total time elapsed: {:?}",
        finish_data
            .stop_time()
            .duration_since(start_data.start_time())
            .unwrap()
    );

    println!("results:\n{}", finish_data.results);

    print!("press enter to exit...");
    std::io::stdout().lock().flush().unwrap();
    std::io::stdin()
        .lock()
        .read_line(&mut String::new())
        .unwrap();
}

const DB_START_KEY: &str = "db_start";
const DB_INIT_KEY: &str = "db_init";
const DB_FINISH_KEY: &str = "db_finish";

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
    recent: Vec<(Results, Instant)>,
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
            recent: Vec::new(),
        }
    }

    fn process(&mut self, result: Results) {
        self.completed += 1;
        self.recent.push((result, Instant::now()));
        self.recent.retain(|(_, i)| i.elapsed() < CLIENT_TIMEOUT);
    }

    fn finished(&self) -> bool {
        self.total == self.completed
    }
}

impl std::fmt::Display for JobTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let duration = self
            .recent
            .first()
            .map(|(_, i)| i.elapsed())
            .unwrap_or(CLIENT_TIMEOUT);
        let (rs, ps) = self
            .recent
            .iter()
            .map(|(r, _)| r)
            .cloned()
            .sum::<Results>()
            .average_rate(duration);
        write!(
            f,
            "jobs done: {}/{}, r/s: {}, p/s: {}",
            self.completed, self.total, rs, ps
        )
    }
}

#[derive(SerBin, DeBin, Clone, Copy)]
struct StartData {
    initial_n: usize,
    target_n: usize,
    start_time: u128,
}

impl StartData {
    fn from_db(meta: &sled::Tree) -> Option<Self> {
        meta.get(DB_START_KEY)
            .unwrap()
            .map(|start_data| Self::deserialize_bin(&start_data).unwrap())
    }

    fn from_cli(args: &mut pico_args::Arguments) -> Self {
        let initial_n = args.free_from_str().expect("Error parsing initial_n");
        let target_n = args.free_from_str().expect("Error parsing target_n");
        Self {
            initial_n,
            target_n,
            start_time: SystemTime::UNIX_EPOCH.elapsed().unwrap().as_millis(),
        }
    }

    fn set_db(&self, meta: &sled::Tree) {
        meta.insert(DB_START_KEY, self.serialize_bin()).unwrap();
    }

    fn start_time(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_millis(self.start_time.try_into().unwrap())
    }
}

#[derive(SerBin, DeBin)]
struct FinishData {
    results: Results,
    stop_time: u128,
}

impl FinishData {
    fn from_results(results: Results) -> Self {
        Self {
            results,
            stop_time: SystemTime::UNIX_EPOCH.elapsed().unwrap().as_millis(),
        }
    }

    fn stop_time(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_millis(self.stop_time.try_into().unwrap())
    }
}

struct ClientTracker {
    clients: std::collections::HashMap<std::net::IpAddr, Instant>,
}

impl ClientTracker {
    fn new() -> Self {
        Self {
            clients: std::collections::HashMap::new(),
        }
    }

    fn seen(&mut self, addr: std::net::IpAddr) {
        self.clients.insert(addr, Instant::now());
        self.clients.retain(|_, i| i.elapsed() < CLIENT_TIMEOUT);
    }
}

impl std::fmt::Display for ClientTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "active clients: {}", self.clients.len())
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_job_server(
    http: Server,
    db: sled::Tree,
    start_data: StartData,
    format_status: impl Fn(&JobTracker, &ClientTracker) -> String + Send + 'static,
    mut job_finder: JobFinder,
    mut job_tracker: JobTracker,
    mut client_tracker: ClientTracker,
    mut overwriter: polycubes::Overwriter,
) -> std::sync::Arc<std::sync::Condvar> {
    let finish = std::sync::Arc::new(std::sync::Condvar::new());

    {
        let finish = finish.clone();
        std::thread::spawn(move || {
            'handle: while let Ok(mut request) = http.recv() {
                if let ("/work", Method::Post) = (request.url(), request.method()) {
                    client_tracker.seen(request.remote_addr().unwrap().ip());

                    // handle result
                    let mut job_request = Vec::new();
                    request.as_reader().read_to_end(&mut job_request).unwrap();
                    let job_request = JobRequest::deserialize_bin(&job_request).unwrap();
                    let mut total_results: Option<Results> = None;

                    for (polycube, result) in job_request.results {
                        let old = db
                            .insert(polycube.as_slice(), result.as_slice())
                            .unwrap()
                            .unwrap();

                        if old.is_empty() {
                            if let Some(total_results) = total_results.as_mut() {
                                *total_results += result.de();
                            } else {
                                total_results = Some(result.de());
                            }
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

                    if let Some(total_results) = total_results {
                        job_tracker.process(total_results);
                    }
                    overwriter.print(format_status(&job_tracker, &client_tracker));

                    // assign job
                    let job_response = JobResponse {
                        target_n: start_data.target_n,
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
