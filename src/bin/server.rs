use {
    polycubes::serialization,
    tiny_http::{Method, Response, Server},
};

const LISTEN_ADDR: &str = "0.0.0.0:1529"; // this is just a random port
const INITIAL_N: usize = 10;
const TARGET_N: usize = 14;

fn main() {
    println!("opening database");

    let db = sled::open("serverdb").unwrap();
    let meta = db.open_tree("meta").unwrap();

    if !meta.contains_key(DB_FINISH_KEY).unwrap() {
        if !meta.contains_key(DB_INIT_KEY).unwrap() {
            println!("generating polycubes");

            polycubes::revisions::latest::solve_out(INITIAL_N, &|view| {
                db.insert(serialization::polycube::serialize(view), &[])
                    .unwrap();
            });
            db.flush().unwrap();

            meta.insert(DB_INIT_KEY, &[]).unwrap();
        }

        println!("listening on http://{LISTEN_ADDR}/ for job requests...");

        let http = Server::http(LISTEN_ADDR).unwrap();

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

        while let Ok(mut request) = http.recv() {
            match (request.url(), request.method()) {
                ("/work", Method::Get) => {
                    if let Some(polycube) = get_job() {
                        request
                            .respond(Response::new(
                                200.into(),
                                vec![],
                                serialization::job::serialize(polycube.as_ref(), TARGET_N)
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

                    let status_code = if let Some(old) = db.insert(polycube, counts).unwrap() {
                        if counts == old.as_ref() {
                            200
                        } else {
                            eprintln!(
                                "conflicting result:\n{old:?}\nvs\n{counts:?}\nfor\n{polycube:?}"
                            );
                            400
                        }
                    } else {
                        200
                    };

                    request.respond(Response::empty(status_code)).unwrap();
                }
                _ => request.respond(Response::empty(404)).unwrap(),
            }
        }

        let counts = db
            .iter()
            .map(Result::unwrap)
            .fold(vec![(0, 0); TARGET_N], |mut a, (_, c)| {
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
    }

    println!("results:");

    let counts = serialization::result::deserialize_counts(
        meta.get(DB_FINISH_KEY).unwrap().unwrap().as_ref(),
    );

    for (i, (r, p)) in counts.iter().enumerate() {
        let i = i + 1;
        println!("n: {:?}, r: {:?}, p: {:?}", i, r, p);
    }
}

const DB_INIT_KEY: &str = "db_init";
const DB_FINISH_KEY: &str = "db_finish";
