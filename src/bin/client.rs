fn main() {
    let mut args = pico_args::Arguments::from_env();
    let url: String = args.free_from_str().expect("Error parsing server url");
    let url = url.strip_suffix('/').unwrap_or(&url);

    let work_url = url.to_owned() + "/work";
    let result_url = url.to_owned() + "/result";

    loop {
        match ureq::get(&work_url).call() {
            Ok(response) => {
                if response.status() == 204 {
                    println!("work finished");
                    break;
                }

                let mut job = Vec::new();
                response.into_reader().read_to_end(&mut job).unwrap();

                let (n, polycube) = polycubes::serialization::job::deserialize(&job);
                let polycube = polycubes::serialization::polycube::deserialize(polycube);

                let map = polycubes::revisions::latest::solve_partial(n, polycube.view());

                let result = polycubes::serialization::result::serialize(polycube.view(), &map);

                ureq::post(&result_url).send_bytes(&result).unwrap();
            }
            Err(err) => {
                eprintln!("error: {err:?}");
                break;
            }
        }
    }
}
