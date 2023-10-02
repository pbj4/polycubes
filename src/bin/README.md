This folder contains the code for distributed processing. To avoid pulling in more dependencies than necessary, each binary requires enabling a feature flag with the same name to use them.

---

To run the server on address ```127.0.0.1:1529``` with the initial set of cubes expanded to n = 6 and a target of n = 13:

```cargo run -r --bin server --features server -- 127.0.0.1:1529 6 13```

If you stop the server and resume it, it will load the saved initial_n and target_n so specifying it again is not required.

To restart a run, delete or rename the ```serverdb``` folder.

The server currently does **NOT** validate anything, so don't expose it to any untrusted input.

---

To run a client with a server running on localhost at port 1529:

```cargo run -r --bin client --features client -- http://127.0.0.1:1529/```

To limit the number of threads to 7:

```cargo run -r --bin client --features client -- http://127.0.0.1:1529/ 7```