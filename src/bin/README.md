This folder contains the code for distributed processing. To avoid pulling in more dependencies than necessary, each binary requires enabling a feature flag with the same name to use them.

To run the server:

```cargo run -r --bin server --features server```

The server currently does **NOT** validate anything, so don't expose it to any untrusted input.