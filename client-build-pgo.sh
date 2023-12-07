#!/bin/bash
set -e

tmpdir=$(mktemp -d)
target=$(rustc -vV | sed -n 's|host: ||p')
EXTRA_RUSTFLAGS="--C target-cpu=native --C opt-level=3 --C lto=yes --C embed-bitcode=y --C codegen-units=1 --C code-model=small --C debuginfo=0"

echo "Target: $target"
echo "Using $tmpdir as temporary directory"

echo "Building instrumented client..."
RUSTFLAGS="-Cprofile-generate=$tmpdir $EXTRA_RUSTFLAGS" cargo build --release --target=$target --bin client --features client

echo "Building server..."
cargo build -r --bin server --features server

echo "Gathering profiles..."

function test(){
    rm -rf serverdb
    ./target/release/server 127.0.0.1:48479 1 $1 &
    local serverpid="$!"
    sleep 1
    ./target/$target/release/client http://127.0.0.1:48479/
    kill $serverpid
}

test 10
test 11
test 12

echo "Merging profiles..."
~/.rustup/toolchains/stable-$target/lib/rustlib/$target/bin/llvm-profdata merge -o "$tmpdir/merged.profdata" "$tmpdir"

echo "Building optimized client..."
RUSTFLAGS="-Cprofile-use=$tmpdir/merged.profdata $EXTRA_RUSTFLAGS" cargo build --release --target=$target --bin client --features client

echo "Removing temporary directory..."
rm -rf "$tmpdir"

echo "Finished"
echo "Run optimized client with \`./target/$target/release/client\`"