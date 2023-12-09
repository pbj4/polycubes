#!/bin/bash
set -e

tmpdir="$(pwd)/target/pgo-data/"
rm -rf "$tmpdir"
mkdir -p "$tmpdir"
target=$(rustc -vV | sed -n 's|host: ||p')
EXTRA_RUSTFLAGS="--C target-cpu=native --C opt-level=3 --C lto=yes --C embed-bitcode=y --C codegen-units=1 --C code-model=small --C debuginfo=0"

echo "Target: $target"
echo "Using $tmpdir as temporary directory"

echo "Building instrumented binary..."
RUSTFLAGS="-Cprofile-generate=$tmpdir $EXTRA_RUSTFLAGS" cargo build --release --target=$target

echo "Gathering profiles..."
./target/$target/release/polycubes 10
./target/$target/release/polycubes 11
./target/$target/release/polycubes 12

echo "Merging profiles..."
~/.rustup/toolchains/stable-$target/lib/rustlib/$target/bin/llvm-profdata merge -o "$tmpdir/merged.profdata" "$tmpdir"

echo "Building optimized binary..."
RUSTFLAGS="-Cprofile-use=$tmpdir/merged.profdata $EXTRA_RUSTFLAGS" cargo build --release --target=$target

echo "Finished"
echo "Run optimized binary with \`./target/$target/release/polycubes\`"