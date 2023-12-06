#!/bin/bash
set -e

tmpdir=$(mktemp -d)
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

echo "Removing temporary directory..."
rm -rf "$tmpdir"

echo "Finished"
echo "Run optimized binary with \`./target/$target/release/polycubes\`"