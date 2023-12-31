# 3d and 4d rotation polycube enumerator

Based on the problem described in the [computerphile video](https://www.youtube.com/watch?v=g9n0a0644B4) and the discussions in the [associated repo](https://github.com/mikepound/opencubes).

This program finds the number of unique polycubes under rotation in 3d and 4d (includes reflections). Primarily uses the [hashless](https://github.com/mikepound/opencubes/issues/11) algorithm by presseyt and its [optimization](https://github.com/mikepound/opencubes/issues/49) by snowmanam2, but the rest of the code was mostly written independently from the community implementation.

## To run

Install the [rust](https://www.rust-lang.org/tools/install) toolchain

To list all numbers of polycubes from 1 to 12 run:

```
cargo run -r 12
```

To limit the number of threads to 7 run:

```
cargo run -r 12 7
```

## Performance

Tested on a Ryzen 7 4600U

| n | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| time | 0.47s | 3.5s | 27s | 3m59s | 32m34s | 4h8m |

## Concepts used

* Hashless algorithm for efficient traversal of the graph of polycubes without needing to maintain a duplicate set

* snowmanam2's algorithm for a faster check for whether a DFS branch should terminate

* A streaming [articulation point algorithm](https://people.cs.umass.edu/~mcgregor/papers/05-tcs.pdf) for a faster check for whether a cube is removable after adding another one (speeds up snowmanam2's algorithm)

    * The max number of neighbors a cube in a polycube can have is 6, so I use ```heapless::Vec``` for the inner disjoint sets to avoid separate allocations

    * The original graph before the addition of a cube always contains the same non articulation points unless a bridge is formed, so the update isn't always necessary

* My algorithm for determining 3d rotation invariant from 4d rotation invariant polycubes:

    * If a 4d rotation invariant polycube has a reflectional symmetry, it counts as one 3d rotation invariant polycube, else two

* My algorithm for canonicalizing polycubes based on their center of mass and the dimensions of their bounding box:

    * Properties that transformed polycubes must satisfy to be canonical:

        1. The dimensions of the smallest bounding box must be in the lowest sorted order

        2. The coordinates of the center of mass must be in the lowest sorted order possible while still preserving 1

        3. If multiple transformations of the polycube satisfy 1 and 2, choose the one with the lowest flattened bit representation

    * If the center of mass lies on one or more axes or a diagonal this results in more work, but my conjecture is that this case becomes rarer for larger polycubes

    * Transformations can be described by reflections on the x, y, and z planes followed by permutations of the x, y, and z axes

    * Checking whether a polycube has reflectional symmetry can be done simultaneously by recording the chirality of one transformation and checking whether any subsequent ones represent the same polycube but have a different chirality

        * The chirality of a transformation can be calculated as the xor of all reflections performed, including axis reflections and permutations that can't be rotated back to the identity

* ```ndarray::Array3<bool>``` and ```ndarray::ArrayView3<bool>``` to represent polycubes and transform and crop them in constant time without allocations

* ```thread_local! { static RefCell<Collection> }``` to easily reuse allocations on each thread

## Values found with the distributed system (src/bin/)

* n = 18 4d rotation: `1758309223457`

* n = 19 3d rotation: `27144143923583`

* n = 19 4d rotation: `13573319825615`

## Compiling with Profile Guided Optimization (thanks @HakMe2Deth)

First, install ```llvm-profdata``` through rustup:

```
rustup component add llvm-tools-preview
```

Then run the build script for the main binary or the client:

```
./build-pgo.sh
```

```
./client-build-pgo.sh
```

This has only been tested on linux on x86_64. On my machine, I get a ~15% performance boost compared to building normally, but your mileage may vary

For windows, see this [issue](https://github.com/pbj4/polycubes/issues/1)