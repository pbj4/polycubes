This folder contains all the old versions of the code before I started using version control so you can see my thought process while writing this program.

Each file is completely self contained and can be swapped in in main.rs by editing ```revisions::latest::solve(n)``` to ```revisions::solve6x::solve(n)```. The latest one is guaranteed to work, but others may be subtly broken or just incorrect.

To reduce the number of dependencies pulled in by default, hash related dependencies are gated behind the ```hash``` feature. To enable them when running old versions:

```cargo run -r --features hash -- 10```

The "solve" part doesn't actually mean anything, it's just what I named the files for some reason.

The rest of development will happen in solve6z, which is the final version.

## Milestones

* solve6: first working version, only counts 4d rotation polycubes

* solve6f: first parallel version using sharded hashmap

* solve6g: implemented counting both 3d and 4d rotation polycubes

* solve6i: started using ndarray::ArrayView3

* solve6k: implemented Hashless algorithm

* solve6l: implemented Tarjan's algorithm

* solve6m: almost independently rediscovered snowmanam2's algorithm but failed

* solve6o: implemented direct calculation of canonical transformation

* solve6p: implemented snowmanam2's algorithm 

* solve6v: implemented the streaming articulation point algorithm

* solve6y: last version before open source release