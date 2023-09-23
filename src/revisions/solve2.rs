use {
    ndarray::Dimension,
    std::{
        collections::HashSet,
        ops::{Index, IndexMut},
    },
};

#[allow(dead_code)]
pub fn solve(n: usize) {
    // start with 1x1xN rod

    let rod = ndarray::Array3::from_elem((n, 1, 1), true);

    let mut results = HashSet::new();

    fn recurse(
        mut array: ndarray::Array3<bool>,
        mut com: CenterOfMass,
        swap_from: ndarray::Ix3,
        out: &mut HashSet<ndarray::Array3<bool>>,
    ) {
        if swap_from[0] == 0 {
            return;
        }

        if !out.insert(array.clone()) {
            return;
        }

        // remove cube from array
        array[swap_from] = false;

        // removed cube from com
        com.remove(swap_from);

        // try shrinking array
        if array
            .index_axis(ndarray::Axis(0), *swap_from.index(0))
            .iter()
            .all(|b| !b)
        {
            array.remove_index(ndarray::Axis(0), *swap_from.index(0));
        }

        // updating next cube to be swapped
        let new_swap_from = apply_axis(swap_from, ndarray::Axis(0), |x| *x -= 1);

        // check inner swaps, same dimensions
        for swap_to in connected_spots(&array).filter(|ix| *ix != swap_from) {
            let mut new_com = com;
            new_com.add(swap_to);

            if !new_com.in_valid_zone(array.raw_dim()) {
                continue;
            }

            let mut new = array.clone();
            new[swap_to] = true;

            recurse(new, new_com, new_swap_from, out);
        }

        // check outer swaps, new dimensions
        for (axis, next_size) in next_valid_bounding_box(array.raw_dim()) {
            let new = ndarray::Array3::from_elem(next_size, false);

            // disallow extending x in negative direction
            if axis != ndarray::Axis(0) {
                for ((x, y, z), &b) in array.index_axis(axis, 0).insert_axis(axis).indexed_iter() {
                    let ix = ndarray::Ix3(x, y, z);

                    if b {
                        let mut new_com = com;
                        new_com.shift_all(axis);
                        new_com.add(ix);

                        if !new_com.in_valid_zone(next_size) {
                            continue;
                        }

                        let mut new = new.clone();
                        let (mut front, mut back) = new.view_mut().split_at(axis, 1);
                        front[ix] = true;
                        back.assign(&array);

                        let new_swap_from = apply_axis(new_swap_from, axis, |x| *x += 1);

                        recurse(new, new_com, new_swap_from, out);
                    }
                }
            }

            for ((x, y, z), &b) in array
                .index_axis(axis, array.raw_dim()[axis.index()] - 1)
                .insert_axis(axis)
                .indexed_iter()
            {
                let ix = ndarray::Ix3(x, y, z);

                if b {
                    let mut new_com = com;
                    new_com.add(ix);

                    if !new_com.in_valid_zone(next_size) {
                        continue;
                    }

                    let mut new = new.clone();
                    let (mut front, mut back) =
                        new.view_mut().split_at(axis, array.raw_dim()[axis.index()]);
                    front.assign(&array);
                    back[ix] = true;

                    recurse(new, new_com, new_swap_from, out);
                }
            }
        }
    }

    let com = CenterOfMass::new(&rod);
    recurse(rod, com, ndarray::Ix3(n - 1, 0, 0), &mut results);

    println!("results: {:?}", results);
    println!("count: {}", results.len());
}

#[derive(Clone, Copy, Debug)]
struct CenterOfMass {
    n: usize,
    ix: ndarray::Ix3,
}

impl CenterOfMass {
    fn new(array: &ndarray::Array3<bool>) -> Self {
        let mut n = 0;
        let ix = array
            .indexed_iter()
            .filter_map(|((x, y, z), b)| {
                if *b {
                    Some(ndarray::Ix3(x, y, z))
                } else {
                    None
                }
            })
            .inspect(|_| n += 1)
            .reduce(|a, b| a + b)
            .unwrap();

        Self { n, ix }
    }

    fn remove(&mut self, ix: ndarray::Ix3) {
        self.ix -= ix;
        self.n -= 1;
    }

    fn add(&mut self, ix: ndarray::Ix3) {
        self.ix += ix;
        self.n += 1;
    }

    fn shift_all(&mut self, a: ndarray::Axis) {
        self.ix[a.index()] += self.n;
    }

    fn in_1_4(&self, dim: ndarray::Ix3) -> bool {
        (1..3).all(|a| 4 * self.ix[a] <= dim[a] * self.n)
    }

    fn in_1_8(&self, dim: ndarray::Ix3) -> bool {
        (0..3).all(|a| 4 * self.ix[a] <= dim[a] * self.n)
    }

    fn in_1_24(&self, dim: ndarray::Ix3) -> bool {
        (0..2).all(|a| 4 * self.ix[a] <= dim[a] * self.n)
            && self.ix[2] * dim[0] <= self.ix[0] * dim[2]
            && self.ix[2] * dim[1] <= self.ix[1] * dim[2]
    }

    // assumes dim is in canonical order
    // TODO: signal ambiguity at edge cases
    fn in_valid_zone(&self, dim: ndarray::Ix3) -> bool {
        match (dim[0] == dim[1], dim[1] == dim[2]) {
            (false, false) => self.in_1_4(dim),
            (true, false) | (false, true) => self.in_1_8(dim),
            (true, true) => self.in_1_24(dim),
        }
    }
}

fn apply_axis(mut ix3: ndarray::Ix3, a: ndarray::Axis, f: impl Fn(&mut usize)) -> ndarray::Ix3 {
    f(ix3.index_mut(a.index()));
    ix3
}

fn connected_spots(array: &ndarray::Array3<bool>) -> impl Iterator<Item = ndarray::Ix3> + '_ {
    array.indexed_iter().filter_map(|((x, y, z), b)| {
        let ix = ndarray::Ix3(x, y, z);
        if !b
            && all_axis().any(|a| {
                array
                    .get(apply_axis(ix, a, |x| *x += 1))
                    .copied()
                    .unwrap_or(false)
                    || if ix[a.index()] == 0 {
                        false
                    } else {
                        array
                            .get(apply_axis(ix, a, |x| *x -= 1))
                            .copied()
                            .unwrap_or(false)
                    }
            })
        {
            Some(ix)
        } else {
            None
        }
    })
}

fn next_valid_bounding_box(
    dim: ndarray::Ix3,
) -> impl Iterator<Item = (ndarray::Axis, ndarray::Ix3)> {
    all_axis().filter_map(move |a| {
        let dim = apply_axis(dim, a, |x| *x += 1);

        if valid_bounding_box(&dim) {
            Some((a, dim))
        } else {
            None
        }
    })
}

fn all_axis() -> impl Iterator<Item = ndarray::Axis> {
    (0..3).map(ndarray::Axis)
}

fn valid_bounding_box(dim: &ndarray::Ix3) -> bool {
    let (x, y, z) = dim.into_pattern();
    x >= y && y >= z
}
