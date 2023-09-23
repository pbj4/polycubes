use {
    ndarray::Dimension,
    std::{collections::HashSet, ops::IndexMut},
};

#[allow(dead_code)]
pub fn solve(n: usize) {
    let cube = Polycube::new(n);
    let mut results = HashSet::new();

    recurse(cube, &mut results);

    println!("results: {:?}", results);
    println!("count: {}", results.len());

    fn recurse(mut cube: Polycube, out: &mut HashSet<ndarray::Array3<bool>>) {
        if !out.insert(cube.array_ref().clone()) {
            return;
        }

        assert_eq!(cube.array_ref().iter().filter(|&&b| b).count(), cube.n);

        println!("array: {:?}", cube.array_ref());
        cube.remove_next();

        let backup_cube = cube.clone();

        for inner_candidate in cube.list_open_spots() {
            if cube.add(inner_candidate) {
                recurse(cube, out);
                cube = backup_cube.clone();
            }
        }

        // do outside extension

        for a in (0..3).map(ndarray::Axis) {
            for side in [true, false] {
                for outer_candidate in cube.list_edge_cubes(a, side) {
                    if cube.extend_add(a, outer_candidate, side) {
                        recurse(cube, out);
                        cube = backup_cube.clone();
                    }
                }
            }
        }
    }
}

// invariant: array is a bounding box of the contents
// dominant axis is x (0)
#[derive(Clone)]
struct Polycube {
    n: usize,
    array: ndarray::Array3<bool>,
    com: ndarray::Ix3,
    next_pop: ndarray::Ix3,
}

impl Polycube {
    const MAIN_AXIS: ndarray::Axis = ndarray::Axis(0);

    fn new(n: usize) -> Self {
        Self {
            n,
            array: ndarray::Array3::from_elem((n, 1, 1), true),
            com: ndarray::Ix3((0..n).sum(), 0, 0),
            next_pop: ndarray::Ix3(n - 1, 0, 0),
        }
    }

    fn remove_next(&mut self) -> bool {
        if self.next_pop[Self::MAIN_AXIS.index()] == 0 {
            return false;
        }

        self.array[self.next_pop] = false;
        self.com -= self.next_pop;
        //self.n -= 1;

        let pop_axis_index = self.next_pop.index_mut(Self::MAIN_AXIS.index());

        if self
            .array
            .index_axis(Self::MAIN_AXIS, *pop_axis_index)
            .iter()
            .all(|b| !b)
        {
            self.array.remove_index(Self::MAIN_AXIS, *pop_axis_index);
        }

        *pop_axis_index -= 1;

        true
    }

    fn add(&mut self, ix: ndarray::Ix3) -> bool {
        assert!(!self.array[ix]);

        let new_com = self.com + ix;

        if !Self::valid_com(new_com, self.array.raw_dim(), self.n) {
            return false;
        }

        self.array[ix] = true;
        self.com = new_com;
        //self.n += 1;

        true
    }

    fn list_open_spots(&self) -> Vec<ndarray::Ix3> {
        self.array
            .indexed_iter()
            .filter_map(|((x, y, z), b)| {
                if !b {
                    Some(ndarray::Ix3(x, y, z))
                } else {
                    None
                }
            })
            .filter(|&ix| {
                (0..3).any(|a| {
                    self.array
                        .get({
                            let mut ix = ix;
                            ix[a] = ix[a].wrapping_sub(1);
                            ix
                        })
                        .copied()
                        .unwrap_or(false)
                        || self
                            .array
                            .get({
                                let mut ix = ix;
                                ix[a] += 1;
                                ix
                            })
                            .copied()
                            .unwrap_or(false)
                })
            })
            .collect()
    }

    // side: true is before, false is after
    // ix: on axis a = 0
    fn extend_add(&mut self, a: ndarray::Axis, ix: ndarray::Ix3, side: bool) -> bool {
        debug_assert_eq!(ix[a.index()], 0);

        if a == Self::MAIN_AXIS && side {
            return false;
        }

        let mut new_dim = self.array.raw_dim();
        new_dim[a.index()] += 1;

        if !Self::valid_dimensions(new_dim) {
            return false;
        }

        let mut plane = ndarray::Array3::from_elem(
            {
                let mut d = self.array.raw_dim();
                d[a.index()] = 1;
                d
            },
            false,
        );
        plane[ix] = true;

        if side {
            // push before

            let mut new_com = self.com + ix;
            new_com[a.index()] += self.n - 1;

            if !Self::valid_com(new_com, new_dim, self.n) {
                return false;
            }

            self.com = new_com;
            //self.n += 1;

            plane.append(a, self.array.view()).unwrap();
            self.array = plane;

            self.next_pop[a.index()] += 1;
        } else {
            // push after

            let mut new_com = self.com + ix;
            new_com[a.index()] += self.array.raw_dim()[a.index()];
            //let new_n = self.n + 1;

            if !Self::valid_com(new_com, new_dim, self.n) {
                return false;
            }

            self.com = new_com;
            //self.n = new_n;

            self.array.append(a, plane.view()).unwrap();
        }

        true
    }

    fn list_edge_cubes(&self, a: ndarray::Axis, side: bool) -> Vec<ndarray::Ix3> {
        self.array
            .index_axis(
                a,
                if side {
                    0
                } else {
                    self.array.raw_dim()[a.index()] - 1
                },
            )
            .insert_axis(a)
            .indexed_iter()
            .filter_map(
                |((x, y, z), &b)| {
                    if b {
                        Some(ndarray::Ix3(x, y, z))
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    fn array_ref(&self) -> &ndarray::Array3<bool> {
        &self.array
    }

    fn valid_dimensions(d: ndarray::Ix3) -> bool {
        d[0] >= d[1] && d[1] >= d[2]
    }

    fn valid_com(com: ndarray::Ix3, d: ndarray::Ix3, n: usize) -> bool {
        let (dx, dy, dz) = d.into_pattern();
        let (cx, cy, cz) = com.into_pattern();

        match (dx == dy, dy == dz) {
            (false, false) => 4 * cy <= n * dy && 4 * cz <= n * dz,
            (true, false) | (false, true) => {
                4 * cx <= n * dx && 4 * cy <= n * dy && 4 * cz <= n * dz
            }
            (true, true) => {
                4 * cx <= n * dx && 4 * cy <= n * dy && cz * dx <= dz * cx && cz * dy <= dz * cy
            }
        }
    }
}

#[test]
fn test_polycube() {
    let calc_com = |a: &ndarray::Array3<bool>| {
        a.indexed_iter()
            .filter_map(|((x, y, z), b)| {
                if *b {
                    Some(ndarray::Ix3(x, y, z))
                } else {
                    None
                }
            })
            .reduce(|a, b| a + b)
            .unwrap()
    };

    let mut cube = Polycube::new(4);
    assert_eq!(cube.array_ref().raw_dim(), ndarray::Ix3(4, 1, 1));
    assert_eq!(cube.next_pop, ndarray::Ix3(3, 0, 0));
    assert_eq!(cube.com, calc_com(cube.array_ref()));

    assert!(cube.remove_next());
    assert_eq!(cube.array_ref().raw_dim(), ndarray::Ix3(3, 1, 1));
    assert_eq!(cube.com, calc_com(cube.array_ref()));
    assert_eq!(cube.n, 3);
    assert_eq!(cube.next_pop, ndarray::Ix3(2, 0, 0));

    assert!(cube.extend_add(ndarray::Axis(1), ndarray::Ix3(2, 0, 0), false));
    assert_eq!(cube.com, calc_com(cube.array_ref()));
}
