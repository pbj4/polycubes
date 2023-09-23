use {
    ndarray::Dimension,
    std::{collections::HashSet, ops::IndexMut},
};

#[allow(dead_code)]
pub fn solve(n: usize) {
    let cube = PolycubeNormal::new(n);
    let mut results = HashSet::new();

    recurse(cube, &mut results);

    println!("results:");
    for a in &results {
        println!("{:?}\n----", a);
    }
    println!("count: {}", results.len());

    fn recurse(cube: PolycubeNormal, out: &mut HashSet<ndarray::Array3<bool>>) {
        if !out.insert(cube.0.array_ref().clone()) {
            return;
        }

        println!("array: {:?}", cube.0.array_ref());
        debug_assert_eq!(cube.0.array_ref().iter().filter(|&&b| b).count(), cube.0.n);
        cube.0.check_com_invariant();

        let Some(mut cube) = cube.remove_next() else {
            return;
        };
        let backup_cube = cube.clone();

        for inner in cube.list_open_inner() {
            match cube.add(inner, out.hasher()) {
                Ok(swapped) => {
                    recurse(swapped, out);
                    cube = backup_cube.clone()
                }
                Err(unswapped) => cube = unswapped,
            }
        }

        for a in (0..3).map(ndarray::Axis) {
            for s in [true, false] {
                for outer in cube.list_edge_outer(a, s) {
                    match cube.extend_add(a, outer, s, out.hasher()) {
                        Ok(swapped) => {
                            recurse(swapped, out);
                            cube = backup_cube.clone();
                        }
                        Err(unswapped) => {
                            cube = unswapped;
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
struct PolycubeBase {
    n: usize,
    array: ndarray::Array3<bool>,
    com: ndarray::Ix3,
    next_pop: ndarray::Ix3,
}

impl PolycubeBase {
    const MAIN_AXIS: ndarray::Axis = ndarray::Axis(0);

    fn valid_dimensions(d: ndarray::Ix3) -> bool {
        d[0] >= d[1] && d[1] >= d[2]
    }

    pub fn valid_com3(com: ndarray::Ix3, d: ndarray::Ix3, n: usize) -> ComPosition {
        let (dx, dy, dz) = (d - ndarray::Ix3(1, 1, 1)).into_pattern();
        let (cx, cy, cz) = com.into_pattern();

        let xmid = (2 * cx).cmp(&(n * dx));
        let ymid = (2 * cy).cmp(&(n * dy));
        let zmid = (2 * cz).cmp(&(n * dz));

        let xydiag = (cy * dx).cmp(&(dy * cx));
        let xzdiag = (cz * dx).cmp(&(dz * cx));
        let yzdiag = (cz * dy).cmp(&(dz * cy));

        use std::cmp::Ordering::*;

        let result = match (
            (dx == dy, dy == dz),
            (xmid, ymid, zmid),
            (xydiag, xzdiag, yzdiag),
        ) {
            ((false, false), (_, Greater, _) | (_, _, Greater), _) => ComPosition::Invalid,
            ((false, false), (Equal, Equal, Equal), _) => ComPosition::Boundary(RotationAxis::Zero),
            ((false, false), (Less, Equal, Equal), _) => ComPosition::Boundary(RotationAxis::X),
            ((false, false), (Equal, Less, Equal), _) => ComPosition::Boundary(RotationAxis::Y),
            ((false, false), (Equal, Equal, Less), _) => ComPosition::Boundary(RotationAxis::Z),
            ((false, false), (_, Less, Less) | (Less, Equal, _) | (Less, _, Equal), _) => {
                ComPosition::Valid
            }
            ((false, false), _, _) => ComPosition::Invalid,

            ((true, false), (Greater, _, _) | (_, Greater, _) | (_, _, Greater), _) => {
                ComPosition::Invalid
            }
            ((true, false), (Equal, Equal, Equal), _) => ComPosition::Boundary(RotationAxis::Zero),
            ((true, false), (Less, Equal, Equal), _) => ComPosition::Boundary(RotationAxis::X),
            ((true, false), (Equal, Less, Equal), _) => ComPosition::Invalid,
            ((true, false), (Equal, Equal, Less), _) => ComPosition::Boundary(RotationAxis::Z),
            ((true, false), (_, _, Equal), (Equal, _, _)) => {
                ComPosition::Boundary(RotationAxis::XY)
            }
            ((true, false), (Equal, _, _), _) => ComPosition::Valid,
            ((true, false), (_, Equal, _), _) => ComPosition::Invalid,
            ((true, false), (_, _, Equal), (Less, _, _)) => ComPosition::Valid,
            ((true, false), _, _) => ComPosition::Invalid,

            ((false, true), (Greater, _, _) | (_, Greater, _) | (_, _, Greater), _) => {
                ComPosition::Invalid
            }
            ((false, true), (Equal, Equal, Equal), _) => ComPosition::Boundary(RotationAxis::Zero),
            ((false, true), (Less, Equal, Equal), _) => ComPosition::Boundary(RotationAxis::X),
            ((false, true), (Equal, Less, Equal), _) => ComPosition::Invalid,
            ((false, true), (Equal, Equal, Less), _) => ComPosition::Boundary(RotationAxis::Z),
            ((false, true), (Equal, _, _), (_, _, Equal)) => {
                ComPosition::Boundary(RotationAxis::YZ)
            }
            ((false, true), (Equal, _, _), (_, _, Less)) => ComPosition::Valid,
            ((false, true), (_, Equal, _), _) => ComPosition::Invalid,
            ((false, true), (_, _, Equal), _) => ComPosition::Valid,
            ((false, true), _, _) => ComPosition::Invalid,

            ((true, true), (Greater, _, _) | (_, Greater, _) | (_, _, Greater), _) => {
                ComPosition::Invalid
            }
            ((true, true), _, (_, Greater, _) | (_, _, Greater)) => ComPosition::Invalid,
            ((true, true), (Equal, Equal, Equal), _) => ComPosition::Boundary(RotationAxis::Zero),
            ((true, true), (Equal, Equal, Less), _) => ComPosition::Boundary(RotationAxis::Z),
            ((true, true), (Equal, _, _), (_, _, Equal)) => ComPosition::Boundary(RotationAxis::YZ),
            ((true, true), _, (Equal, Equal, Equal)) => ComPosition::Boundary(RotationAxis::XYZ),
            ((true, true), (Equal, _, _), _) => ComPosition::Valid,
            ((true, true), _, (Less, _, Equal)) => ComPosition::Valid,
            ((true, true), _, (_, Less, Less)) => ComPosition::Valid,
            ((true, true), _, _) => ComPosition::Invalid,
        };

        if let ComPosition::Boundary(raxis) = &result {
            match raxis {
                RotationAxis::Zero => {
                    debug_assert!(xmid == Equal && ymid == Equal && zmid == Equal)
                }
                RotationAxis::X => {
                    debug_assert!(xmid == Less && ymid == Equal && zmid == Equal)
                }
                RotationAxis::Y => {
                    debug_assert!(xmid == Equal && ymid == Less && zmid == Equal)
                }
                RotationAxis::Z => {
                    debug_assert!(xmid == Equal && ymid == Equal && zmid == Less)
                }
                RotationAxis::XY => {
                    debug_assert!(xydiag == Equal)
                }
                RotationAxis::YZ => {
                    debug_assert!(yzdiag == Equal)
                }
                RotationAxis::XZ => {
                    debug_assert!(xzdiag == Equal)
                }
                RotationAxis::XYZ => {
                    debug_assert!(xydiag == Equal && yzdiag == Equal && xzdiag == Equal)
                }
            }
        }

        result
    }

    // FIXME: nondeterministic???
    fn rotate_canonical(&mut self, r: RotationAxis, h: &impl std::hash::BuildHasher) {
        let (dx, dy, dz) = self.array.raw_dim().into_pattern();
        let old_com = self.com;

        let mut best_orientation = self.array.view();
        let mut lowest_hash = h.hash_one(best_orientation);

        // 1 indexed so -0 can be represented
        let mut try_rotation = |transform: [isize; 3]| {
            let mut rotated = self.array.view();

            rotated.permuted_axes(transform.map(|x| x.unsigned_abs() - 1));
            for (i, t) in transform.into_iter().enumerate() {
                if t.is_negative() {
                    rotated.invert_axis(ndarray::Axis(i));
                }
            }

            debug_assert_eq!((dx, dy, dz), rotated.raw_dim().into_pattern());
            debug_assert_eq!(PolycubeBase::compute_com(&rotated), old_com);

            let hash = h.hash_one(rotated);

            if hash < lowest_hash {
                lowest_hash = hash;
                best_orientation = rotated;
            }
        };

        match r {
            RotationAxis::X => {
                // rotate 180
                try_rotation([1, -2, -3]);

                if dy == dz {
                    // rotate 90 and 270
                    try_rotation([1, -3, 2]);
                    try_rotation([1, 3, -2]);
                }
            }
            RotationAxis::Y => {
                try_rotation([-1, 2, -3]);

                if dx == dz {
                    try_rotation([-3, 2, 1]);
                    try_rotation([3, 2, -1]);
                }
            }
            RotationAxis::Z => {
                try_rotation([-1, -2, 3]);

                if dx == dy {
                    try_rotation([-2, 1, 3]);
                    try_rotation([2, -1, 3]);
                }
            }
            RotationAxis::XY => {
                try_rotation([2, 1, -3]);
            }
            RotationAxis::XZ => {
                try_rotation([3, -2, 1]);
            }
            RotationAxis::YZ => {
                try_rotation([-1, 3, 2]);
            }
            RotationAxis::XYZ => {
                try_rotation([2, 3, 1]);
                try_rotation([3, 1, 2]);
            }
            RotationAxis::Zero => {
                for swap_xy in [false, true] {
                    for (pos_x, pos_y) in
                        [(true, true), (false, true), (true, false), (false, false)]
                    {
                        let pos_z = swap_xy != (pos_x == pos_y);

                        let x: isize = if pos_x { 1 } else { -1 };
                        let y: isize = if pos_z { 2 } else { -2 };
                        let z: isize = if pos_z { 3 } else { -3 };
                        let (x, y) = if swap_xy { (y, x) } else { (x, y) };

                        // iterate on all 3 rotations
                        for (x, y, z) in [(x, y, z), (y, z, x), (z, x, y)] {
                            // check if rotation leaves dimensions the same
                            let dxyz = [dx, dy, dz];

                            if dx == dxyz[x.unsigned_abs() - 1]
                                && dy == dxyz[y.unsigned_abs() - 1]
                                && dz == dxyz[y.unsigned_abs() - 1]
                            {
                                try_rotation([x, y, z]);
                            }
                        }
                    }
                }
            }
        }

        self.array = best_orientation.to_owned();

        self.check_com_invariant();
    }

    fn compute_com(a: &ndarray::ArrayView3<bool>) -> ndarray::Ix3 {
        a.indexed_iter()
            .filter_map(|((x, y, z), &b)| if b { Some(ndarray::Ix3(x, y, z)) } else { None })
            .reduce(|a, b| a + b)
            .unwrap()
    }

    fn get_neighbors(
        a: &ndarray::Array3<bool>,
        ix: ndarray::Ix3,
    ) -> impl Iterator<Item = (ndarray::Ix3, bool)> {
        {
            let mut ix = ix;
            ix[0] += 1;
            a.get(ix).map(|&b| (ix, b)).into_iter()
        }
        .chain({
            let mut ix = ix;
            ix[0] = ix[0].wrapping_sub(1);
            a.get(ix).map(|&b| (ix, b))
        })
        .chain({
            let mut ix = ix;
            ix[1] += 1;
            a.get(ix).map(|&b| (ix, b)).into_iter()
        })
        .chain({
            let mut ix = ix;
            ix[1] = ix[1].wrapping_sub(1);
            a.get(ix).map(|&b| (ix, b)).into_iter()
        })
        .chain({
            let mut ix = ix;
            ix[2] += 1;
            a.get(ix).map(|&b| (ix, b)).into_iter()
        })
        .chain({
            let mut ix = ix;
            ix[2] = ix[2].wrapping_sub(1);
            a.get(ix).map(|&b| (ix, b)).into_iter()
        })
    }

    fn check_com_invariant(&self) {
        println!("array check: {:?}", self.array);
        debug_assert_eq!(Self::compute_com(&self.array.view()), self.com);
    }

    fn array_ref(&self) -> &ndarray::Array3<bool> {
        &self.array
    }
}

enum ComPosition {
    Valid,
    Boundary(RotationAxis),
    Invalid,
}

#[allow(dead_code, clippy::upper_case_acronyms)]
enum RotationAxis {
    Zero,
    X,
    Y,
    Z,
    XY,
    YZ,
    XZ,
    XYZ,
}

#[derive(Clone)]
struct PolycubeNormal(PolycubeBase);

impl PolycubeNormal {
    fn new(n: usize) -> Self {
        let new = Self(PolycubeBase {
            n,
            array: ndarray::Array3::from_elem((n, 1, 1), true),
            com: ndarray::Ix3((0..n).sum(), 0, 0),
            next_pop: ndarray::Ix3(n - 1, 0, 0),
        });
        new.0.check_com_invariant();
        new
    }

    fn remove_next(self) -> Option<PolyCubePreSwap> {
        self.0.check_com_invariant();

        if self.0.next_pop[PolycubeBase::MAIN_AXIS.index()] == 0 {
            return None;
        }

        let mut new_array = self.0.array;
        new_array[self.0.next_pop] = false;

        let mut new_com = self.0.com;
        new_com -= self.0.next_pop;

        debug_assert_eq!(PolycubeBase::compute_com(&new_array.view()), new_com);

        let mut new_next_pop = self.0.next_pop;
        let pop_axis_index = new_next_pop.index_mut(PolycubeBase::MAIN_AXIS.index());

        println!("before: {:?}", new_array);

        if *pop_axis_index == new_array.raw_dim()[PolycubeBase::MAIN_AXIS.index()] - 1
            && new_array
                .index_axis(PolycubeBase::MAIN_AXIS, *pop_axis_index)
                .iter()
                .all(|b| !b)
        {
            if !PolycubeBase::valid_dimensions({
                let mut d = new_array.raw_dim();
                d[PolycubeBase::MAIN_AXIS.index()] -= 1;
                d
            }) {
                return None;
            }

            new_array.remove_index(PolycubeBase::MAIN_AXIS, *pop_axis_index);
        }

        println!("after: {:?}", new_array);

        debug_assert_eq!(PolycubeBase::compute_com(&new_array.view()), new_com);

        *pop_axis_index -= 1;

        let base = PolycubeBase {
            array: new_array,
            com: new_com,
            next_pop: new_next_pop,
            ..self.0
        };

        base.check_com_invariant();

        Some(PolyCubePreSwap(base))
    }
}

#[derive(Clone)]
struct PolyCubePreSwap(PolycubeBase);

#[allow(clippy::result_large_err)]
impl PolyCubePreSwap {
    fn add(
        self,
        ix: ndarray::Ix3,
        h: &impl std::hash::BuildHasher,
    ) -> Result<PolycubeNormal, Self> {
        debug_assert!(!self.0.array[ix]);
        self.0.check_com_invariant();

        let new_com = self.0.com + ix;

        let raxis = match PolycubeBase::valid_com3(new_com, self.0.array.raw_dim(), self.0.n) {
            ComPosition::Valid => None,
            ComPosition::Boundary(raxis) => Some(raxis),
            ComPosition::Invalid => return Err(self),
        };

        let mut new_array = self.0.array.clone();
        new_array[ix] = true;

        // TODO: check that the cubes are all still connected

        if !PolycubeBase::get_neighbors(&new_array, self.0.next_pop + ndarray::Ix3(1, 0, 0)).all(
            |(ix, b)| {
                if !b {
                    PolycubeBase::get_neighbors(&new_array, ix).any(|(_, b)| b)
                } else {
                    true
                }
            },
        ) {
            return Err(self);
        }

        let mut base = PolycubeBase {
            com: new_com,
            array: new_array,
            ..self.0
        };

        if let Some(raxis) = raxis {
            base.rotate_canonical(raxis, h);
        }

        base.check_com_invariant();

        Ok(PolycubeNormal(base))
    }

    fn extend_add(
        self,
        a: ndarray::Axis,
        ix: ndarray::Ix3,
        side: bool,
        h: &impl std::hash::BuildHasher,
    ) -> Result<PolycubeNormal, Self> {
        debug_assert_eq!(ix[a.index()], 0);
        self.0.check_com_invariant();

        if a == PolycubeBase::MAIN_AXIS && side {
            return Err(self);
        }

        let mut new_dim = self.0.array.raw_dim();
        new_dim[a.index()] += 1;

        if !PolycubeBase::valid_dimensions(new_dim) {
            return Err(self);
        }

        let mut plane_dim = self.0.array.raw_dim();
        plane_dim[a.index()] = 1;
        let mut plane = ndarray::Array3::from_elem(plane_dim, false);
        plane[ix] = true;

        if side {
            let mut new_com = self.0.com + ix;
            new_com[a.index()] += self.0.n - 1;

            let raxis = match PolycubeBase::valid_com3(new_com, new_dim, self.0.n) {
                ComPosition::Valid => None,
                ComPosition::Boundary(raxis) => Some(raxis),
                ComPosition::Invalid => return Err(self),
            };

            plane.append(a, self.0.array.view()).unwrap();
            let new_array = plane;

            let mut new_next_pop = self.0.next_pop;
            new_next_pop[a.index()] += 1;

            // TODO: check that the cubes are all still connected

            if !PolycubeBase::get_neighbors(&new_array, new_next_pop + ndarray::Ix3(1, 0, 0)).all(
                |(ix, b)| {
                    if !b {
                        PolycubeBase::get_neighbors(&new_array, ix).any(|(_, b)| b)
                    } else {
                        true
                    }
                },
            ) {
                return Err(self);
            }

            let mut base = PolycubeBase {
                array: new_array,
                com: new_com,
                next_pop: new_next_pop,
                ..self.0
            };

            if let Some(raxis) = raxis {
                base.rotate_canonical(raxis, h);
            }

            base.check_com_invariant();

            Ok(PolycubeNormal(base))
        } else {
            let mut new_com = self.0.com + ix;
            new_com[a.index()] += self.0.array.raw_dim()[a.index()];

            let raxis = match PolycubeBase::valid_com3(new_com, new_dim, self.0.n) {
                ComPosition::Valid => None,
                ComPosition::Boundary(raxis) => Some(raxis),
                ComPosition::Invalid => return Err(self),
            };

            let mut new_array = self.0.array.clone();
            new_array.append(a, plane.view()).unwrap();

            // TODO: check that the cubes are all still connected

            if !PolycubeBase::get_neighbors(&new_array, self.0.next_pop + ndarray::Ix3(1, 0, 0))
                .all(|(ix, b)| {
                    if !b {
                        PolycubeBase::get_neighbors(&new_array, ix).any(|(_, b)| b)
                    } else {
                        true
                    }
                })
            {
                return Err(self);
            }

            let mut base = PolycubeBase {
                array: new_array,
                com: new_com,
                ..self.0
            };

            if let Some(raxis) = raxis {
                base.rotate_canonical(raxis, h);
            }

            base.check_com_invariant();

            Ok(PolycubeNormal(base))
        }
    }

    fn list_open_inner(&self) -> Vec<ndarray::Ix3> {
        self.0.check_com_invariant();

        self.0
            .array
            .indexed_iter()
            .filter_map(|((x, y, z), b)| {
                if !b {
                    Some(ndarray::Ix3(x, y, z))
                } else {
                    None
                }
            })
            .filter(|&ix| PolycubeBase::get_neighbors(&self.0.array, ix).any(|(_, b)| b))
            .collect()
    }

    fn list_edge_outer(&self, a: ndarray::Axis, side: bool) -> Vec<ndarray::Ix3> {
        self.0.check_com_invariant();

        self.0
            .array
            .index_axis(
                a,
                if side {
                    0
                } else {
                    self.0.array.raw_dim()[a.index()] - 1
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
}
