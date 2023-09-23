use {
    ndarray::{Array3, ArrayView3, Axis, Dimension, Ix3},
    std::collections::BTreeMap,
};

type CubeCache = BTreeMap<[usize; 3], BTreeMap<Box<[bool]>, CubeType>>;

#[derive(Debug, PartialEq)]
enum CubeType {
    Unique,
    Duplicate,
}

fn cache_add_cube(cache: &mut CubeCache, a: ArrayView3<bool>, t: CubeType) -> bool {
    let map = cache.entry(a.raw_dim().into_pattern().into()).or_default();
    let standard = a.as_standard_layout();
    let slice = standard.as_slice().unwrap();

    if map.contains_key(slice) {
        false
    } else {
        map.insert(slice.into(), t);
        true
    }
}

#[allow(dead_code)]
pub fn solve(n: usize) {
    let cube = PolycubeFull::new(n);
    let mut cache = CubeCache::new();

    recurse(&cube, &mut cache);

    println!("results:");
    for (k, v) in cache.iter() {
        println!("{:?}", k);
        for (s, t) in v {
            //println!("{:?}, {:?}", s, t);
            println!(
                "{:?}, {:?}",
                ArrayView3::from_shape(*k, s)
                    .unwrap()
                    .indexed_iter()
                    .filter_map(|(ix, &b)| if b { Some(ix) } else { None })
                    .collect::<Vec<_>>(),
                t
            );
        }
    }
    println!(
        "count: {}",
        cache
            .values()
            .flat_map(|m| m.values())
            .filter(|&t| *t == CubeType::Unique)
            .count()
    );

    fn recurse(cube: &PolycubeFull, cache: &mut CubeCache) {
        cube.assert_all();

        if !cache_add_cube(cache, cube.0.array.view(), CubeType::Unique) {
            return;
        } else {
            cube.try_gen_symmetrical(cache);
        }

        let Some(missing) = cube.remove_1() else {
            return;
        };
        missing.assert_all();

        for ix in missing.list_candidates() {
            if let Some(full) = missing.add_1(ix) {
                recurse(&full, cache);
            }
        }
    }
}

#[derive(Clone)]
struct PolycubeBase {
    array: Array3<bool>,
    n: usize,
    com: Ix3,
    pop_next: Ix3,
}

fn neighbor_ixs(ix: Ix3) -> [Ix3; 6] {
    let (x, y, z) = ix.into_pattern();
    [
        Ix3(x.wrapping_add(1), y, z),
        Ix3(x, y.wrapping_add(1), z),
        Ix3(x, y, z.wrapping_add(1)),
        Ix3(x.wrapping_sub(1), y, z),
        Ix3(x, y.wrapping_sub(1), z),
        Ix3(x, y, z.wrapping_sub(1)),
    ]
}

// debug assertions
impl PolycubeBase {
    fn assert_connected(&self) {
        let all_have_neighbors = self.array.indexed_iter().all(|((x, y, z), &b)| {
            let ix = Ix3(x, y, z);
            if b {
                neighbor_ixs(ix)
                    .into_iter()
                    .any(|jx| self.array.get(jx).copied().unwrap_or(false))
            } else {
                true
            }
        });

        if count_set_cubes_slow(self.array.view()) > 1 {
            debug_assert!(all_have_neighbors);
        }
    }

    fn assert_bounding_invariant(&self, min_pad: Ix3, max_pad: Ix3) {
        let (min, max) = get_bounding_box(
            self.array
                .view()
                .indexed_iter()
                .filter_map(|((x, y, z), &b)| if b { Some(Ix3(x, y, z)) } else { None }),
        );
        debug_assert_eq!(min, min_pad);
        debug_assert_eq!(max + max_pad, self.array.raw_dim() - Ix3(1, 1, 1));
    }

    fn assert_valid_dimensions(&self) {
        debug_assert!(validate_dimensions(self.array.raw_dim()));
    }

    fn assert_n_invariant(&self) {
        debug_assert_eq!(count_set_cubes_slow(self.array.view()), self.n);
    }

    fn assert_com_invariant(&self) {
        debug_assert_eq!(calculate_com_slow(self.array.view()), self.com);
    }

    fn assert_present_pop_next(&self) {
        debug_assert!(self.array[self.pop_next]);
    }

    fn assert_valid_com(&self) {
        /*
        debug_assert!(!invalid_com(
            classify_com(self.com, self.array.raw_dim(), self.n),
            self.array.raw_dim()
        ));*/
        debug_assert!(valid_com2(self.com, self.array.raw_dim(), self.n));
    }
}

fn validate_dimensions(d: Ix3) -> bool {
    let (dx, dy, dz) = d.into_pattern();
    dx >= dy && dy >= dz
}

fn calculate_com_slow(a: ArrayView3<bool>) -> Ix3 {
    a.indexed_iter()
        .filter_map(|((x, y, z), &b)| b.then_some(Ix3(x, y, z)))
        .reduce(|a, b| a + b)
        .unwrap()
}

fn count_set_cubes_slow(a: ArrayView3<bool>) -> usize {
    a.iter().filter(|&&b| b).count()
}

fn get_bounding_box(ixs: impl Iterator<Item = Ix3>) -> (Ix3, Ix3) {
    use std::cmp::{max, min};

    let (ixmin, ixmax) = ixs.map(|ix| ix.into_pattern()).fold(
        (
            (usize::MAX, usize::MAX, usize::MAX),
            (usize::MIN, usize::MIN, usize::MIN),
        ),
        |(ixmin, ixmax), ix| {
            (
                (min(ixmin.0, ix.0), min(ixmin.1, ix.1), min(ixmin.2, ix.2)),
                (max(ixmax.0, ix.0), max(ixmax.1, ix.1), max(ixmax.2, ix.2)),
            )
        },
    );

    (
        Ix3(ixmin.0, ixmin.1, ixmin.2),
        Ix3(ixmax.0, ixmax.1, ixmax.2),
    )
}

/*
fn classify_com(c: Ix3, d: Ix3, n: usize) -> Region {
    let (dx, dy, dz) = (d - Ix3(1, 1, 1)).into_pattern();
    let (cx, cy, cz) = c.into_pattern();

    let xmid = (2 * cx).cmp(&(n * dx));
    let ymid = (2 * cy).cmp(&(n * dy));
    let zmid = (2 * cz).cmp(&(n * dz));

    let xy = (cy * dx).cmp(&(dy * cx));
    let yz = (cz * dy).cmp(&(dz * cy));

    //let xy = (dy * cx).cmp(&(cy * dx));
    //let yz = (dz * cy).cmp(&(cz * dy));

    use std::cmp::Ordering::*;

    match ((xmid, ymid, zmid), (xy, yz)) {
        ((Equal, Equal, Equal), _) => Region::Xeq0_Yeq0_Zeq0,

        ((Less, Equal, Equal), _) => Region::Xle0_Yeq0_Zeq0,
        ((Equal, Less, Equal), _) => Region::Xeq0_Yle0_Zeq0,
        ((Equal, Equal, Less), _) => Region::Xeq0_Yeq0_Zle0,

        ((Less, Less, Equal), (Equal, _)) => Region::XeqYle0_Zeq0,
        ((Equal, Less, Less), (_, Equal)) => Region::Xeq0_YeqZle0,
        ((Less, Less, Less), (Equal, Equal)) => Region::XeqYeqZle0,

        ((Equal, Less, Less), _) => Region::Xeq0_Yle0_Zle0,
        ((Less, Equal, Less), _) => Region::Xle0_Yeq0_Zle0,
        ((Less, Less, Equal), _) => Region::Xle0_Yle0_Zeq0,

        ((Less, Less, Less), (Equal, _)) => Region::XeqYle0_Zle0,
        ((Less, Less, Less), (_, Equal)) => Region::Xle0_YeqZle0,

        ((Less, Less, Less), (Less, Less)) => Region::XleYleZle0,
        ((Less, Less, Less), (Less, _)) => Region::XleYle0_Zle0,
        ((Less, Less, Less), (_, Less)) => Region::Xle0_YleZle0,
        ((Less, Less, Less), _) => Region::Xle0_Yle0_Zle0,

        _ => Region::Other,
    }
}
*/
/*
fn invalid_com(r: Region, d: Ix3) -> bool {
    let (dx, dy, dz) = d.into_pattern();

    match (dx == dy, dy == dz) {
        (false, false) => matches!(r, Region::Other),
        (true, false) => matches!(
            r,
            Region::Other | Region::Xle0_Yle0_Zle0 | Region::Xle0_YleZle0
        ),
        (false, true) => matches!(
            r,
            Region::Other | Region::Xle0_Yle0_Zle0 | Region::XleYle0_Zle0
        ),
        (true, true) => matches!(
            r,
            Region::Other | Region::Xle0_Yle0_Zle0 | Region::Xle0_YleZle0 | Region::XleYle0_Zle0
        ),
    }
}
*/

fn valid_com2(c: Ix3, d: Ix3, n: usize) -> bool {
    let (dx, dy, dz) = (d - Ix3(1, 1, 1)).into_pattern();
    let (cx, cy, cz) = c.into_pattern();

    if 2 * cx > n * dx || 2 * cy > n * dy || 2 * cz > n * dz {
        return false;
    }

    let xy = cy * dx <= dy * cx;
    let yz = cz * dy <= dz * cy;

    match (dx == dy, dy == dz) {
        (false, false) => true,
        (true, false) => xy,
        (false, true) => yz,
        (true, true) => xy && yz,
    }
}

/*
// in A cmp B _ C cmp D form where eq is = and le is <
// from most specific to least specific
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug)]
enum Region {
    // point
    Xeq0_Yeq0_Zeq0, // center

    // lines
    Xle0_Yeq0_Zeq0, // x axis
    Xeq0_Yle0_Zeq0, // y axis
    Xeq0_Yeq0_Zle0, // z axis

    XeqYle0_Zeq0, // xy diagonal on z = 0
    Xeq0_YeqZle0, // yz diagonal on x = 0
    XeqYeqZle0,   // xyz diagonal

    // planes
    Xeq0_Yle0_Zle0, // x plane
    Xle0_Yeq0_Zle0, // y plane
    Xle0_Yle0_Zeq0, // z plane

    XeqYle0_Zle0, // xy diagonal (excluding z = 0)
    Xle0_YeqZle0, // yz diagonal (excluding x = 0)

    // volumes
    XleYleZle0,     // 1/48 bounded volume
    XleYle0_Zle0,   // 1/16 bounded volume symmetric over x = y
    Xle0_YleZle0,   // 1/16 bounded volume symmetric over y = z
    Xle0_Yle0_Zle0, // 1/8 bounded volume

    Other, // everything else (7/8 volume)
}

impl Region {
    /*
    fn eq0(&self) -> [bool; 3] {
        match self {
            Self::Xeq0_Yeq0_Zeq0 => [true, true, true],

            Self::Xle0_Yeq0_Zeq0 => [false, true, true],
            Self::Xeq0_Yle0_Zeq0 => [true, false, true],
            Self::Xeq0_Yeq0_Zle0 => [true, true, false],

            Self::XeqYle0_Zeq0 => [false, false, true],
            Self::Xeq0_YeqZle0 => [true, false, false],
            Self::XeqYeqZle0 => [false, false, false],

            Self::Xeq0_Yle0_Zle0 => [true, false, false],
            Self::Xle0_Yeq0_Zle0 => [false, true, false],
            Self::Xle0_Yle0_Zeq0 => [false, false, true],

            Self::XeqYle0_Zle0 => [false, false, false],
            Self::Xle0_YeqZle0 => [false, false, false],

            Self::XleYleZle0 => [false, false, false],
            Self::XleYle0_Zle0 => [false, false, false],
            Self::Xle0_YleZle0 => [false, false, false],
            Self::Xle0_Yle0_Zle0 => [false, false, false],

            Self::Other => [false, false, false],
        }
    }
    */
}
*/

fn com_eq0(c: Ix3, d: Ix3, n: usize) -> [bool; 3] {
    let (dx, dy, dz) = (d - Ix3(1, 1, 1)).into_pattern();
    let (cx, cy, cz) = c.into_pattern();

    let xmid = 2 * cx == n * dx;
    let ymid = 2 * cy == n * dy;
    let zmid = 2 * cz == n * dz;

    [xmid, ymid, zmid]
}

struct PolycubeFull(PolycubeBase);

impl PolycubeFull {
    #[allow(dead_code)]
    fn assert_all(&self) {
        self.0.assert_valid_dimensions();
        self.0.assert_bounding_invariant(Ix3(0, 0, 0), Ix3(0, 0, 0));
        self.0.assert_n_invariant();
        self.0.assert_connected();
        self.0.assert_com_invariant();
        self.0.assert_present_pop_next();
        self.0.assert_valid_com();
    }

    fn new(n: usize) -> Self {
        let array = Array3::from_elem((n, 1, 1), true);
        let com = calculate_com_slow(array.view());
        let pop_next = Ix3(n - 1, 0, 0);

        Self(PolycubeBase {
            array,
            n,
            com,
            pop_next,
        })
    }

    fn remove_1(&self) -> Option<PolycubeMissing> {
        if self.0.pop_next[0] == 0 {
            return None;
        }

        let mut new = self.0.clone();

        new.array[new.pop_next] = false;
        new.com -= new.pop_next;

        let last_plane_index = new.array.raw_dim()[0] - 1;
        let last_plane_empty = !new
            .array
            .index_axis(Axis(0), last_plane_index)
            .iter()
            .any(|&b| b);

        let mut larger_array = Array3::from_elem(
            new.array.raw_dim() + Ix3(if last_plane_empty { 0 } else { 1 }, 2, 2),
            false,
        );
        #[allow(clippy::reversed_empty_ranges)]
        let mut slice = larger_array.slice_mut(ndarray::s![0..=last_plane_index, 1..-1, 1..-1]);
        slice.assign(&new.array);

        new.array = larger_array;
        new.com += Ix3(0, 1, 1) * (new.n - 1);
        new.pop_next += Ix3(0, 1, 1);
        new.pop_next[0] -= 1;

        Some(PolycubeMissing(new))
    }

    fn try_gen_symmetrical(&self, cache: &mut CubeCache) {
        let d: [usize; 3] = self.0.array.raw_dim().into_pattern().into();
        let c: [usize; 3] = self.0.com.into_pattern().into();

        // iterate on length 3 permutations
        for [px, py, pz] in PERMUTATIONS {
            // check if dimensions are the same
            if d != [d[px], d[py], d[pz]] {
                continue;
            }

            // iterate on length 3 bitstrings
            // true = sign unchanged
            for s in COMBINATIONS {
                let nc = [c[px], c[py], c[pz]];

                // check if com is the same
                if !(c == nc
                    && std::iter::zip(s, com_eq0(self.0.com, self.0.array.raw_dim(), self.0.n))
                        .all(|(s, c)| s || c))
                {
                    continue;
                }

                // add cube
                let mut view = self.0.array.view().permuted_axes((px, py, pz));
                for (i, s) in s.into_iter().enumerate() {
                    if !s {
                        view.invert_axis(Axis(i));
                    }
                }

                cache_add_cube(cache, view, CubeType::Duplicate);
            }
        }
    }
}

const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [1, 0, 2],
    [2, 0, 1],
    [0, 2, 1],
    [1, 2, 0],
    [2, 1, 0],
];

const COMBINATIONS: [[bool; 3]; 8] = [
    [false, false, false],
    [true, false, false],
    [false, true, false],
    [false, false, true],
    [true, true, false],
    [true, false, true],
    [false, true, true],
    [true, true, true],
];

struct PolycubeMissing(PolycubeBase);

impl PolycubeMissing {
    #[allow(dead_code)]
    fn assert_all(&self) {
        self.0.assert_bounding_invariant(Ix3(0, 1, 1), Ix3(1, 1, 1));
        self.0.assert_com_invariant();
        //self.0.assert_connected();
        self.0.assert_present_pop_next();
    }

    fn add_1(&self, ix: Ix3) -> Option<PolycubeFull> {
        debug_assert!(!self.0.array[ix]);

        let mut new = self.0.clone();
        new.array[ix] = true;
        new.com += ix;

        // check for connectiveness

        let connected = neighbor_ixs(new.pop_next + Ix3(1, 0, 0))
            .into_iter()
            .all(|ix| {
                if new.array.get(ix).copied().unwrap_or(false) {
                    neighbor_ixs(ix)
                        .into_iter()
                        .any(|jx| new.array.get(jx).copied().unwrap_or(false))
                } else {
                    true
                }
            });

        if !connected {
            return None;
        }

        // shrink unused space

        let (ixmin, ixmax) = get_bounding_box(std::iter::once(ix));
        let dmax = new.array.raw_dim() - Ix3(2, 2, 2);

        new.array = new.array.slice_move(ndarray::s![
            0..=ixmax[0].max(dmax[0]),
            ixmin[1].min(1)..=ixmax[1].max(dmax[1]),
            ixmin[2].min(1)..=ixmax[2].max(dmax[2]),
        ]);

        if ixmin[1] != 0 {
            new.com[1] -= new.n;
            new.pop_next[1] -= 1;
        }

        if ixmin[2] != 0 {
            new.com[2] -= new.n;
            new.pop_next[2] -= 1;
        }

        // check dimensions

        if !validate_dimensions(new.array.raw_dim()) {
            return None;
        }

        // check com

        //let r = classify_com(new.com, new.array.raw_dim(), new.n);

        /*
        if invalid_com(r, new.array.raw_dim()) {
            return None;
        }*/

        if !valid_com2(new.com, new.array.raw_dim(), new.n) {
            return None;
        }

        Some(PolycubeFull(new))
    }

    fn list_candidates(&self) -> impl Iterator<Item = Ix3> + '_ {
        self.0.array.indexed_iter().filter_map(|((x, y, z), b)| {
            let ix = Ix3(x, y, z);
            if !b {
                if neighbor_ixs(ix)
                    .into_iter()
                    .any(|jx| self.0.array.get(jx).copied().unwrap_or(false))
                {
                    Some(ix)
                } else {
                    None
                }
            } else {
                None
            }
        })
    }
}

#[test]
fn test_new() {
    for n in 1..20 {
        let new = PolycubeFull::new(n);
        new.assert_all();
    }
}

#[test]
fn test_remove1() {
    for n in 1..20 {
        let new = PolycubeFull::new(n);
        if let Some(missing) = new.remove_1() {
            missing.assert_all();
        }
    }
}

#[test]
fn test_valid_regions() {
    let n = 5;
    let d = Ix3(n, n, n);
    let cube = Array3::from_elem(d, false);

    for ((x, y, z), _) in cube.indexed_iter() {
        let ix = Ix3(x, y, z);

        if valid_com2(ix, d, 1) {
            println!("{ix:?}");
        }
    }
}
