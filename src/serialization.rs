use nanoserde::{DeBin, SerBin};

#[derive(SerBin, DeBin, PartialEq, Eq, Hash)]
pub struct SerPolycube(Vec<u8>);

#[derive(SerBin, DeBin)]
pub struct Results {
    counts: Vec<(usize, usize)>,
}

#[derive(SerBin, DeBin)]
pub struct SerResults(Vec<u8>);

#[derive(SerBin, DeBin)]
pub struct JobRequest {
    pub jobs_wanted: usize,
    pub results: std::collections::HashMap<SerPolycube, SerResults>,
}

#[derive(SerBin, DeBin)]
pub struct JobResponse {
    pub target_n: usize,
    pub jobs: Vec<SerPolycube>,
}

impl SerPolycube {
    pub fn ser(view: ndarray::ArrayView3<bool>) -> Self {
        use ndarray::Dimension;

        let mut out = vec![0; 3 + (view.len() - 1) / 8 + 1];

        let (header, data) = out.split_at_mut(3);

        header.copy_from_slice(
            &Into::<[usize; 3]>::into(view.raw_dim().into_pattern()).map(|d| d.try_into().unwrap()),
        );

        for (i, &bit) in view.iter().enumerate() {
            data[i / 8] |= (bit as u8) << (i % 8);
        }

        Self(out)
    }

    pub fn de(&self) -> ndarray::Array3<bool> {
        let (header, data) = self.0.split_at(3);
        let xyz = TryInto::<[u8; 3]>::try_into(header)
            .unwrap()
            .map(|u| u.into());

        let mut vec = Vec::with_capacity(data.len() * 8);

        for byte in data {
            vec.extend_from_slice(&[
                byte & 0b00000001 != 0,
                byte & 0b00000010 != 0,
                byte & 0b00000100 != 0,
                byte & 0b00001000 != 0,
                byte & 0b00010000 != 0,
                byte & 0b00100000 != 0,
                byte & 0b01000000 != 0,
                byte & 0b10000000 != 0,
            ]);
        }

        vec.truncate(xyz.iter().product());

        ndarray::Array3::from_shape_vec(xyz, vec).unwrap()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn from_slice(slice: &[u8]) -> Self {
        Self(slice.to_vec())
    }
}

impl SerResults {
    pub fn de(&self) -> Results {
        Results::deserialize_bin(&self.0).unwrap()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn from_slice(slice: &[u8]) -> Self {
        Self(slice.to_vec())
    }
}

impl Results {
    pub fn ser(&self) -> SerResults {
        SerResults(self.serialize_bin())
    }

    pub fn from_map(map: &std::collections::BTreeMap<usize, (usize, usize)>) -> Self {
        Self {
            counts: map.values().copied().collect(),
        }
    }

    pub fn from_vec(counts: Vec<(usize, usize)>) -> Self {
        Self { counts }
    }

    pub fn into_vec(self) -> Vec<(usize, usize)> {
        self.counts
    }
}
