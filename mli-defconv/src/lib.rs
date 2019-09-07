use arraymap::ArrayMap;
use itertools::Itertools;
use mli::*;
use mli_ndarray::Ndeep;
use ndarray::{s, Array, Array1, Array2, ArrayBase, ArrayView1, Data, OwnedRepr};
use std::marker::PhantomData;

type D1 = ndarray::Ix1;
type D2 = ndarray::Ix2;

pub struct DefConvInput {
    pub offsets: Array2<f32>,
    pub features: Array2<f32>,
}

impl DefConvInput {
    /// Extracts the corner features and relative coordinate for a bilinear interpolation.
    fn extract_corners(&self, coordinate: [f32; 2]) -> ([f32; 4], [f32; 2]) {
        // Get the integer versions of four corners.
        let c00 = coordinate.map(|f| f.floor() as isize);
        let c11 = coordinate.map(|f| f.ceil() as isize);
        let c01 = [c00[0], c11[1]];
        let c10 = [c11[0], c00[1]];
        let coords = [c00, c01, c10, c11];

        // Get the relative coordinate.
        let rc = [coordinate[0] - c00[0] as f32, coordinate[1] - c00[1] as f32];

        // Create a closure to detect out-of-bounds.
        let in_bounds = |[y, x]: [isize; 2]| {
            if y < 0
                || x < 0
                || y >= self.features.shape()[0] as isize
                || x >= self.features.shape()[1] as isize
            {
                None
            } else {
                Some([y, x].map(|&n| n as usize))
            }
        };

        // Create a closure to extract the feature with zero-padding.
        let extract = |c| in_bounds(c).map(|c| self.features[c]).unwrap_or(0.0);

        // Extract the corner features.
        (coords.map(|&c| extract(c)), rc)
    }

    /// Looks up a location in the features using bilinear interpolation and zero-padding.
    fn bilinear(&self, coordinate: [f32; 2]) -> f32 {
        // Extract the corner features and the relative coordinate.
        let (f, rc) = self.extract_corners(coordinate);

        // Perform the x interpolation.
        let fx = [
            (1.0 - rc[0]) * f[0] + rc[0] * f[1],
            (1.0 - rc[0]) * f[2] + rc[0] * f[3],
        ];

        // Perform the y interpolation.
        (1.0 - rc[1]) * fx[0] + rc[1] * fx[1]
    }

    /// Looks up a gradient from a location in the features using bilinear interpolation and zero-padding.
    fn bilinear_gradient(&self, coordinate: [f32; 2]) -> [f32; 2] {
        // Extract the corner features and the relative coordinate.
        let (f, rc) = self.extract_corners(coordinate);

        // Perform the y interpolation to get the x values.
        let fx = [
            (1.0 - rc[0]) * f[0] + rc[0] * f[1],
            (1.0 - rc[0]) * f[2] + rc[0] * f[3],
        ];

        // Perform the x interpolation to get the y values.
        let fy = [
            (1.0 - rc[1]) * f[0] + rc[1] * f[2],
            (1.0 - rc[1]) * f[1] + rc[1] * f[3],
        ];

        // The gradient is the difference between the interpolation values in each dimension.
        [fy[1] - fy[0], fx[1] - fx[0]]
    }
}

#[derive(Clone, Debug)]
pub struct DefConv2 {
    weights: Array1<f32>,
    output_shape: [usize; 2],
}

impl DefConv2 {
    pub fn new(weights: Array1<f32>, output_shape: [usize; 2]) -> Self {
        Self {
            weights,
            output_shape,
        }
    }
}

impl Forward for DefConv2 {
    type Input = DefConvInput;
    type Internal = ();
    type Output = Array2<f32>;

    fn forward(&self, input: &Self::Input) -> ((), Self::Output) {
        // Get shapes.
        let outshape = self.output_shape;
        let inshape = input.features.shape();

        // Compute the coordinate multiplier.
        let multipliers = [
            inshape[0] as f32 / outshape[0] as f32,
            inshape[1] as f32 / outshape[1] as f32,
        ];

        // Compute bilinear interpolation for (y, x) pairs.
        (
            (),
            Array::from_shape_vec(
                outshape,
                (0..outshape[0])
                    .cartesian_product(0..outshape[1])
                    .map(|(y, x)| {
                        input.bilinear([
                            (y as f32 + 0.5) * multipliers[0],
                            (x as f32 + 0.5) * multipliers[1],
                        ])
                    })
                    .collect(),
            )
            .expect("mli-defconv: unexpected shape vec"),
        )
    }
}
