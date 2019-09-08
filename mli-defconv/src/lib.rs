use arraymap::ArrayMap;
use itertools::Itertools;
use mli::*;
use mli_ndarray::Ndeep;
use ndarray::{Array, Array1, Array2, OwnedRepr};

type D1 = ndarray::Ix1;

fn corners(coordinate: [f32; 2]) -> [[isize; 2]; 4] {
    let c00 = coordinate.map(|f| f.floor() as isize);
    let c11 = coordinate.map(|f| f.ceil() as isize);
    let c01 = [c00[0], c11[1]];
    let c10 = [c11[0], c00[1]];
    [c00, c01, c10, c11]
}

/// Looks up a position gradient from a location in the features using bilinear interpolation and zero-padding.
fn bilinear_position_gradient(f: [f32; 4], rc: [f32; 2]) -> [f32; 2] {
    // Perform the y interpolation to get the x values.
    let fx = [
        (1.0 - rc[0]) * f[0] + rc[0] * f[2],
        (1.0 - rc[0]) * f[1] + rc[0] * f[3],
    ];

    // Perform the x interpolation to get the y values.
    let fy = [
        (1.0 - rc[1]) * f[0] + rc[1] * f[1],
        (1.0 - rc[1]) * f[2] + rc[1] * f[3],
    ];

    // The gradient is the difference between the interpolation values in each dimension.
    [fy[1] - fy[0], fx[1] - fx[0]]
}

pub struct DefConvData<'a> {
    pub features: &'a Array2<f32>,
    pub offsets: &'a Array2<f32>,
}

impl<'a> DefConvData<'a> {
    fn validate_corners(&self, coordinate: [f32; 2]) -> ([Option<[usize; 2]>; 4], [f32; 2]) {
        // Get the integer versions of four corners.
        let coords = corners(coordinate);

        // Get the relative coordinate.
        let rc = [
            coordinate[0] - coords[0][0] as f32,
            coordinate[1] - coords[0][1] as f32,
        ];

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
        let extract = |c| in_bounds(c);

        (coords.map(|&c| extract(c)), rc)
    }

    /// Extracts the corner features and relative coordinate for a bilinear interpolation.
    fn extract_corners(&self, validated_corners: [Option<[usize; 2]>; 4]) -> [f32; 4] {
        validated_corners.map(|c| c.as_ref().map(|&c| self.features[c]).unwrap_or(0.0))
    }

    /// Looks up a location in the features using bilinear interpolation and zero-padding.
    fn bilinear(&self, coordinate: [f32; 2]) -> f32 {
        // Extract the corner features and the relative coordinate.
        let (validated_corners, rc) = self.validate_corners(coordinate);
        let f = self.extract_corners(validated_corners);

        // Perform the y interpolation.
        let fx = [
            (1.0 - rc[0]) * f[0] + rc[0] * f[2],
            (1.0 - rc[0]) * f[1] + rc[0] * f[3],
        ];

        // Perform the x interpolation.
        (1.0 - rc[1]) * fx[0] + rc[1] * fx[1]
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
    type Input = (Array2<f32>, Array2<f32>);
    type Internal = ();
    type Output = Array2<f32>;

    fn forward(&self, (features, offsets): &Self::Input) -> ((), Self::Output) {
        let input = DefConvData { features, offsets };
        assert_eq!(self.weights.len(), input.offsets.shape()[0]);
        // Get shapes.
        let outshape = self.output_shape;
        let inshape = input.features.raw_dim();

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
                        self.weights
                            .iter()
                            .zip(input.offsets.outer_iter())
                            .map(|(weight, offset)| {
                                input.bilinear([
                                    (y as f32 + 0.5 + offset[0]) * multipliers[0],
                                    (x as f32 + 0.5 + offset[1]) * multipliers[1],
                                ]) * weight
                            })
                            .sum()
                    })
                    .collect(),
            )
            .expect("mli-defconv: unexpected shape vec"),
        )
    }
}

impl Backward for DefConv2 {
    type OutputDelta = Array2<f32>;
    type InputDelta = (Array2<f32>, Array2<f32>);
    type TrainDelta = Ndeep<OwnedRepr<f32>, D1>;

    fn backward(
        &self,
        (features, offsets): &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let input = DefConvData { features, offsets };
        // Get shapes.
        let outshape = self.output_shape;
        let inshape = input.features.raw_dim();

        // Compute the coordinate multiplier.
        let multipliers = [
            inshape[0] as f32 / outshape[0] as f32,
            inshape[1] as f32 / outshape[1] as f32,
        ];

        // We cannot determine in advance what locations are affected
        let mut feature_deltas: Array2<f32> = Array2::zeros(inshape);
        let mut offset_deltas: Array2<f32> = Array2::zeros(input.offsets.raw_dim());
        let mut weight_deltas: Array1<f32> = Array1::zeros(self.weights.raw_dim());

        for (y, x) in (0..outshape[0]).cartesian_product(0..outshape[1]) {
            let output_delta = output_delta[[y, x]];
            for (ix, (weight, offset)) in self
                .weights
                .iter()
                .zip(input.offsets.outer_iter())
                .enumerate()
            {
                // Compute the original sample coordinate.
                let sample_coordinate = [
                    (y as f32 + 0.5 + offset[0]) * multipliers[0],
                    (x as f32 + 0.5 + offset[1]) * multipliers[1],
                ];

                // Find the corners if they are within the input.
                let (validated_corners, rc) = input.validate_corners(sample_coordinate);

                // Loop over all the corners and the amount each contributed.
                for (coord, coeff) in validated_corners.iter().zip(
                    [
                        (1.0 - rc[0]) * (1.0 - rc[1]),
                        (1.0 - rc[0]) * rc[1],
                        rc[0] * (1.0 - rc[1]),
                        rc[0] * rc[1],
                    ]
                    .iter(),
                ) {
                    // If the coordinate was in bounds of the input tensor.
                    if let Some(coord) = coord {
                        // Add the contribution.
                        feature_deltas[*coord] += coeff * weight * output_delta;
                    }
                }

                // Compute the position gradients.
                let f = input.extract_corners(validated_corners);
                let position_gradient = bilinear_position_gradient(f, rc);
                // Compute and add the offset gradients based on the position gradients and chain rule.
                // The multipliers affect the gradient of the position because they multiply the sampling locations.
                // The greater the weight the greater the effect on sampling location because the output is multiplied by the weight.
                offset_deltas[[ix, 0]] +=
                    multipliers[0] * position_gradient[0] * weight * output_delta;
                offset_deltas[[ix, 1]] +=
                    multipliers[1] * position_gradient[1] * weight * output_delta;

                // TODO: This is incredibly inefficient recomputing the entire bilinear interpolation. Use internals to store it.
                weight_deltas[ix] += input.bilinear(sample_coordinate) * output_delta;
            }
        }

        // Compute bilinear interpolation for (y, x) pairs.
        ((feature_deltas, offset_deltas), Ndeep(weight_deltas))
    }
}
