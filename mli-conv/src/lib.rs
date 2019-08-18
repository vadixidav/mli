use mli::*;
pub use ndarray::{Array, Array2};

pub struct Conv2(pub Array2<f32>);

impl Forward for Conv2 {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let Self(filter) = self;
        let filter_dims = filter.raw_dim();
        let output_dims = (
            input.shape()[0] + 1 - filter_dims[0],
            input.shape()[1] + 1 - filter_dims[1],
        );
        Array::from_shape_vec(
            output_dims,
            input
                .windows(filter_dims)
                .into_iter()
                .map(|view| (view.to_owned() * filter).sum())
                .collect(),
        )
        .expect("convolution produced incorrectly sized output")
    }
}
