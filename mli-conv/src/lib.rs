use mli::*;
pub use ndarray::{azip, s, Array, Array2, ArrayView2};

fn convolve<'a>(signal: ArrayView2<'a, f32>, filter: ArrayView2<'a, f32>) -> Array2<f32> {
    let filter_dims = filter.raw_dim();
    let output_dims = (
        signal.shape()[0] + 1 - filter_dims[0],
        signal.shape()[1] + 1 - filter_dims[1],
    );
    Array::from_shape_vec(
        output_dims,
        signal
            .windows(filter_dims)
            .into_iter()
            .map(|view| (view.to_owned() * filter).sum())
            .collect(),
    )
    .expect("convolution produced incorrectly sized output")
}

pub struct Conv2(pub Array2<f32>);

impl Forward<Array2<f32>> for Conv2 {
    type Internal = ();
    type Output = Array2<f32>;

    fn forward(&self, input: &Array2<f32>) -> ((), Self::Output) {
        let Self(filter) = self;
        ((), convolve(input.view(), filter.view()))
    }
}

impl Backward<Array2<f32>, Array2<f32>> for Conv2 {
    type InputDelta = Array2<f32>;
    type TrainDelta = Array2<f32>;

    fn backward(
        &self,
        input: &Array2<f32>,
        internal: &Self::Internal,
        output_delta: &Array2<f32>,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        // To compute the `input_delta`, we need to convolve the filter flipped across all its axes with
        // the zero-padded version of the `output_delta`. The amount of zero-padding that needs to be added
        // is `(l - 1) * 2` where `l` is the length of the filter along a given dimension.
        let Self(filter) = self;
        let filter_dims = filter.raw_dim();
        // If the filter is 1 in either dimension, the below code wont work.
        if filter_dims[0] == 1 || filter_dims[1] == 1 {
            unimplemented!("mli-conv: filter dimensions of 1 not implemented yet");
        }
        let padding = (2 * (filter_dims[0] - 1), 2 * (filter_dims[1] - 1));
        let pad_dims = (input.shape()[0] + padding.0, input.shape()[1] + padding.1);
        let mut pad = Array::zeros(pad_dims);
        pad.slice_mut(s![
            padding.0 as i32..-(padding.0 as i32),
            padding.1 as i32..-(padding.1 as i32)
        ])
        .assign(output_delta);
        let input_delta = convolve(pad.view(), filter.slice(s![..;-1,..;-1]));
        (
            input_delta.clone(),
            unimplemented!("train data not yet computed"),
        )
    }
}
