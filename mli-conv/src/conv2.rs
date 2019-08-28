use mli::*;
use ndarray::{s, Array, Array2, ArrayBase, ArrayView2, Data};
use std::marker::PhantomData;

type D = ndarray::Ix2;

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

#[derive(Clone, Debug)]
pub struct Conv2<S>(Array2<f32>, PhantomData<S>);

impl<S> Conv2<S> {
    pub fn new(filter: Array2<f32>) -> Self {
        Self(filter, PhantomData)
    }
}

impl<S> Forward for Conv2<S>
where
    S: Data<Elem = f32>,
{
    type Input = ArrayBase<S, D>;
    type Internal = ();
    type Output = Array2<f32>;

    fn forward(&self, input: &Self::Input) -> ((), Self::Output) {
        let Self(filter, _) = self;
        ((), convolve(input.view(), filter.view()))
    }
}

impl<S> Backward for Conv2<S>
where
    S: Data<Elem = f32>,
{
    type OutputDelta = Array2<f32>;
    type InputDelta = Array2<f32>;
    type TrainDelta = Array2<f32>;

    fn backward(
        &self,
        input: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        // To compute the `input_delta`, we need to convolve the filter flipped across all its axes with
        // the zero-padded version of the `output_delta`. The amount of zero-padding that needs to be added
        // is `(l - 1) * 2` where `l` is the length of the filter along a given dimension.
        let Self(filter, _) = self;
        let filter_dims = filter.raw_dim();
        // If the filter is 1 in either dimension, the below code wont work.
        if filter_dims[0] == 1 || filter_dims[1] == 1 {
            unimplemented!("mli-conv: filter dimensions of 1 not implemented yet");
        }
        let padding = (2 * (filter_dims[0] - 1), 2 * (filter_dims[1] - 1));
        let pad_dims = (
            output_delta.shape()[0] + padding.0,
            output_delta.shape()[1] + padding.1,
        );
        let mut pad = Array::zeros(pad_dims);
        #[allow(clippy::deref_addrof)]
        pad.slice_mut(s![
            padding.0 as i32 / 2..-(padding.0 as i32) / 2,
            padding.1 as i32 / 2..-(padding.1 as i32) / 2
        ])
        .assign(output_delta);
        #[allow(clippy::deref_addrof)]
        let input_delta = convolve(pad.view(), filter.slice(s![..;-1,..;-1]));

        let train_delta = input
            .windows(filter_dims)
            .into_iter()
            .zip(output_delta.iter())
            .map(|(view, &delta)| (view.to_owned() * delta))
            .fold(Array2::zeros(filter_dims), |acc, item| acc + item);
        (input_delta, train_delta)
    }
}

impl<S> Train for Conv2<S>
where
    S: Data<Elem = f32>,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0 += train_delta;
    }
}
