use crate::convolve2;
use mli::*;
use mli_ndarray::Ndeep;
use ndarray::{Array, Array2, ArrayBase, Data, OwnedRepr, s};
use std::marker::PhantomData;

type D = ndarray::Ix2;

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
    type Internal = EmptyData;
    type Output = Array2<f32>;

    fn forward(&self, input: &Self::Input) -> (EmptyData, Self::Output) {
        let Self(filter, _) = self;
        (EmptyData, convolve2(input.view(), filter.view()))
    }
}

impl<S> Backward for Conv2<S>
where
    S: Data<Elem = f32>,
{
    type OutputDelta = Array2<f32>;
    type InputDelta = Array2<f32>;
    type TrainDelta = Ndeep<OwnedRepr<f32>, D>;

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
        let input_delta = convolve2(pad.view(), filter.slice(s![..;-1,..;-1]));

        let train_delta = input
            .windows(filter_dims)
            .into_iter()
            .zip(output_delta.iter())
            .map(|(view, &delta)| (view.to_owned() * delta))
            .fold(Array2::zeros(filter_dims), |acc, item| acc + item);
        (input_delta, Ndeep(train_delta))
    }
}

impl<S> Train for Conv2<S>
where
    S: Data<Elem = f32>,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0 += &train_delta.0;
    }
}
