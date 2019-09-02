use crate::{convolve2n, convolve3};
use mli::*;
use mli_ndarray::Ndeep;
use ndarray::{s, Array, Array2, Array3, ArrayBase, Axis, Data, OwnedRepr};
use std::marker::PhantomData;

type D2 = ndarray::Ix2;
type D3 = ndarray::Ix3;

#[derive(Clone, Debug)]
pub struct Conv2n<S>(Array3<f32>, PhantomData<S>);

impl<S> Conv2n<S> {
    /// The dimensions of the filters array are `[filter, row, col]`.
    pub fn new(filters: Array3<f32>) -> Self {
        Self(filters, PhantomData)
    }
}

impl<S> Forward for Conv2n<S>
where
    S: Data<Elem = f32>,
{
    type Input = ArrayBase<S, D2>;
    type Internal = ();
    type Output = Array3<f32>;

    fn forward(&self, input: &Self::Input) -> ((), Self::Output) {
        let Self(filter, _) = self;
        ((), convolve2n(input.view(), filter.view()))
    }
}

impl<S> Backward for Conv2n<S>
where
    S: Data<Elem = f32>,
{
    type OutputDelta = Array3<f32>;
    type InputDelta = Array2<f32>;
    type TrainDelta = Ndeep<OwnedRepr<f32>, D3>;

    fn backward(
        &self,
        input: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        // To compute the `input_delta`, we need to convolve the filter flipped across all its axes with
        // the zero-padded version of the `output_delta`. The amount of zero-padding that needs to be added
        // is `(l - 1) * 2` where `l` is the length of the filter along a given dimension.
        let Self(filters, _) = self;
        let filter_dims = (filters.shape()[1], filters.shape()[2]);
        // If the filter is 1 in either dimension, the below code wont work.
        if filter_dims.0 <= 1 || filter_dims.1 <= 1 {
            unimplemented!("mli-conv: filter dimensions of 1 not implemented yet");
        }
        let padding = (2 * (filter_dims.0 - 1), 2 * (filter_dims.1 - 1));
        let pad_dims = (
            output_delta.shape()[0],
            output_delta.shape()[1] + padding.0,
            output_delta.shape()[2] + padding.1,
        );
        let mut pad = Array::zeros(pad_dims);
        #[allow(clippy::deref_addrof)]
        pad.slice_mut(s![
            ..,
            padding.0 as i32 / 2..-(padding.0 as i32) / 2,
            padding.1 as i32 / 2..-(padding.1 as i32) / 2
        ])
        .assign(output_delta);
        #[allow(clippy::deref_addrof)]
        // Produces a 1xRxC output.
        let input_delta = convolve3(pad.view(), filters.slice(s![.., ..;-1, ..;-1]));
        assert_eq!(
            input_delta.shape()[0],
            1,
            "did not expect dimension other than 1"
        );
        let input_delta_shape = (input_delta.shape()[1], input_delta.shape()[2]);
        let input_delta = input_delta
            .into_shape(input_delta_shape)
            .expect("unable to reshape in conv2d");

        let train_delta = input
            .windows(filter_dims)
            .into_iter()
            .zip(output_delta.lanes(Axis(0)))
            .map(|(view, delta_lane)| {
                let reshaped_delta = delta_lane.insert_axis(Axis(1)).insert_axis(Axis(1));
                let broadcasted_delta = reshaped_delta
                    .broadcast(filters.raw_dim())
                    .expect("unable to broadcast delta");
                let reshaped_view = view.insert_axis(Axis(0));
                let broadcasted_view = reshaped_view
                    .broadcast(filters.raw_dim())
                    .expect("unable to broadcast view");
                broadcasted_view.to_owned() * broadcasted_delta
            })
            .fold(Array3::zeros(filters.raw_dim()), |acc, item| acc + item);
        (input_delta, Ndeep(train_delta))
    }
}

impl<S> Train for Conv2n<S>
where
    S: Data<Elem = f32>,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0 += &train_delta.0;
    }
}
