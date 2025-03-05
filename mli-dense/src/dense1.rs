use mli::*;
use mli_ndarray::Ndeep;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, OwnedRepr};
use std::marker::PhantomData;

type D1 = ndarray::Ix1;
type D2 = ndarray::Ix2;

#[derive(Clone, Debug)]
pub struct Dense1<S>(Array2<f32>, PhantomData<S>);

impl<S> Dense1<S> {
    /// The dimensions of the filters array are `[filter, col]`.
    pub fn new(weights: Array2<f32>) -> Self {
        Self(weights, PhantomData)
    }
}

impl<S> Forward for Dense1<S>
where
    S: Data<Elem = f32>,
{
    type Input = ArrayBase<S, D1>;
    type Internal = EmptyData;
    type Output = Array1<f32>;

    fn forward(&self, input: &Self::Input) -> (EmptyData, Self::Output) {
        let Self(weights, _) = self;
        assert_eq!(
            input.shape(),
            &self.0.shape()[1..],
            "dense layer shapes dont match"
        );
        // TODO: This is inefficient, use ndarray::linalg::general_mat_mul().
        let output = Array::from_shape_vec(
            weights.shape()[0],
            weights
                .outer_iter()
                .map(|filter| (input.to_owned() * filter).sum())
                .collect(),
        )
        .expect("dense layer produced incorrectly sized output");

        (EmptyData, output)
    }
}

impl<S> Backward for Dense1<S>
where
    S: Data<Elem = f32>,
{
    type OutputDelta = Array1<f32>;
    type InputDelta = Array1<f32>;
    type TrainDelta = Ndeep<OwnedRepr<f32>, D2>;

    fn backward(
        &self,
        input: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let mut input_delta = self.0.to_owned();
        for (mut delta_slice, &output) in input_delta.outer_iter_mut().zip(output_delta.iter()) {
            delta_slice *= output;
        }
        let input_delta = input_delta.sum_axis(Axis(0));

        // TODO: This works but would be vastly more efficient if the input was broadcasted into Zip::map_collect().
        // TODO: It also looks like this can be implemented purely with broadcasting the
        // input and output delta and multiplication.
        let mut train_delta: Array2<f32> = Array::from_shape_vec(
            self.0.raw_dim(),
            input.iter().cloned().cycle().take(self.0.len()).collect(),
        )
        .expect("mli-dense: input could not be broadcasted into train_delta");

        for (mut delta_slice, &output) in train_delta.outer_iter_mut().zip(output_delta.iter()) {
            delta_slice *= output;
        }

        (input_delta, Ndeep(train_delta))
    }
}

impl<S> Train for Dense1<S>
where
    S: Data<Elem = f32>,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0 += &train_delta.0;
    }
}
