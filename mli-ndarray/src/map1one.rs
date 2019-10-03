use itertools::izip;
use mli::{Backward, Forward, Train};
use ndarray::{Array, Array1};
use num_traits::Zero;
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Map1One<G>(pub G);

impl<G> Forward for Map1One<G>
where
    G: Forward,
{
    type Input = Array1<G::Input>;
    type Internal = Array1<G::Internal>;
    type Output = Array1<G::Output>;

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        let both_vec: Vec<(G::Internal, G::Output)> =
            input.iter().map(|input| self.0.forward(input)).collect();
        let (internal_vec, output_vec) = both_vec.into_iter().fold(
            (vec![], vec![]),
            |(mut internal_vec, mut output_vec), (internal, output)| {
                internal_vec.push(internal);
                output_vec.push(output);
                (internal_vec, output_vec)
            },
        );
        let internal_array = Array::from_shape_vec(input.raw_dim(), internal_vec).unwrap();
        let output_array = Array::from_shape_vec(input.raw_dim(), output_vec).unwrap();
        (internal_array, output_array)
    }
}

impl<G> Backward for Map1One<G>
where
    G: Backward,
    G::TrainDelta: Clone + Add + Zero,
{
    type OutputDelta = Array1<G::OutputDelta>;
    type InputDelta = Array1<G::InputDelta>;
    type TrainDelta = G::TrainDelta;

    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let both_vec: Vec<(G::InputDelta, G::TrainDelta)> =
            izip!(input.iter(), internal.iter(), output_delta.iter(),)
                .map(|(input, internal, output_delta)| {
                    self.0.backward(input, internal, output_delta)
                })
                .collect();
        let (input_delta_vec, train_delta_vec) = both_vec.into_iter().fold(
            (vec![], vec![]),
            |(mut input_delta_vec, mut train_delta_vec), (input_delta, train_delta)| {
                input_delta_vec.push(input_delta);
                train_delta_vec.push(train_delta);
                (input_delta_vec, train_delta_vec)
            },
        );
        let input_delta_array = Array::from_shape_vec(input.raw_dim(), input_delta_vec).unwrap();
        let train_delta_array = Array::from_shape_vec(input.raw_dim(), train_delta_vec).unwrap();
        (input_delta_array, train_delta_array.sum())
    }
}

impl<G> Train for Map1One<G>
where
    G: Train,
    G::TrainDelta: Clone + Add + Zero,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0.train(&train_delta);
    }
}
