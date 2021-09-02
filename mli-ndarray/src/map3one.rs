use itertools::izip;
use mli::{Backward, Forward, Train};
use ndarray::{Array, Array3};
use num_traits::Zero;
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Map3One<G>(pub G);

impl<G> Forward for Map3One<G>
where
    G: Forward,
{
    type Input = Array3<G::Input>;
    type Internal = Array3<G::Internal>;
    type Output = Array3<G::Output>;

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        let both = input.iter().map(|input| self.0.forward(input));
        let (internal_vec, output_vec) = both.fold(
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

impl<G> Backward for Map3One<G>
where
    G: Backward,
    G::TrainDelta: Clone + Add + Zero,
{
    type OutputDelta = Array3<G::OutputDelta>;
    type InputDelta = Array3<G::InputDelta>;
    type TrainDelta = G::TrainDelta;

    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let both = izip!(input.iter(), internal.iter(), output_delta.iter(),)
            .map(|(input, internal, output_delta)| self.0.backward(input, internal, output_delta));
        let (input_delta_vec, train_delta_vec) = both.fold(
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

impl<G> Train for Map3One<G>
where
    G: Train,
    G::TrainDelta: Clone + Add + Zero,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0.train(train_delta);
    }
}
