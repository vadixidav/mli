use itertools::izip;
use mli::{Backward, Forward, Train};
use ndarray::{Array, Array2};
use num_traits::Zero;
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Map2One<G>(pub G);

impl<G, Input> Forward<Array2<Input>> for Map2One<G>
where
    G: Forward<Input>,
{
    type Internal = Array2<G::Internal>;
    type Output = Array2<G::Output>;

    fn forward(&self, input: &Array2<Input>) -> (Self::Internal, Self::Output) {
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

impl<G, Input, OutputDelta> Backward<Array2<Input>, Array2<OutputDelta>> for Map2One<G>
where
    G: Backward<Input, OutputDelta>,
{
    type InputDelta = Array2<G::InputDelta>;
    type TrainDelta = Array2<G::TrainDelta>;

    fn backward(
        &self,
        input: &Array2<Input>,
        internal: &Self::Internal,
        output_delta: &Array2<OutputDelta>,
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
        (input_delta_array, train_delta_array)
    }
}

impl<G, Input, OutputDelta> Train<Array2<Input>, Array2<OutputDelta>> for Map2One<G>
where
    G: Train<Input, OutputDelta>,
    G::TrainDelta: Clone + Add + Zero,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0.train(&train_delta.sum());
    }
}
