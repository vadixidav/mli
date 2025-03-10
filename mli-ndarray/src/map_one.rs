use itertools::izip;
use mli::{Backward, Forward, Train};
use ndarray::{Array, Dimension};
use num_traits::Zero;
use std::{marker::PhantomData, ops::Add};

#[derive(Clone, Debug)]
pub struct MapOne<G, D>(pub G, PhantomData<D>);

impl<G, D> MapOne<G, D> {
    pub fn new(g: G) -> Self {
        Self(g, PhantomData)
    }
}

impl<G, D: Dimension> Forward for MapOne<G, D>
where
    G: Forward,
{
    type Input = Array<G::Input, D>;
    type Internal = Array<G::Internal, D>;
    type Output = Array<G::Output, D>;

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

impl<G, D: Dimension> Backward for MapOne<G, D>
where
    G: Backward,
    G::TrainDelta: Clone + Add + Zero,
{
    type OutputDelta = Array<G::OutputDelta, D>;
    type InputDelta = Array<G::InputDelta, D>;
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

impl<G, D: Dimension> Train for MapOne<G, D>
where
    G: Train,
    G::TrainDelta: Clone + Add + Zero,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0.train(train_delta);
    }
}
