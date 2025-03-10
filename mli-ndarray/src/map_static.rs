use itertools::izip;
use mli::{Backward, EmptyData, Forward, Train};
use ndarray::{Array, Dimension};
use num_traits::Zero;
use std::{marker::PhantomData, ops::Add};

#[derive(Clone, Debug)]
pub struct MapStatic<G, D>(pub G, PhantomData<D>);

impl<G, D> MapStatic<G, D> {
    pub fn new(g: G) -> Self {
        Self(g, PhantomData)
    }
}

impl<G, D: Dimension> Forward for MapStatic<G, D>
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

impl<G, D: Dimension> Backward for MapStatic<G, D>
where
    G: Backward,
    G::TrainDelta: Clone + Add + Zero,
{
    type OutputDelta = Array<G::OutputDelta, D>;
    type InputDelta = Array<G::InputDelta, D>;
    type TrainDelta = EmptyData;

    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let input_deltas = izip!(input.iter(), internal.iter(), output_delta.iter(),).map(
            |(input, internal, output_delta)| self.0.backward(input, internal, output_delta).0,
        );
        let input_delta_array =
            Array::from_shape_vec(input.raw_dim(), input_deltas.collect()).unwrap();
        (input_delta_array, EmptyData)
    }
}

impl<G, D: Dimension> Train for MapStatic<G, D>
where
    G: Train,
    G::TrainDelta: Clone + Add + Zero,
{
    fn train(&mut self, _train_delta: &Self::TrainDelta) {}
}
