use mli::{Backward, Forward, Train};
use ndarray::{Array, Dimension, azip};
use num_traits::Zero;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct MapMany<G, D: Dimension>(pub Array<G, D>, PhantomData<D>);

impl<G, D: Dimension> MapMany<G, D> {
    pub fn new(gs: Array<G, D>) -> Self {
        Self(gs, PhantomData)
    }
}

impl<G, D: Dimension> Forward for MapMany<G, D>
where
    G: Forward,
    G::Internal: Clone + Zero,
    G::Output: Clone + Zero,
{
    type Input = Array<G::Input, D>;
    type Internal = Array<G::Internal, D>;
    type Output = Array<G::Output, D>;

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        let mut internals = Array::zeros(input.raw_dim());
        let mut outputs = Array::zeros(input.raw_dim());
        azip!((internal in &mut internals, output in &mut outputs, x in input.view(), g in self.0.view()) {
            let (internal_v, output_v) = g.forward(x);
            *internal = internal_v;
            *output = output_v;
        });
        (internals, outputs)
    }
}

impl<G, D: Dimension> Backward for MapMany<G, D>
where
    G: Backward,
    G::Internal: Clone + Zero,
    G::Output: Clone + Zero,
    G::TrainDelta: Clone + Zero,
    G::InputDelta: Clone + Zero,
{
    type OutputDelta = Array<G::OutputDelta, D>;
    type InputDelta = Array<G::InputDelta, D>;
    type TrainDelta = Array<G::TrainDelta, D>;

    fn backward(
        &self,
        inputs: &Self::Input,
        internals: &Self::Internal,
        output_deltas: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let mut input_deltas = Array::zeros(inputs.raw_dim());
        let mut train_deltas = Array::zeros(inputs.raw_dim());
        azip!((input_delta in &mut input_deltas, train_delta in &mut train_deltas, input in inputs, internal in internals, output_delta in output_deltas, g in &self.0) {
            let (input_delta_v, train_delta_v) = g.backward(input, internal, output_delta);
            *input_delta = input_delta_v;
            *train_delta = train_delta_v;
        });
        (input_deltas, train_deltas)
    }
}

impl<G, D: Dimension> Train for MapMany<G, D>
where
    G: Train,
    G::Internal: Clone + Zero,
    G::Output: Clone + Zero,
    G::TrainDelta: Clone + Zero,
    G::InputDelta: Clone + Zero,
{
    fn train(&mut self, train_deltas: &Self::TrainDelta) {
        azip!((g in &mut self.0, train_delta in train_deltas) {
            g.train(train_delta);
        });
    }
}
