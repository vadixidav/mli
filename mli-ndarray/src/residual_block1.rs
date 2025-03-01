use mli::{Backward, Forward, Train};
use ndarray::{Array1, ArrayBase, Data, OwnedRepr};
use std::marker::PhantomData;

type D1 = ndarray::Ix1;

#[derive(Clone, Debug)]
pub struct ResidualBlock1<G, S>(pub G, PhantomData<S>);

impl<G, S> ResidualBlock1<G, S> {
    pub fn new(graph: G) -> Self {
        Self(graph, PhantomData)
    }
}

impl<G, S> Forward for ResidualBlock1<G, S>
where
    S: Data<Elem = f32>,
    G: Forward<Input = ArrayBase<S, D1>, Output = Array1<f32>>,
{
    type Input = G::Input;
    type Internal = G::Internal;
    type Output = ArrayBase<OwnedRepr<f32>, D1>;

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        let (block_internal, block_output) = self.0.forward(input);
        let final_output = input.to_owned() + &block_output;
        (block_internal, final_output)
    }
}

impl<G, S> Backward for ResidualBlock1<G, S>
where
    S: Data<Elem = f32>,
    G: Forward<Input = ArrayBase<S, D1>, Output = Array1<f32>>
        + Backward<OutputDelta = Array1<f32>, InputDelta = Array1<f32>>,
{
    type OutputDelta = G::OutputDelta;
    type InputDelta = G::InputDelta;
    type TrainDelta = G::TrainDelta;

    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let (block_input_delta, block_train_delta) = self.0.backward(input, internal, output_delta);
        (block_input_delta + output_delta, block_train_delta)
    }
}

impl<G, S> Train for ResidualBlock1<G, S>
where
    S: Data<Elem = f32>,
    G: Forward<Input = ArrayBase<S, D1>, Output = Array1<f32>>
        + Backward<OutputDelta = Array1<f32>, InputDelta = Array1<f32>>
        + Train,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0.train(train_delta);
    }
}
