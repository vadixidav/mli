use crate::Ndeep;
use mli::*;
use ndarray::{Array, Dimension, OwnedRepr, azip};
use num_traits::Float;

#[derive(Clone, Debug)]
pub struct Bias<S, D: Dimension>(Array<S, D>);

impl<S, D: Dimension> Bias<S, D> {
    pub fn new(biases: Array<S, D>) -> Self {
        Self(biases)
    }
}

impl<S: Float, D: Dimension> Forward for Bias<S, D> {
    type Input = Array<S, D>;
    type Internal = EmptyData;
    type Output = Array<S, D>;

    fn forward(&self, input: &Self::Input) -> (EmptyData, Self::Output) {
        let Self(biases) = self;
        assert_eq!(
            input.raw_dim(),
            biases.raw_dim(),
            "input shape does not match biases"
        );

        (EmptyData, biases + input)
    }
}

impl<S: Float, D: Dimension> Backward for Bias<S, D> {
    type OutputDelta = Array<S, D>;
    type InputDelta = Array<S, D>;
    type TrainDelta = Ndeep<OwnedRepr<S>, D>;

    fn backward(
        &self,
        _: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (output_delta.clone(), Ndeep(output_delta.clone()))
    }
}

impl<S: Float, D: Dimension> Train for Bias<S, D> {
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        azip!((s in &mut self.0, &d in &train_delta.0) {
            *s = *s + d;
        });
    }
}
