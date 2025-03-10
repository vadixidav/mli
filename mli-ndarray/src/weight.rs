use crate::Ndeep;
use mli::*;
use ndarray::{Array, Dimension, OwnedRepr, azip};
use num_traits::Float;

#[derive(Clone, Debug)]
pub struct Weight<S, D: Dimension>(Array<S, D>);

impl<S, D: Dimension> Weight<S, D> {
    pub fn new(weights: Array<S, D>) -> Self {
        Self(weights)
    }
}

impl<S: Float, D: Dimension> Forward for Weight<S, D> {
    type Input = Array<S, D>;
    type Internal = EmptyData;
    type Output = Array<S, D>;

    fn forward(&self, input: &Self::Input) -> (EmptyData, Self::Output) {
        let Self(weights) = self;
        assert_eq!(
            input.raw_dim(),
            weights.raw_dim(),
            "input shape does not match biases"
        );

        (EmptyData, weights * input)
    }
}

impl<S: Float, D: Dimension> Backward for Weight<S, D> {
    type OutputDelta = Array<S, D>;
    type InputDelta = Array<S, D>;
    type TrainDelta = Ndeep<OwnedRepr<S>, D>;

    fn backward(
        &self,
        input: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let Self(weights) = self;
        assert_eq!(
            output_delta.raw_dim(),
            weights.raw_dim(),
            "input shape does not match biases"
        );
        (output_delta * weights, Ndeep(output_delta * input))
    }
}

impl<S: Float, D: Dimension> Train for Weight<S, D> {
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        azip!((s in &mut self.0, &d in &train_delta.0) {
            *s = *s + d;
        });
    }
}
