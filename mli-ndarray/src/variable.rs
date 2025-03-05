use crate::Ndeep;
use mli::{Backward, EmptyData, Forward, Train};
use ndarray::{Array, Dimension, OwnedRepr};
use std::ops::AddAssign;

/// This wraps a D-dimesnional ndarray that acts as a constant input in a neural network.
/// It can be learned through training as well. It has no input.
pub struct Variable<T, D>(pub Array<T, D>);

impl<T, D> Forward for Variable<T, D>
where
    T: Clone,
    D: Clone,
{
    type Input = EmptyData;
    type Internal = EmptyData;
    type Output = Array<T, D>;

    fn forward(&self, _: &Self::Input) -> (EmptyData, Self::Output) {
        (EmptyData, self.0.clone())
    }
}

impl<T, D> Backward for Variable<T, D>
where
    T: Clone,
    D: Clone,
{
    type OutputDelta = Array<T, D>;
    type InputDelta = EmptyData;
    type TrainDelta = Ndeep<OwnedRepr<T>, D>;

    fn backward(
        &self,
        _: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (EmptyData, Ndeep(output_delta.clone()))
    }
}

impl<T, D> Train for Variable<T, D>
where
    T: Clone + AddAssign,
    D: Clone + Dimension,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0 += &train_delta.0;
    }
}
