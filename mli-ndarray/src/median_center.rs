use mli::*;
use ndarray::{Array, Dimension};
use ordered_float::{FloatCore, OrderedFloat};
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct MedianCenter<S, D>(PhantomData<(S, D)>);

impl<S, D> MedianCenter<S, D> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<S, D> Default for MedianCenter<S, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: FloatCore, D: Dimension> Forward for MedianCenter<S, D> {
    type Input = Array<S, D>;
    type Internal = EmptyData;
    type Output = Array<S, D>;

    fn forward(&self, input: &Self::Input) -> (EmptyData, Self::Output) {
        let mut data = input.iter().copied().collect::<Vec<S>>();
        data.sort_unstable_by_key(|&v| OrderedFloat(v));
        let approx_median = data[data.len() / 2];
        // In theory we could save the index and propogate the gradients back to the median value.
        // While this would be technically correct, there is a good chance it would cause issues.
        (EmptyData, input.mapv(|v| v - approx_median))
    }
}

impl<S: FloatCore, D: Dimension> Backward for MedianCenter<S, D> {
    type OutputDelta = Array<S, D>;
    type InputDelta = Array<S, D>;
    type TrainDelta = EmptyData;

    fn backward(
        &self,
        _: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (output_delta.clone(), EmptyData)
    }
}

impl<S: FloatCore, D: Dimension> Train for MedianCenter<S, D> {
    fn train(&mut self, _train_delta: &Self::TrainDelta) {}
}
