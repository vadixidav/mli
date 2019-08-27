use crate::{Relu, Softplus};

use mli::*;

#[derive(Copy, Clone, Debug)]
pub struct ReluSoftplus;

impl Forward<f32> for ReluSoftplus {
    type Internal = ();
    type Output = f32;

    fn forward(&self, input: &f32) -> ((), f32) {
        Relu.forward(input)
    }
}

impl Backward<f32, f32> for ReluSoftplus {
    type InputDelta = f32;
    type TrainDelta = ();

    fn backward(
        &self,
        input: &f32,
        _: &(),
        output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        Softplus.backward(input, &(), output_delta)
    }
}

impl Train<f32, f32> for ReluSoftplus {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
