use float_ord::FloatOrd;
use mli::*;

#[derive(Copy, Clone, Debug)]
pub struct Relu;

fn relu(n: f32) -> f32 {
    std::cmp::max(FloatOrd(0.0), FloatOrd(n)).0
}

fn heaviside(n: f32) -> f32 {
    if n.is_sign_positive() { 1.0 } else { 0.0 }
}

impl Forward for Relu {
    type Input = f32;
    type Internal = EmptyData;
    type Output = f32;

    fn forward(&self, &input: &f32) -> (EmptyData, f32) {
        (EmptyData, relu(input))
    }
}

impl Backward for Relu {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = EmptyData;

    fn backward(
        &self,
        &input: &f32,
        _: &EmptyData,
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (heaviside(input) * output_delta, EmptyData)
    }
}

impl Train for Relu {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
