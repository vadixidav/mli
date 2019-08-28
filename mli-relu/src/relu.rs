use float_ord::FloatOrd;
use mli::*;

#[derive(Copy, Clone, Debug)]
pub struct Relu;

fn relu(n: f32) -> f32 {
    std::cmp::max(FloatOrd(0.0), FloatOrd(n)).0
}

fn heaviside(n: f32) -> f32 {
    (n.signum() + 1.0) * 0.5
}

impl Forward for Relu {
    type Input = f32;
    type Internal = ();
    type Output = f32;

    fn forward(&self, &input: &f32) -> ((), f32) {
        ((), relu(input))
    }
}

impl Backward for Relu {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = ();

    fn backward(
        &self,
        &input: &f32,
        _: &(),
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (heaviside(input) * output_delta, ())
    }
}

impl Train for Relu {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
