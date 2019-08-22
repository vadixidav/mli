use float_ord::FloatOrd;
use mli::*;

fn sigmoid(n: f32) -> f32 {
    (1.0 + n.exp()).recip()
}

fn relu(n: f32) -> f32 {
    std::cmp::max(FloatOrd(0.0), FloatOrd(n)).0
}

#[derive(Copy, Clone, Debug)]
pub struct ReluSoftplus;

impl Forward<f32> for ReluSoftplus {
    type Internal = ();
    type Output = f32;

    fn forward(&self, &input: &f32) -> ((), f32) {
        ((), relu(input))
    }
}

impl Backward<f32, f32> for ReluSoftplus {
    type InputDelta = f32;
    type TrainDelta = ();

    fn backward(
        &self,
        &input: &f32,
        _: &(),
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (sigmoid(input) * output_delta, ())
    }
}

impl Train<f32, f32> for ReluSoftplus {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
