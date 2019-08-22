use float_ord::FloatOrd;
use mli::*;

fn sigmoid(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

fn relu(n: f32) -> f32 {
    std::cmp::max(FloatOrd(0.0), FloatOrd(n)).0
}

fn softplus(n: f32) -> f32 {
    (1.0 + n.exp()).ln()
}

fn heaviside(n: f32) -> f32 {
    (n.signum() + 1.0) * 0.5
}

#[derive(Copy, Clone, Debug)]
pub struct Softplus;

impl Forward<f32> for Softplus {
    type Internal = ();
    type Output = f32;

    fn forward(&self, &input: &f32) -> ((), f32) {
        ((), softplus(input))
    }
}

impl Backward<f32, f32> for Softplus {
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

impl Train<f32, f32> for Softplus {
    fn train(&mut self, _: &Self::TrainDelta) {}
}

#[derive(Copy, Clone, Debug)]
pub struct Relu;

impl Forward<f32> for Relu {
    type Internal = ();
    type Output = f32;

    fn forward(&self, &input: &f32) -> ((), f32) {
        ((), relu(input))
    }
}

impl Backward<f32, f32> for Relu {
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

impl Train<f32, f32> for Relu {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
