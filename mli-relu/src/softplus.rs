use mli::*;

#[derive(Copy, Clone, Debug)]
pub struct Softplus;

fn sigmoid(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

fn softplus(n: f32) -> f32 {
    (1.0 + n.exp()).ln()
}

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
