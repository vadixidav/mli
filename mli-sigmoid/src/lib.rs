use float_ord::FloatOrd;
use mli::*;

fn logistic(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

fn logistic_derivative(n: f32) -> f32 {
    let sig = logistic(n);
    sig * (1.0 - sig)
}

#[derive(Copy, Clone, Debug)]
pub struct Logistic;

impl Forward<f32> for Logistic {
    type Internal = ();
    type Output = f32;

    fn forward(&self, &input: &f32) -> ((), f32) {
        ((), logistic(input))
    }
}

impl Backward<f32, f32> for Logistic {
    type InputDelta = f32;
    type TrainDelta = ();

    fn backward(
        &self,
        &input: &f32,
        _: &(),
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (logistic_derivative(input) * output_delta, ())
    }
}

impl Train<f32, f32> for Logistic {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
