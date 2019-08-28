use mli::*;

fn logistic(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip() - 0.5
}

fn logistic_derivative(n: f32) -> f32 {
    let sig = logistic(n);
    sig * (1.0 - sig)
}

#[derive(Copy, Clone, Debug)]
pub struct Logistic;

impl Forward for Logistic {
    type Input = f32;
    type Internal = ();
    type Output = f32;

    fn forward(&self, &input: &f32) -> ((), f32) {
        ((), logistic(input))
    }
}

impl Backward for Logistic {
    type OutputDelta = f32;
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

impl Train for Logistic {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
