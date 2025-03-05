use mli::*;

fn logistic(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

fn logistic_derivative(n: f32) -> f32 {
    let en = n.exp();
    en * (1.0 + en).powi(-2)
}

#[derive(Copy, Clone, Debug)]
pub struct Logistic;

impl Forward for Logistic {
    type Input = f32;
    type Internal = EmptyData;
    type Output = f32;

    fn forward(&self, &input: &f32) -> (EmptyData, f32) {
        (EmptyData, logistic(input))
    }
}

impl Backward for Logistic {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = f32;

    fn backward(
        &self,
        &input: &f32,
        _: &EmptyData,
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (logistic_derivative(input) * output_delta, 0.0)
    }
}

impl Train for Logistic {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
