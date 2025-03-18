use mli::*;

fn logistic(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

fn logistic_derivative(n: f32) -> f32 {
    let en = logistic(n);
    en * (1.0 - en)
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

#[derive(Copy, Clone, Debug)]
pub struct LogisticCentered;

impl Forward for LogisticCentered {
    type Input = f32;
    type Internal = EmptyData;
    type Output = f32;

    fn forward(&self, &input: &f32) -> (EmptyData, f32) {
        (EmptyData, 2.0 * logistic(input) - 1.0)
    }
}

impl Backward for LogisticCentered {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = f32;

    fn backward(
        &self,
        &input: &f32,
        _: &EmptyData,
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (2.0 * logistic_derivative(input) * output_delta, 0.0)
    }
}

impl Train for LogisticCentered {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
