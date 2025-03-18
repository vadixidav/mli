use mli::*;

#[derive(Copy, Clone, Debug)]
pub struct Abs;

impl Forward for Abs {
    type Input = f32;
    type Internal = EmptyData;
    type Output = f32;

    fn forward(&self, &input: &f32) -> (EmptyData, f32) {
        (EmptyData, input.abs())
    }
}

impl Backward for Abs {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = EmptyData;

    fn backward(
        &self,
        &input: &f32,
        _: &EmptyData,
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (
            if input.is_sign_positive() { 1.0 } else { -1.0 } * output_delta,
            EmptyData,
        )
    }
}

impl Train for Abs {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
