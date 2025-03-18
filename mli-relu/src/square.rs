use mli::*;

#[derive(Copy, Clone, Debug)]
pub struct Square;

impl Forward for Square {
    type Input = f32;
    type Internal = EmptyData;
    type Output = f32;

    fn forward(&self, &input: &f32) -> (EmptyData, f32) {
        (EmptyData, input * input)
    }
}

impl Backward for Square {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = EmptyData;

    fn backward(
        &self,
        &input: &f32,
        _: &EmptyData,
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (2.0 * input * output_delta, EmptyData)
    }
}

impl Train for Square {
    fn train(&mut self, _: &Self::TrainDelta) {}
}
