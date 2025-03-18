use mli::*;

#[derive(Copy, Clone, Debug)]
pub struct SingleBias(pub f32);

impl Forward for SingleBias {
    type Input = f32;
    type Internal = EmptyData;
    type Output = f32;

    fn forward(&self, &input: &f32) -> (EmptyData, f32) {
        (EmptyData, input + self.0)
    }
}

impl Backward for SingleBias {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = f32;

    fn backward(
        &self,
        _: &f32,
        _: &EmptyData,
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (output_delta, output_delta)
    }
}

impl Train for SingleBias {
    fn train(&mut self, &delta: &Self::TrainDelta) {
        self.0 += delta;
    }
}
