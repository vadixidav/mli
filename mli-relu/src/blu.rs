use mli::*;
use nalgebra::Vector2;

#[derive(Copy, Clone, Debug)]
pub struct Blu {
    alpha: f32,
    beta: f32,
}

impl Blu {
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }
}

const EPSILON: f32 = 0.001;

fn blu(a: f32, b: f32, x: f32) -> f32 {
    b * ((x.powi(2) + a.powi(2) + EPSILON).sqrt() - a) + x
}

fn dblu_dx(a: f32, b: f32, x: f32) -> f32 {
    b * x / (x.powi(2) + a.powi(2) + EPSILON).sqrt() + 1.0
}

fn dblu_da(a: f32, b: f32, x: f32) -> f32 {
    b * (a / (x.powi(2) + a.powi(2) + EPSILON).sqrt() - 1.0)
}

fn dblu_db(a: f32, x: f32) -> f32 {
    (x.powi(2) + a.powi(2) + EPSILON).sqrt() - a
}

impl Forward for Blu {
    type Input = f32;
    type Internal = EmptyData;
    type Output = f32;

    fn forward(&self, &input: &f32) -> (EmptyData, f32) {
        (EmptyData, blu(self.alpha, self.beta, input))
    }
}

impl Backward for Blu {
    type OutputDelta = f32;
    type InputDelta = f32;
    type TrainDelta = Vector2<f32>;

    fn backward(
        &self,
        &input: &f32,
        _: &EmptyData,
        &output_delta: &f32,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (
            dblu_dx(self.alpha, self.beta, input) * output_delta,
            Vector2::new(
                dblu_da(self.alpha, self.beta, input),
                dblu_db(self.alpha, input),
            ) * output_delta,
        )
    }
}

impl Train for Blu {
    fn train(&mut self, v: &Self::TrainDelta) {
        self.alpha += v.x;
        self.beta += v.y;
    }
}
