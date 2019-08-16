use float_ord::FloatOrd;
use mli::*;

fn sigmoid(n: f32) -> f32 {
    (1.0 + n.exp()).recip()
}

pub struct Relu;

impl Forward<f32> for Relu {
    type Out = f32;

    fn forward(&self, input: f32) -> f32 {
        std::cmp::max(FloatOrd(0.0), FloatOrd(input)).0
    }
}

pub struct ReluSoftplus;

impl Forward<f32> for ReluSoftplus {
    type Out = f32;

    fn forward(&self, input: f32) -> f32 {
        std::cmp::max(FloatOrd(0.0), FloatOrd(input)).0
    }
}

impl Backward<f32> for ReluSoftplus {
    type Delta = f32;

    fn backward(&self, input: f32) -> f32 {
        sigmoid(input)
    }
}
