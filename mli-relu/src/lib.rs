use float_ord::FloatOrd;
use mli::*;

fn sigmoid(n: f32) -> f32 {
    (1.0 + n.exp()).recip()
}

pub struct Relu;

impl Forward for Relu {
    type Input = f32;
    type Output = f32;

    fn forward(&self, input: f32) -> f32 {
        std::cmp::max(FloatOrd(0.0), FloatOrd(input)).0
    }
}

pub struct ReluSoftplus;

impl Forward for ReluSoftplus {
    type Input = f32;
    type Output = f32;

    fn forward(&self, input: f32) -> f32 {
        Relu.forward(input)
    }
}

impl Static for ReluSoftplus {
    type Derivative = f32;

    fn partial(&self, input: f32) -> f32 {
        sigmoid(input)
    }
}
