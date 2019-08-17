use float_ord::FloatOrd;
use mli::*;

fn sigmoid(n: f32) -> f32 {
    (1.0 + n.exp()).recip()
}

pub struct Relu;

impl Forward<f32> for Relu {
    type O = f32;

    fn forward(&self, input: f32) -> f32 {
        std::cmp::max(FloatOrd(0.0), FloatOrd(input)).0
    }
}

pub struct ReluSoftplus;

impl Forward<f32> for ReluSoftplus {
    type O = f32;

    fn forward(&self, input: f32) -> f32 {
        Relu.forward(input)
    }
}

impl Backward<f32> for ReluSoftplus {
    type InputDerivative = f32;
    type InternalDerivative = ();
    type Error = f32;

    fn partials(&self, input: f32) -> (Self::InputDerivative, Self::InternalDerivative) {
        (sigmoid(input), ())
    }

    fn train(&mut self, _: Self::InternalDerivative, _: Self::Error, _: f32) {}
}
