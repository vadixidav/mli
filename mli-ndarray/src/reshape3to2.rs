use mli::*;
use ndarray::{Array2, Array3, ArrayBase, Axis, Data};
use std::marker::PhantomData;

type D3 = ndarray::Ix3;

#[derive(Clone, Debug)]
pub struct Reshape3to2<S>(PhantomData<S>);

impl<S> Reshape3to2<S>
where
    S: Data,
{
    pub fn new() -> Self {
        Default::default()
    }
}

impl<S> Default for Reshape3to2<S>
where
    S: Data,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S> Forward for Reshape3to2<S>
where
    S: Data,
    S::Elem: Clone,
{
    type Input = ArrayBase<S, D3>;
    type Internal = ();
    type Output = Array2<S::Elem>;

    fn forward(&self, input: &Self::Input) -> ((), Self::Output) {
        ((), input.to_owned().index_axis_move(Axis(0), 0))
    }
}

impl<S> Backward for Reshape3to2<S>
where
    S: Data,
    S::Elem: Clone,
{
    type OutputDelta = Array2<S::Elem>;
    type InputDelta = Array3<S::Elem>;
    // TODO: This is bad, but done to allow multiplying by f32.
    type TrainDelta = f32;

    fn backward(
        &self,
        _: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (output_delta.to_owned().insert_axis(Axis(0)), 0.0)
    }
}

impl<S> Train for Reshape3to2<S>
where
    S: Data,
    S::Elem: Clone,
{
    fn train(&mut self, _: &Self::TrainDelta) {}
}
