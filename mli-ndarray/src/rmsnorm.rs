use mli::{Backward, ChainData, EmptyData, Forward, Train};
use ndarray::{Array, ArrayBase, Data, Dimension, Zip};
use num_traits::{Float, NumCast, One, Zero};
use std::{iter::Sum, marker::PhantomData};

/// This performs RmsNorm. This normalizes the entire layer by dividing by the RMS with a small epsilon.
///
/// Typical implementations also include a global or per-feature scaling mechanism called gamma.
/// This implementation does not provide gamma. If you want it, use a Weight layer.
#[derive(Clone, Debug)]
pub struct RmsNorm<S: Data, D: Dimension> {
    _phantom: PhantomData<(S, D)>,
}

impl<S: Data, D: Dimension> RmsNorm<S, D> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S: Data, D: Dimension> Default for RmsNorm<S, D>
where
    S::Elem: One + Zero,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Data, D: Dimension> Forward for RmsNorm<S, D>
where
    S::Elem: Float + Sum,
{
    type Input = ArrayBase<S, D>;
    /// The internal value is the normalized input tensor.
    type Internal = ChainData<S::Elem, S::Elem>;
    type Output = Array<S::Elem, D>;

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        let float_epsilon = <S::Elem as NumCast>::from(1e-5).unwrap();
        let recip_len = <S::Elem as NumCast>::from(input.len()).unwrap().recip();
        let variance = input.iter().map(|&v| v * v).sum::<S::Elem>() * recip_len;
        let recip_std = (variance + float_epsilon).sqrt().recip();
        let output = input.mapv(|v| v * recip_std);
        (ChainData(recip_std, recip_len), output)
    }
}

impl<S: Data, D: Dimension> Backward for RmsNorm<S, D>
where
    S::Elem: Float + Sum,
{
    type OutputDelta = Array<S::Elem, D>;
    type InputDelta = Array<S::Elem, D>;
    type TrainDelta = EmptyData;

    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let &ChainData(recip_std, recip_len) = internal;
        let recip_factor = recip_std.powi(3) * recip_len;

        let sum_dot = Zip::from(output_delta)
            .and(input)
            .fold(S::Elem::zero(), |acc, &d, &x| acc + d * x);

        let input_delta = Zip::from(output_delta)
            .and(input)
            .map_collect(|&od, &x| recip_std * od - recip_factor * x * sum_dot);

        (input_delta, EmptyData)
    }
}

impl<S: Data, D: Dimension> Train for RmsNorm<S, D>
where
    S::Elem: Float + Sum,
{
    fn train(&mut self, _train_delta: &Self::TrainDelta) {}
}
