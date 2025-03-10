use mli::{Backward, ChainData, Forward, Train};
use ndarray::{Array, ArrayBase, Data, Dimension, Zip};
use num_traits::{Float, NumCast, One, Zero};
use std::{iter::Sum, marker::PhantomData};

/// This performs LayerNorm. This normalizes the entire layer and then applies the same linear
/// function to all features. The linear function is learnable. This helps avoid vanishing and
/// exploding gradients during backpropogation by essentially propogating only the relative gradients
/// to input features to distribute the gradients more evenly, but ultimately allows the model to recover
/// the magnitude and apply a single bias. Since the single magnitude and bias receive gradients from all the
/// features instead of just one, they remain stable.
#[derive(Clone, Debug)]
pub struct LayerNorm<S: Data, D> {
    pub gamma: S::Elem,
    pub beta: S::Elem,
    _phantom: PhantomData<D>,
}

impl<S: Data, D> LayerNorm<S, D> {
    pub fn new_with_params(gamma: S::Elem, beta: S::Elem) -> Self {
        Self {
            gamma,
            beta,
            _phantom: PhantomData,
        }
    }
}

impl<S: Data, D> LayerNorm<S, D>
where
    S::Elem: One + Zero,
{
    pub fn new() -> Self {
        Self {
            gamma: S::Elem::one(),
            beta: S::Elem::zero(),
            _phantom: PhantomData,
        }
    }
}

impl<S: Data, D> Default for LayerNorm<S, D>
where
    S::Elem: One + Zero,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Data, D: Dimension> Forward for LayerNorm<S, D>
where
    S::Elem: Float + Sum,
{
    type Input = ArrayBase<S, D>;
    /// The internal value is the normalized input tensor.
    type Internal = Array<S::Elem, D>;
    type Output = Array<S::Elem, D>;

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        let float_epsilon = <S::Elem as NumCast>::from(1e-5).unwrap();
        let float_len = <S::Elem as NumCast>::from(input.len()).unwrap();
        let variance = input.iter().map(|&v| v * v).sum::<S::Elem>() / float_len;
        let mean = input.iter().copied().sum::<S::Elem>() / float_len;
        let recip_std = (variance + float_epsilon).sqrt().recip();
        let normalized = input.mapv(|v| (v - mean) * recip_std);
        let output = normalized.mapv(|v| v * self.gamma + self.beta);
        (normalized, output)
    }
}

impl<S: Data, D: Dimension> Backward for LayerNorm<S, D>
where
    S::Elem: Float + Sum,
{
    type OutputDelta = Array<S::Elem, D>;
    type InputDelta = Array<S::Elem, D>;
    type TrainDelta = ChainData<S::Elem, S::Elem>;

    fn backward(
        &self,
        _input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let norms = internal;

        let gamma_delta = norms
            .iter()
            .zip(output_delta.iter())
            .map(|(&v, &d)| v * d)
            .sum();

        let beta_delta = output_delta.iter().copied().sum();

        let float_len = <S::Elem as NumCast>::from(output_delta.len()).unwrap();

        let norm_deltas = output_delta.mapv(|v| self.gamma * v);

        let mean_norm_delta = norm_deltas.iter().copied().sum::<S::Elem>() / float_len;
        let mean_norm_norm_delta = norms
            .iter()
            .zip(norm_deltas.iter())
            .map(|(&n, &nd)| n * nd)
            .sum::<S::Elem>()
            / float_len;

        let input_delta = Zip::from(norms)
            .and(&norm_deltas)
            .map_collect(|&n, &nd| nd - mean_norm_delta - n * mean_norm_norm_delta);

        (input_delta, ChainData(gamma_delta, beta_delta))
    }
}

impl<S: Data, D: Dimension> Train for LayerNorm<S, D>
where
    S::Elem: Float + Sum,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.gamma = self.gamma + train_delta.0;
        self.beta = self.beta + train_delta.1;
    }
}
