//! MLI intends to provide modern, data-driven abstractions for machine learning.
//!
//! MLI provides traits that work much like Combine's `Parser`,
//! Serde's `Serialize` and `Deserialize`, std's `Iterator`, and future's `Future` and `Stream`.
//! One should only need to have to create tensor processing primitives and then string them together to craft
//! a large system that continues to be useful as a primitive in an even larger system.
//!
//! ## Goals
//! - Very fast CPU forwards and backwards propogation.
//! - Abstraction over multiple backends.
//! - Automatic serialization/deserialization  derivations of all state data (using serde).
//! - Get complete reuse out of tensor code without any overhead for abstractions.
//! - Allow building tensor pipelines that can talk over streams to allow multi-node setups.
//!   - It is a non-goal to implement the orchestration of these pipelines in this crate.
//!
//! ## Const Generics
//! Const generics must land before this API can be stabilized. This API will not be able to interact
//! with tensor sizes and dimensions to allow static checking without const generics. Until that lands in stable
//! and nalgebra is updated to use it, this library will continue to be in an unstable state.

#![no_std]

/// This trait indicates support of backwards propogation.
///
/// This trait also contains methods to perform training if training is possible.
/// If training is not possible, this trait can still be implemented with those definitions
/// being empty. In that case, machine learning algorithms will still be able to back propogate
/// over this operation, but training it will be a no-op.
pub trait Backward: Forward {
    type InputDerivative;
    type InternalDerivative;

    /// `partials` produces the partial derivatives `df/dx` and `df/dv` where:
    ///
    /// - `f` is the output
    /// - `x` is the input
    /// - `v` is the internal variables
    ///
    /// Either of these can be an approximation, particularly if the function is not differentiable.
    fn partials(&self, input: Self::Input) -> (Self::InputDerivative, Self::InternalDerivative);

    /// `train` takes in `dfdv`, `dedf`, and `rate` where:
    ///
    /// - `dfdv` is the partial derivative `df/dv`
    /// - `dedf` is the partial derivative `dE/df`
    /// - `f` is the output
    /// - `E` is the error
    /// - `v` is the internal variables
    fn train(&mut self, dfdv: Self::InternalDerivative, dedf: Self::Output, rate: f32);

    /// `partial_input` produces the partial derivative `df/dx`.
    ///
    /// See `partials` for more information.
    fn partial_input(&self, input: Self::Input) -> Self::InputDerivative {
        self.partials(input).0
    }

    /// `partial_internal` produces the partial derivative `df/dv`.
    ///
    /// See `partials` for more information.
    fn partial_internal(&self, input: Self::Input) -> Self::InternalDerivative {
        self.partials(input).1
    }
}

pub trait Static: Forward {
    type Derivative;

    /// `partial` produces the partial derivative `df/dx`
    /// where `f` is the output and `x` in the `input`.
    fn partial(&self, input: Self::Input) -> Self::Derivative;
}

impl<T> Backward for T
where
    T: Static,
{
    type InputDerivative = T::Derivative;
    type InternalDerivative = ();

    fn partials(&self, input: Self::Input) -> (Self::InputDerivative, Self::InternalDerivative) {
        (self.partial(input), ())
    }

    fn train(&mut self, _: Self::InternalDerivative, _: Self::Output, _: f32) {}
}

/// This trait is for algorithms that have an input and produce an output.
pub trait Forward {
    type Input;
    type Output;

    /// `forward` produces the output `f` given an `input`.
    fn forward(&self, input: Self::Input) -> Self::Output;
}
