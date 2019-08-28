//! MLI intends to provide modern, data-driven abstractions for machine learning.
//!
//! MLI provides traits that work much like Combine's `Parser` in that one should only
//! need to have to create tensor processing primitives and then string them together to craft
//! a large system that continues to be useful as a primitive in an even larger system.
//!
//! To understand MLI, we actually just need to go over some basic Rust concepts.
//!
//! In Rust, the following relationships relate the owned versions of data with the borrowed versions:
//! - `Vec<T>` -> `&mut [T]` -> `&[T]`
//! - `String` -> `&mut str` -> `&str`
//! - `PathBuf` -> `Path`
//! - `OsString` -> `OsStr`
//! - `FnOnce` -> `FnMut` -> `Fn`
//!
//! Most people don't think about the last one, because it is a trait, but it is critical to understand.
//! A `FnOnce` owns its enclosed environment. It can implement `FnMut` because since it owns the environment,
//! it can create a `&mut` to it. Similarly, a `FnMut` has a mutable borrow to its environment, thus it can
//! downgrade that mutable borrow to an immutable borrow.
//!
//! In MLI, the equivalent is:
//! - `Graph` -> `RunGraph`
//!
//! Let us decompose the notion of what a compute graph is. A compute graph simply takes several inputs,
//! produces several intermediary outputs, and then produces the actual outputs. Each one of these
//! intermediary outputs is important because they affect the gradients of the functions that came after them.
//! If we discard them, then they will have to be recomputed from the input. If we only have a simple
//! convolution operation as our whole graph, there are no intermediary computations, just the
//! input and the output. However, we are more likely to have several layers of convolution, activation
//! functions, splitting, and merging. If we have just two layers of convolution with no activation function,
//! the first layer produces an output which is necessary to calculate the gradient `𝛿output/𝛿filter`.
//!
//! We would like to have an abstraction that treats a whole graph, its inputs, intermediary computations, and
//! its outputs, the same way we would treat a single simple static function like a `tanh`. We would also like
//! that abstraction to be zero-cost and ideally parallel.
//!
//! This is where the magic happens. The traits [`Forward`] and [`Backward`] only require a `&self` as an
//! argument. This is because they use the trainable variables immutably to do a forward and backward
//! propogation. The trainable variables within the graph are stored as the [`Backward::TrainDelta`] type.
//! This means that if we have a graph composed of [`Forward`] and [`Backward`] items, we can compute
//! all of the deltas for a batch without ever mutating the graph, thus we can compute all of the deltas
//! **in parallel** for each batch. Since the learning rate is intended to be incorporated into the
//! delta before hand, you can just sum all of the deltas in the batch and train the network.
//!
//! The network which is trained though can't possibly be the one we distributed to all the threads,
//! however, due to Rust's ownership and borrowing. This is where the `Graph` and the `RunGraph` come in.
//! The `Graph` is the trainable graph which we have complete ownership over. The `RunGraph` immutably borrows
//! the `Graph`, so we can create a lot of `RunGraph` and propogate [`Forward`] and [`Backward`] across them
//! in multiple threads. We can then sum all of the deltas from the threads when they are done and then
//! update the `Graph`, which is no longer borrowed.
//!
//! Currently there are three main traits:
//! - [`Forward`]
//!     - Implemented on anything that can go into a forward graph.
//!     - Outputs intermediary computations (to compute gradients later) and final output.
//! - [`Backward`]
//!     - Implemented on anything that can go into a backward graph.
//!     - Propogates gradients backwards through the neural network.
//!     - This is provided the original input and any intermediary computations.
//!     - Propogates the change from the output to the input and trainable variables.
//!     - Is `&self` and does not do the actual training, so this can be ran in parallel.
//! - [`Train`]
//!     - Implemented on anything that can go into a training graph.
//!     - Uses the change to update the trainable variables.
//!         - Change can be normalized across a mini-batch before being passed.
//!     - Implemented on the `mutable` version of the graph.

#![no_std]

mod chain;
pub use chain::*;

/// This trait is for algorithms that have an input and produce an output.
pub trait Forward {
    type Input;
    type Internal;
    type Output;

    /// `forward` produces:
    /// - All intermediary values produced, which can be reused in the back propogation step
    /// - The output `f` given an `input`
    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output);

    /// `run` only produces the output from the input. The default implementation uses `forward`.
    /// You can make a custom implementation of this to avoid the overhead of producing and returning
    /// the internal variables.
    fn run(&self, input: &Self::Input) -> Self::Output {
        self.forward(input).1
    }
}

/// This trait indicates support of backwards propogation.
///
/// This trait also contains methods to perform training if training is possible.
/// If training is not possible, this trait can still be implemented with those definitions
/// being empty. In that case, machine learning algorithms will still be able to back propogate
/// over this operation, but training it will be a no-op.
pub trait Backward: Forward {
    type OutputDelta;
    type InputDelta;
    type TrainDelta;

    /// `partials` produces the change required in the input and trainable variables.
    ///
    /// Key:
    /// - `f` is the output
    /// - `x` is the input
    /// - `v` is the trainable variables
    /// - `E` is the loss or error
    /// - `delta` is equivalent to `Δf` or `-η * 𝛿E/𝛿f`
    ///
    /// This method should produce `(Δx, Δv)`:
    /// - `Δx = Δf * 𝛿f/𝛿x`
    /// - `Δv = Δf * 𝛿f/𝛿v`
    ///
    /// `𝛿f/𝛿x` and `𝛿f/𝛿v` can be an approximation, particularly if the function is not differentiable
    /// (e.g. ReLU and signum).
    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta);

    /// See [`Backward::backward`] for documentation.
    fn backward_input(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> Self::InputDelta {
        self.backward(input, internal, output_delta).0
    }

    /// See [`Backward::backward`] for documentation.
    fn backward_train(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> Self::TrainDelta {
        self.backward(input, internal, output_delta).1
    }
}

/// This trait is implemented on all operations that can be included in a trainable model.
///
/// This trait is implemented on all structures that implement `Backward` that are mutable
/// or `Static` even if they are immutable. This exists so that an immutable model can have
/// back propogation ran against in several different threads without the need for locking,
/// and then the mutable version of the model can implement `Train` to recieve the gradients
/// and be updated. If training is not possible, this trait can still be implemented with those
/// definitions being empty. In that case, machine learning algorithms will still be able to
/// incorporate this operation, but training it will be a no-op. This is necessary to
/// implement to be included in a trainable model.
pub trait Train: Backward {
    /// `train` takes in a train delta `Δv` and applies it to the trained variables.
    ///
    /// This should effectively perform `v += Δv`.
    fn train(&mut self, train_delta: &Self::TrainDelta);

    /// `propogate` should have the same effect as running `backwards` followed by `train`.
    ///
    /// See the source (by clicking the SRC link on the right) to see the simple definition.
    /// If the training is running on the same instance as the backpropogation, then `propogate`
    /// might perform better since it may not need to allocate more memory to update the trained variables.
    /// If an implementation can do this efficiently, it should create a custom implemenatation.
    fn propogate(
        &mut self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> Self::InputDelta {
        let (input_delta, train_delta) = self.backward(input, internal, output_delta);
        self.train(&train_delta);
        input_delta
    }
}

pub trait Graph: Train + Sized {
    fn chain<U>(self, other: U) -> Chain<Self, U> {
        Chain(self, other)
    }
}

impl<T> Graph for T where T: Train {}
