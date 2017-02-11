#![no_std]

/// Interface for algorithms which implement the capability to perform stateless computation.
pub trait Stateless<'a, I, O> {
    fn process(&'a self, input: I) -> O;
}

/// Interface for algorithms which implement the capability to perform stateful computation.
pub trait Stateful<'a, I, O> {
    fn process(&'a mut self, input: I) -> O;
}

/// Interface for algorithms which are consumed when computing.
pub trait Instruction<I, O> {
    fn process(self, input: I) -> O;
}

/// Any stateless algorithm can also be used in the same context as a stateful algorithm.
impl<'a, I, O> Stateful<'a, I, O> for Stateless<'a, I, O> {
    fn process(&'a mut self, input: I) -> O {
        Stateless::process(self, input)
    }
}

/// Implement this on algorithms which can be trained online.
///
/// This assumes that the algorithm stores internal state regarding previous training information.
pub trait Online<S> {
    fn train_online(&mut self, set: S);
}

/// Like `Online`, but also requires an Rng.
pub trait OnlineRand<S, R> {
    fn train_online_rng(&mut self, set: S, rng: &mut R);
}

impl<S, R, O> OnlineRand<S, R> for O
    where O: Online<S>
{
    fn train_online_rng(&mut self, set: S, _: &mut R) {
        O::train_online(self, set)
    }
}

/// Implement this on algorithms which can be trained offline.
///
/// Notice that this also creates the algorithm, rather than updating it. For training algorithms
/// after they have been created, use `Online`.
pub trait Offline<S> {
    fn train_offline(set: S) -> Self;
}

/// Like `Offline`, but also requires an Rng.
pub trait OfflineRand<S, R> {
    fn train_offline_rng(set: S, rng: &mut R) -> Self;
}

impl<S, R, O> OfflineRand<S, R> for O
where O: Offline<S>
{
    fn train_offline_rng(set: S, _: &mut R) -> Self {
        O::train_offline(set)
    }
}

/// Things which can be mated.
pub trait Mate {
    /// The mate function takes two parent references and returns a new child.
    fn mate(&self, rhs: &Self) -> Self;
}

/// Things which can be mated using a source of randomness.
pub trait MateRand<R> {
    /// The mate function takes two parent references and an Rng (`r`), then returns a new child.
    /// So long as `r` is deterministic, the output is also deterministic as no other sources of
    /// randomness are involved.
    fn mate(&self, rhs: &Self, rng: &mut R) -> Self;
}

impl<R, M> MateRand<R> for M
    where M: Mate
{
    fn mate(&self, rhs: &Self, _: &mut R) -> Self {
        M::mate(self, rhs)
    }
}

/// Things which can be mutated.
pub trait Mutate<R> {
    /// Perform a unit mutation.
    fn mutate(&mut self, rng: &mut R);
}
