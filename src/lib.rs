#![feature(plugin)]
#![feature(proc_macro)]

#[macro_use]
extern crate serde_derive;
extern crate rand;

pub mod mep;
pub use mep::*;

/// Interface for algorithms which implement the capability to perform stateless computation.
pub trait Stateless<'a, I, O> {
    fn process(&'a self, input: I) -> O;
}

/// Interface for algorithms which implement the capability to perform stateful computation.
pub trait Stateful<'a, I, O> {
    fn process(&'a mut self, input: I) -> O;
}

/// Any stateless algorithm can also be used in the same context as a stateful algorithm.
impl<'a, I, O> Stateful<'a, I, O> for Stateless<'a, I, O> {
    fn process(&'a mut self, input: I) -> O {
        Stateless::process(self, input)
    }
}

/// Interface for algorithms which can be trained online.
pub trait Online<S> {
    fn learn_online(&mut self, set: S);
}

/// Interface for algorithms which can be trained offline.
pub trait Offline<S> {
    fn learn_offline(&mut self, sets: S);
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
