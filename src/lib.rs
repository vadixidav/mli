#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]

extern crate serde;
extern crate serde_json;
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

/// Genetic is a trait that allows genetic manipulation. Genetic algorithms require duplication, which is why
/// there is a requirement for Clone. It is parameterized with an Rng (R) type so that the algorithms can extract random
/// data when mating so that mating can happen randomly.
///
/// Note: This API is highly likely to change before version 1.0.
pub trait Genetic<R, M>: Clone {
    /// The mate function takes a tuple of two parent references and an Rng, then returns a new child.
    fn mate(&self, rhs: &Self, rng: &mut R) -> Self;
    /// The mutate function performs a unit mutation. A single, several, or no actual mutations may occour.
    fn mutate(&mut self, mut mutator: M, rng: &mut R);
}
