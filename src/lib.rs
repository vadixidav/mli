extern crate rand;
pub mod mep;
pub use mep::Mep;
use rand::Rng;

/*
GeneticAlgorithm is a trait that allows genetic manipulation. Genetic algorithms require duplication, which is why
there is a requirement for Clone.

It is parameterized with an Rng (R) type so that the algorithms can extract random data in the interface.

It is also parameterized with a program (P). This is some sort of function or closure that is accepted into the call
of the genetic algorithm. Each and every GeneticAlgorithm has its own unique program signature that allows it to
execute individual instructions or the entire program. GeneticAlgorithms that implement a specific type of program
may implement further specific types of GeneticAlgorithms, so that they can be dynamically dispatched if desired.

M is a mutator function. This is a function that is also parameterized so that mutation can be implemented for different
GeneticAlgorithms.
*/
pub trait GeneticAlgorithm<R, Ins, C> : Clone where R: Rng {
    //The mate function takes a tuple of two parent references and returns a new child; this can be non-deterministic.
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self;
    //The mutate function performs a unit mutation. A single, several, or no actual mutations may occour.
    fn mutate<F>(&mut self, rng: &mut R, mutator: F) where F: FnMut(&mut Ins);
    //The call function allows a closure to aquire a program so it can operate on it.
    fn call<F>(&self, program: F) where F: FnOnce(&C);
}
