extern crate rand;
pub mod mep;
pub use mep::Mep;

trait GeneticAlgorithm {
    fn mate(parents: (&Self, &Self)) -> Self;
}
