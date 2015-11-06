use std::collections::BTreeSet;
use std::cmp;
use rand::Rng;
use super::GeneticAlgorithm;

/*
A multi-expression program represented using a series of operations that can reuse results of previous operations.
*/
pub struct Mep<Ins> {
    instructions: Vec<Ins>,
    unit_mutate_size: usize,
    crossover_points: usize,
}

impl<Ins> Clone for Mep<Ins>
    where Ins: Clone {
    fn clone(&self) -> Self {
        Mep{instructions: self.instructions.clone(), unit_mutate_size: self.unit_mutate_size,
            crossover_points: self.crossover_points}
    }
}

impl<Ins> Mep<Ins> {
    //Generates a new Mep with a particular size and takes a closure to generate random instructions
    pub fn new<I>(unit_mutate_size: usize, crossover_points: usize, instruction_iter: I) -> Mep<Ins>
        where I: Iterator<Item=Ins> {
        Mep{instructions: instruction_iter.collect(), unit_mutate_size: unit_mutate_size,
            crossover_points: crossover_points}
    }
}

impl<R, Ins> GeneticAlgorithm<R, Ins, Vec<Ins>> for Mep<Ins> where R: Rng, Ins: Clone {
    fn mate(parents: (&Mep<Ins>, &Mep<Ins>), rng: &mut R) -> Mep<Ins> {
        //Get the smallest of the two lengths
        let total_instructions = cmp::min(parents.0.instructions.len(), parents.1.instructions.len());
        Mep{instructions:
            //Generate a randomly sized sequence between 1 and half of the total possible crossover points
            (0..rng.gen_range(1, total_instructions / 2))
            //Map these to random crossover points
            .map(|_| rng.gen_range(0, total_instructions))
            //Add total_instructions at the end so we can generate a range with it
            .chain(Some(total_instructions))
            //Sort them by value into BTree, which removes duplicates and orders them
            .fold(BTreeSet::new(), |mut set, i| {set.insert(i); set})
            //Add total_instructions at the end so we can generate a range with it
            //.insert(total_instructions)
            //Iterate over the sorted values
            .iter()
            //Turn every copy of two, prepending a 0, into a range
            .scan(0, |prev, x| {let out = Some(*prev..*x); *prev = *x; out})
            //Enumerate by index to differentiate odd and even values
            .enumerate()
            //Map even pairs to ranges in parent 0 and odd ones to ranges in parent 1 and expand the ranges
            .flat_map(|(index, range)| {
                {if index % 2 == 0 {parents.0} else {parents.1}}.instructions[range].iter().cloned()
            })
            //Collect all the groups from each parent
            .collect(),
            unit_mutate_size: if parents.0.unit_mutate_size < parents.1.unit_mutate_size {
                rng.gen_range(parents.0.unit_mutate_size, parents.1.unit_mutate_size + 1)
            } else {
                rng.gen_range(parents.1.unit_mutate_size, parents.0.unit_mutate_size + 1)
            },
            crossover_points: if parents.0.crossover_points < parents.1.crossover_points {
                rng.gen_range(parents.0.crossover_points, parents.1.crossover_points + 1)
            } else {
                rng.gen_range(parents.1.crossover_points, parents.0.crossover_points + 1)
            },
        }
    }

    /*
    The Mep mutate function operates using the unit_mutate_size. This variable specifies the amount of instructions for
    which a single mutation is expect to occour every time mutate is called. This variable can be mutated inside of
    mutate, in which case it may never go below 1, but may tend towards infinity in increments of 1. This variable is
    implemented as a u64 to permit it to expand unbounded to mutation levels that are so low that mutations virtually
    never happen. Allowing this to mutate allows species to find the equilibrium between genomic adaptability and
    stability. If a species develops information gathering, then it can adapt intellegently, making it possibly more
    beneficial to operate at lower mutation rates. Setting the default mutation rate for species properly, or allowing
    it to adapt as the simulation continues, permits species to survive more frequently that are randomly generated.

    Likewise, the functions for random instruction generation and mutation can be adapted as the simulation continues
    to optimize the generation of more desireable random mutations and generations. For instance, instructions that
    occur more frequently should be generated randomly more frequently.
    */
    fn mutate<F>(&mut self, rng: &mut R, mut mutator: F) where F: FnMut(&mut Ins) {
        //Mutate unit_mutate_size
        if rng.gen_range(0, self.unit_mutate_size) == 0 {
            //Make it possibly go up or down by 1
            match rng.gen_range(0, 2) {
                0 => self.unit_mutate_size += 1,
                1 => self.unit_mutate_size -= 1,
                _ => unreachable!(),
            }
            //It isnt allowed to be 0
            if self.unit_mutate_size == 0 {
                self.unit_mutate_size = 1;
            }
        }
        //Mutate crossover_points
        if rng.gen_range(0, self.unit_mutate_size) == 0 {
            //Make it possibly go up or down by 1
            match rng.gen_range(0, 2) {
                0 => self.crossover_points += 1,
                1 => self.crossover_points -= 1,
                _ => unreachable!(),
            }
            //It isnt allowed to be 0
            if self.crossover_points == 0 {
                self.crossover_points = 1;
            }
        }

        //Mutate the instructions using the mutator
        loop {
            //Choose a random location in the instructions and then add a random value up to the unit_mutate_size
            let choice = rng.gen_range(0, self.instructions.len()) + rng.gen_range(0, self.unit_mutate_size);
            //Whenever we choose a location outside the vector reject the choice and end mutation
            if choice >= self.instructions.len() {
                break;
            }
            //Mutate the valid location using the instruction mutator
            mutator(&mut self.instructions[choice]);
        }
    }

    fn call<F>(&self, program: F) where F: FnOnce(&Vec<Ins>) {
        program(&self.instructions);
    }
}

#[cfg(test)]
mod tests {
    use rand::{Isaac64Rng, SeedableRng, Rng};
    use super::*;
    use super::super::GeneticAlgorithm;

    #[test]
    fn mep_new() {
        let a: Mep<u32> = Mep::new(3, 3, 0..8);

        assert_eq!(a.instructions, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn mep_crossover() {
        let mut rng = Isaac64Rng::from_seed(&[1, 2, 3, 4]);
        let len = 10;
        let (a, b) = {
            let mut clos = || Mep::new(3, 3, rng.gen_iter::<u32>().map(|x| x % 10).take(len));
            (clos(), clos())
        };
        let old_rngs: Vec<_> = rng.clone().gen_iter::<u32>().take(5).collect();
        let mut c = Mep::mate((&a, &b), &mut rng);
        //Ensure that rng was borrowed mutably
        assert!(rng.clone().gen_iter::<u32>().take(5).collect::<Vec<_>>() != old_rngs);

        c.mutate(&mut rng, |ins: &mut u32| *ins = 2);
        //c.call(|x: &Vec<u32>| println!("{}", x.len()));

        assert_eq!(c.instructions, [0, 7, 5, 4, 2, 8, 5, 6, 0, 2]);
    }
}
