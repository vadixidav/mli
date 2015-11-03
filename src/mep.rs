use std::collections::BTreeSet;
use std::cmp;
//use super::GeneticAlgorithm;

/*
A multi-expression program represented using a series of operations that can reuse results of previous operations.
*/
pub struct Mep<Ins> {
    instructions: Vec<Ins>,
}

impl<Ins> Clone for Mep<Ins>
    where Ins: Clone {
    fn clone(&self) -> Self {
        Mep{instructions: self.instructions.clone()}
    }
}

impl<Ins> Mep<Ins> {
    //Generates a new Mep with a particular size and takes a closure to generate random instructions
    pub fn new<I>(instruction_iter: I) -> Mep<Ins>
        where I: Iterator<Item=Ins> {
        Mep{instructions: instruction_iter.collect()}
    }

    /*
    Performs a crossover that switches at random points in the genome.

    crossover_point_iter_generator takes an argument that specifies the end of the range to generate and produces
        an iterator that iterates over a finite amount of crossover points
    */
    pub fn crossover<'a, F, I>(parents: (&'a Mep<Ins>, &'a Mep<Ins>), crossover_point_iter_generator: F) -> Mep<Ins>
        where F: FnOnce(usize) -> I, I: Iterator<Item=usize>, Ins: Clone + 'a {
        //Get the smallest of the two lengths

        let total_instructions = cmp::min(parents.0.instructions.len(), parents.1.instructions.len());
        Mep{instructions: crossover_point_iter_generator(total_instructions)
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
            .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Isaac64Rng, SeedableRng, Rng};
    use super::*;

    #[test]
    fn mep_new() {
        let a: Mep<u32> = Mep::new(0..8);

        assert_eq!(a.instructions, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn mep_crossover() {
        let mut rng = Isaac64Rng::from_seed(&[1, 2, 3, 4]);
        let len = 10;
        let (a, b) = {
            let mut clos = || Mep::new(rng.gen_iter::<u32>().map(|x| x % 10).take(len));
            (clos(), clos())
        };
        let old_rngs: Vec<_> = rng.clone().gen_iter::<u32>().take(5).collect();
        let c = {
            let rng = &mut rng; //Capture mutable reference to rng into move closure instead of cloning it
            Mep::crossover((&a, &b), |x| (0..3).map(move |_| rng.gen_range::<usize>(0, x)).take(3))
        };
        //Ensure that rng was borrowed mutably
        assert!(rng.clone().gen_iter::<u32>().take(5).collect::<Vec<_>>() != old_rngs);

        assert_eq!(c.instructions, [0, 7, 5, 4, 2, 8, 5, 8, 4, 8]);
    }
}
