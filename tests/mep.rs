extern crate gpi;
extern crate rand;
use rand::isaac::Isaac64Rng;
use rand::SeedableRng;
use rand::Rng;
use gpi::Mep;

#[test]
fn mep_new() {
    let mut state = 0..;
    let a: Mep<u32> = Mep::new(8, || state.next().unwrap());

    assert_eq!(a.instructions, (0..8).collect::<Vec<_>>());
}

#[test]
fn mep_crossover() {
    let mut rng = Isaac64Rng::from_seed(&[1, 2, 3, 4]);
    let len = 10;
    let a: Mep<u32> = Mep::new(len, || rng.next_u32() % 10);
    let b: Mep<u32> = Mep::new(len, || rng.next_u32() % 10);
    let c = Mep::crossover(&a, &b, 3, |x| (rng.next_u64() as usize) % x);

    assert_eq!(c.instructions, [0, 7, 5, 4, 2, 8, 5, 8, 4, 8]);
}
