use core::ops::{Add, AddAssign, Mul, MulAssign};
use num_traits::{One, Zero};

/// `ChainData` derives several traits and attempts to allow chaining the IntoIterator
/// implementation over `&` and `&mut` to allow access to the data within.
/// This is critical to allow optimizers to perform per-weight gradient update rules.
#[derive(Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EmptyData;

impl Add for EmptyData {
    type Output = EmptyData;

    fn add(self, _rhs: Self) -> Self::Output {
        EmptyData
    }
}

impl Add<f32> for EmptyData {
    type Output = EmptyData;

    fn add(self, _rhs: f32) -> Self::Output {
        EmptyData
    }
}

impl Add<f64> for EmptyData {
    type Output = EmptyData;

    fn add(self, _rhs: f64) -> Self::Output {
        EmptyData
    }
}

impl AddAssign for EmptyData {
    fn add_assign(&mut self, _rhs: Self) {}
}

impl AddAssign<f32> for EmptyData {
    fn add_assign(&mut self, _rhs: f32) {}
}

impl AddAssign<f64> for EmptyData {
    fn add_assign(&mut self, _rhs: f64) {}
}

impl Zero for EmptyData {
    fn zero() -> Self {
        EmptyData
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl Mul for EmptyData {
    type Output = EmptyData;

    fn mul(self, _rhs: Self) -> Self::Output {
        EmptyData
    }
}

impl MulAssign for EmptyData {
    fn mul_assign(&mut self, _rhs: Self) {}
}

impl MulAssign<f32> for EmptyData {
    fn mul_assign(&mut self, _rhs: f32) {}
}

impl MulAssign<f64> for EmptyData {
    fn mul_assign(&mut self, _rhs: f64) {}
}

impl One for EmptyData {
    fn one() -> Self {
        EmptyData
    }
}
