use core::ops::{Add, AddAssign, Mul, MulAssign};
use num_traits::{Float, One, Zero};

use crate::Deep;

/// `EmptyData` allows arithmetic operations on it despite containing nothing.
/// It pretends to be a number for deep learning purposes.
#[derive(Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EmptyData;

impl<F: Float> Deep<F> for EmptyData {
    fn map(&mut self, _f: impl Fn(F) -> F) {}
}

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
