//! This module contains neural network implementations.
extern crate collenchyma as co;
extern crate collenchyma_nn as nn;
use self::co::backend::{Backend, BackendConfig};
use self::co::framework::IFramework;
use self::co::frameworks::{Cuda, Native};
use self::co::memory::MemoryType;
use self::co::tensor::SharedTensor;
use self::nn::*;
use super::{SISO, Learning};
