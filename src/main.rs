use equator::assert;
use faer::{stats::StandardNormalCol, Col, Mat};
use half::{bf16, f16};
use rand::{distributions::Distribution, rngs::StdRng, Rng, SeedableRng};

pub fn main() {
    let dim = [25, 3, 4, 7];
    let total_dim = dim.iter().product::<usize>();

    let ref mut rng = StdRng::seed_from_u64(234324);
    let mut stuff = Stuff::new(dim, rng);
    let width = 10000;
    let target_error = stuff.bf16_error();
    let target_bits = (total_dim * 16) as f32;
    let width_bits = stuff.width_bits();
    for w in 1..=width {
        stuff.fill_matrices();
        let cut = stuff.cut(rng);
        let dot = stuff.t.transpose() * cut.blowup();
        let err = dot - cut.c;
        stuff.update(&cut);
        let curr_error = stuff.error();
        let expected_width = expected_width(target_error, curr_error, w);
        let expected_bits = expected_width * width_bits as f32;
        let expected_over_target = expected_bits / target_bits;
        dbg!(
            w,
            cut.c,
            dot,
            err,
            curr_error,
            target_error,
            expected_over_target,
        );
    }
}

fn expected_width(target: f32, curr: f32, w: usize) -> f32 {
    let rho = curr.powf((w as f32).recip());
    target.ln() / rho.ln()
}

struct Stuff<const O: usize> {
    dim: [usize; O],
    ndims: usize,
    t: Col<f32>,
    init_norm: f32,
    mats: [Mat<f32>; O],
}

impl<const O: usize> Stuff<O> {
    fn new(dim: [usize; O], rng: &mut impl Rng) -> Self {
        let ndims = dim.iter().product::<usize>();
        assert!(ndims != 0);

        let t = StandardNormalCol { nrows: ndims }.sample(rng);
        let init_norm = t.norm_l2();
        Self {
            dim,
            ndims,
            t,
            init_norm,
            mats: core::array::from_fn(|i| Mat::zeros(dim[i], ndims / dim[i])),
        }
    }

    fn bf16_error(&self) -> f32 {
        Col::from_fn(self.ndims, |i| {
            let a = self.t[i];
            let b = bf16::from_f32(a);
            a - b.to_f32()
        })
        .norm_l2()
            / self.norm_l2()
    }

    fn f16_error(&self) -> f32 {
        Col::from_fn(self.ndims, |i| {
            let a = self.t[i];
            let b = f16::from_f32(a);
            a - b.to_f32()
        })
        .norm_l2()
            / self.norm_l2()
    }

    fn norm_l2(&self) -> f32 {
        self.t.norm_l2()
    }

    fn fill_matrices(&mut self) {
        for i in 0..self.ndims {
            let ti = self.t[i];
            let i = SubIndices::new(i, self.dim);
            for axis in 0..O {
                let (row, col) = i.entry(axis);
                self.mats[axis][(row, col)] = ti;
            }
        }
    }

    fn cut(&self, rng: &mut impl Rng) -> Cut<O> {
        let mut cut = Cut::<O>::new(self.dim, rng);
        loop {
            let mut improved = false;
            for axis in 0..O {
                improved |= cut.improve(axis, self);
            }
            if !improved {
                return cut;
            }
        }
    }

    fn update(&mut self, cut: &Cut<O>) {
        let blowup = cut.blowup();
        let scale = cut.c / self.ndims as f32;
        let update = faer::scale(scale) * blowup;
        self.t -= update
    }

    fn error(&self) -> f32 {
        self.norm_l2() / self.init_norm
    }

    fn width_bits(&self) -> usize {
        32 + self.dim.iter().sum::<usize>()
    }
}

struct SubIndices<const O: usize> {
    basis: [usize; O],
    dim: [usize; O],
}

impl<const O: usize> SubIndices<O> {
    fn new(i: usize, dim: [usize; O]) -> Self {
        let mut i = i;
        let mut basis = [0usize; O];

        for axis in 0..O {
            let dim = dim[axis];
            basis[axis] = i % dim;
            i /= dim;
        }
        Self { basis, dim }
    }

    fn entry(&self, axis: usize) -> (usize, usize) {
        let row = self.basis[axis];
        let mut col = 0;
        let mut stride = 1usize;
        for a in 0..O {
            if a != axis {
                col += self.basis[a] * stride;
                stride *= self.dim[a];
            }
        }
        (row, col)
    }
}

struct Cut<const O: usize> {
    c: f32,
    s: [Col<f32>; O],
}

impl<const O: usize> Cut<O> {
    fn new(dim: [usize; O], rng: &mut impl Rng) -> Self {
        Self {
            c: f32::NEG_INFINITY,
            s: dim.map(|dim| Col::from_fn(dim, |_| if rng.gen() { 1.0 } else { -1.0 })),
        }
    }

    fn improve(&mut self, axis: usize, stuff: &Stuff<O>) -> bool {
        let s: Col<f32> = self.blowup_along(axis);
        let s_image = &stuff.mats[axis] * &s;
        let cut = s_image.norm_l1();
        // dbg!(cut);
        let improved = if cut > self.c {
            self.c = cut;
            true
        } else {
            false
        };
        let sa = &mut self.s[axis];
        s_image
            .iter()
            .zip(sa.as_slice_mut().iter_mut())
            .for_each(|(&si, sa)| *sa = f32::copysign(1.0, si));
        improved
    }

    fn blowup_along(&self, axis: usize) -> Col<f32> {
        let mut s = Mat::identity(1, 1);
        for a in (0..O).rev() {
            if a != axis {
                s = s.kron(&self.s[a])
            }
        }
        assert!(s.ncols() == 1);
        s.col(0).to_owned()
    }

    fn blowup(&self) -> Col<f32> {
        let mut s = Mat::identity(1, 1);
        for a in (0..O).rev() {
            s = s.kron(&self.s[a])
        }
        assert!(s.ncols() == 1);
        s.col(0).to_owned()
    }
}
