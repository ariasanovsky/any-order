use faer::{stats::StandardNormalCol, Col, Mat};
use half::{bf16, f16};
use rand::{distributions::Distribution, rngs::StdRng, Rng, SeedableRng};

fn main() {
    let nbits = 24;
    const O: usize = 2;
    let ref mut rng = StdRng::seed_from_u64(234324);
    let mut stuff = Stuff::<O>::new(nbits, rng);
    let width = 10000;
    let target_error = stuff.bf16_error();
    // let target_error = stuff.f16_error();
    dbg!(target_error);
    let target_bits = ((1 << nbits) * 16) as f64;
    let width_bits = stuff.width_bits();
    for w in 1..=width {
        stuff.fill_matrices();
        let cut = stuff.cut(rng);
        let dot = stuff.t.transpose() * cut.blowup();
        let err = dot - cut.c;
        dbg!(cut.c, dot, err);
        stuff.update(&cut);
        let curr_error = stuff.error();
        let expected_width = expected_width(target_error, curr_error, w);
        let expected_bits = expected_width * width_bits as f64;
        // dbg!(curr_error, expected_width);
        let expected_over_target = expected_bits / target_bits;
        dbg!(w, expected_over_target);
    }
}

fn expected_width(target: f64, curr: f64, w: usize) -> f64 {
    let rho = curr.powf((w as f64).recip());
    target.ln() / rho.ln()
}

struct Stuff<const O: usize> {
    nbits: usize,
    nrowbits: usize,
    ncolbits: usize,
    ndims: usize,
    nrows: usize,
    ncols: usize,
    t: Col<f64>,
    init_norm: f64,
    mats: [Mat<f64>; O],
}

impl<const O: usize> Stuff<O> {
    fn new(nbits: usize, rng: &mut impl Rng) -> Self {
        assert!(nbits % O == 0);
        let nrowbits = nbits / O;
        let ncolbits = nrowbits * (O - 1);
        let nrows = 1 << nrowbits;
        let ncols = 1 << ncolbits;
        let ndims = 1 << nbits;
        let t = StandardNormalCol { nrows: ndims }.sample(rng);
        let init_norm = t.norm_l2();
        Self {
            nbits,
            nrowbits,
            ncolbits,
            ndims,
            nrows,
            ncols,
            t,
            init_norm,
            mats: core::array::from_fn(|_| Mat::zeros(nrows, ncols)),
        }
    }

    fn bf16_error(&self) -> f64 {
        Col::from_fn(self.ndims, |i| {
            let a = self.t[i];
            let b = bf16::from_f64(a);
            a - b.to_f64()
        }).norm_l2() / self.norm_l2()
    }

    fn f16_error(&self) -> f64 {
        Col::from_fn(self.ndims, |i| {
            let a = self.t[i];
            let b = f16::from_f64(a);
            a - b.to_f64()
        }).norm_l2() / self.norm_l2()
    }

    fn norm_l2(&self) -> f64 {
        self.t.norm_l2()
    }

    fn fill_matrices(&mut self) {
        for i in 0..self.ndims {
            let ti = self.t[i];
            let i = SubIndices::<O>::new(i, self.nrowbits);
            for axis in 0..O {
                let (row, col) = i.entry(axis, self.nrowbits);
                self.mats[axis][(row, col)] = ti;
            }
        }
    }

    fn cut(&self, rng: &mut impl Rng) -> Cut<O> {
        let mut cut = Cut::<O>::new(self.nrows, rng);
        loop {
            let mut improved = false;
            for axis in 0..O {
                improved |= cut.improve(axis, self);
            }
            if !improved {
                return cut
            }
        }
    }

    fn update(&mut self, cut: &Cut<O>) {
        let blowup = cut.blowup();
        let scale = cut.c / self.ndims as f64;
        let update = faer::scale(scale) * blowup;
        self.t -= update
    }

    fn error(&self) -> f64 {
        self.norm_l2() / self.init_norm
    }

    fn width_bits(&self) -> usize {
        64 + O * self.nrows
    }
}

struct SubIndices<const O: usize>([usize; O]);

impl<const O: usize> SubIndices<O> {
    fn new(i: usize, nbits: usize) -> Self {
        let mut i = i;
        let mask = (1 << nbits) - 1;
        let mut indices = [0; O];
        for axis in 0..O {
            indices[axis] = i & mask;
            i >>= nbits;
        }
        Self(indices)
    }

    fn entry(&self, axis: usize, nbits: usize) -> (usize, usize) {
        let row = self.0[axis];
        let mut col = 0;
        let mut offset = 0;
        for a in 0..O {
            if a != axis {
                col += self.0[a] << offset;
                offset += nbits;
            }
        }
        (row, col)
    }
}

struct Cut<const O: usize> {
    c: f64,
    s: [Col<f64>; O],
}

impl<const O: usize> Cut<O> {
    fn new(nrows: usize, rng: &mut impl Rng) -> Self {
        Self {
            c: f64::NEG_INFINITY,
            s: core::array::from_fn(|_| {
                Col::from_fn(nrows, |_| {
                    if rng.gen() {
                        1.0
                    } else {
                        -1.0
                    }
                })
            })
        }
    }

    fn improve(&mut self, axis: usize, stuff: &Stuff<O>) -> bool {
        let s: Col<f64> = self.blowup_along(axis);
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
        s_image.iter().zip(sa.as_slice_mut().iter_mut()).for_each(|(&si, sa)| {
            if si >= 0.0 && *sa == -1.0 {
                *sa = 1.0
            } else if si < 0.0 && *sa == 1.0 {
                *sa = -1.0
            }
        });
        improved
    }

    fn blowup_along(&self, axis: usize) -> Col<f64> {
        let mut s = Mat::identity(1, 1);
        for a in (0..O).rev() {
            if a != axis {
                s = s.kron(&self.s[a])
            }
        }
        assert!(s.ncols() == 1);
        s.col(0).to_owned()
    }

    fn blowup(&self) -> Col<f64> {
        let mut s = Mat::identity(1, 1);
        for a in (0..O).rev() {
            s = s.kron(&self.s[a])
        }
        assert!(s.ncols() == 1);
        s.col(0).to_owned()
    }
}
