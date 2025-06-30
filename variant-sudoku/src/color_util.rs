
pub fn color_ave2(a: (u8, u8, u8), b: (u8, u8, u8)) -> (u8, u8, u8) {
    let r = ((a.0 as f32)*0.5 + (b.0 as f32)*0.5) as u8;
    let g = ((a.1 as f32)*0.5 + (b.1 as f32)*0.5) as u8;
    let b = ((a.2 as f32)*0.5 + (b.2 as f32)*0.5) as u8;
    (r, g, b)
}

pub fn color_ave(colors: &Vec<(u8, u8, u8)>) -> (u8, u8, u8) {
    let sum = colors.iter().fold((0.0, 0.0, 0.0), |a, b| {
        (a.0 + b.0 as f32, a.1 + b.1 as f32, a.2 + b.2 as f32)
    });
    let n = colors.len() as f32;
    ((sum.0/n) as u8, (sum.1/n) as u8, (sum.2/n) as u8)
}

pub fn color_scale(c: (u8, u8, u8), ratio: f32) -> (u8, u8, u8) {
    color_lerp((0, 0, 0), c, ratio)
}

pub fn color_lerp(a: (u8, u8, u8), b: (u8, u8, u8), ratio: f32) -> (u8, u8, u8) {
    let (t, tc) = (ratio, 1.0 - ratio);
    let r = ((a.0 as f32)*tc + (b.0 as f32)*t) as u8;
    let g = ((a.1 as f32)*tc + (b.1 as f32)*t) as u8;
    let b = ((a.2 as f32)*tc + (b.2 as f32)*t) as u8;
    (r, g, b)
}

// Green is low, orange is high, pink is exact middle, if any.
pub fn color_polarity(min: u8, max: u8, v: u8) -> (u8, u8, u8) {
    let mid = ((min as f32) + (max as f32))/2.0;
    if (v as f32) < mid {
        (0, 200, 0)
    } else if (v as f32) > mid {
        (200, 130, 0)
    } else {
        (200, 0, 200)
    }
}

pub fn color_dist(a: (u8, u8, u8), b: (u8, u8, u8)) -> f32 {
    let dr = a.0 as i16 - b.0 as i16;
    let dg = a.1 as i16 - b.1 as i16;
    let db = a.2 as i16 - b.2 as i16;
    ((dr * dr + dg * dg + db * db) as f32).sqrt()
}

pub fn color_clamp_i32(c: (i32, i32, i32)) -> (u8, u8, u8) {
    (
        c.0.clamp(0, 255) as u8,
        c.1.clamp(0, 255) as u8,
        c.2.clamp(0, 255) as u8,
    )
}

pub fn color_clamp_f32(c: (f32, f32, f32)) -> (u8, u8, u8) {
    (
        c.0.clamp(0.0, 255.0) as u8,
        c.1.clamp(0.0, 255.0) as u8,
        c.2.clamp(0.0, 255.0) as u8,
    )
}

// Generate a palette of multiple similar colors. Uses the fibonacci sphere
// algorithm to distribute them evenly-ish on the surface of the sphere with
// the given radius, centered at the prototype.
pub fn color_fib_palette(prototype: (u8, u8, u8), n: usize, radius: f32) -> Vec<(u8, u8, u8)> {
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut palette = Vec::with_capacity(n);
    for i in 0..n {
        let i = i as f32;
        let n = n as f32;
        let theta = 2.0 * std::f32::consts::PI * i / phi;
        let z = 1.0 - (2.0 * i + 1.0) / n;
        let r = (1.0 - z * z).sqrt();
        let x = r * theta.cos();
        let y = r * theta.sin();
        let dx = (x * radius).round() as i32;
        let dy = (y * radius).round() as i32;
        let dz = (z * radius).round() as i32;
        let rgb = color_clamp_i32((
            prototype.0 as i32 + dx,
            prototype.1 as i32 + dy,
            prototype.2 as i32 + dz,
        ));
        palette.push(rgb);
    }
    let stride = (n / 3).max(1);
    let mut interleaved = Vec::with_capacity(n);
    for i in 0..stride {
        let mut j = 0;
        loop {
            let idx = i + j * stride;
            if idx >= n { break; }
            interleaved.push(palette[idx]);
            j += 1;
        }
    }
    interleaved
}
